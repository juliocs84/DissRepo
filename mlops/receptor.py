"""
Receptor de telemetria do nó TinyML (Nano 33 BLE Sense + ESP-01).

Recebe POST /telemetria com JSON enviado pelo firmware, grava no SQLite
(telemetria.db) e mantém estatísticas de entrega da sessão.
O dashboard (dashboard.py) lê o mesmo banco.

Alertas (RF5): quando a classe ENTRA em Crítico ou Falha (ou passa de Crítico para Falha),
o receptor grava o evento na tabela "alertas" e o encaminha ao webhook do n8n, que envia
e-mail e Telegram e devolve a confirmação. O encaminhamento roda em segundo plano:
a resposta ao nó sai na hora, e uma falha do n8n não afeta a telemetria.

Uso:  python receptor.py                       (porta 8000, banco telemetria.db)
      python receptor.py 8001 simulacao.db     (outra porta e outro banco)
Webhook do n8n (variável de ambiente; sem ela, os alertas são só registrados localmente):
      PowerShell:  $env:N8N_WEBHOOK = "http://localhost:5678/webhook/alerta-tinyml"
Parar: Ctrl+C  (mostra o resumo da sessão)

Formato esperado do JSON (campos ausentes são gravados como nulos):
{"t": 182777, "classe": "Normal", "p": [0.91, 0.05, 0.03, 0.01],
 "temp": 23.9, "umid": 45.2, "press": 1013.2, "lat_ms": 12.4, "ram_kb": 46.0}
"""
import json
import os
import queue
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORTA = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
BANCO = sys.argv[2] if len(sys.argv) > 2 else "telemetria.db"
WEBHOOK = os.environ.get("N8N_WEBHOOK", "").strip()
INTERVALO_MS = 10_000          # período de envio do firmware (para estimar perdas)
CLASSES = ["Normal", "Atenção", "Crítico", "Falha"]
NIVEL = {c: i for i, c in enumerate(CLASSES)}
TENTATIVAS, ESPERA_S, TIMEOUT_S = 3, 5, 30

ESQUEMA = """
CREATE TABLE IF NOT EXISTS telemetria (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    recebido_em TEXT    NOT NULL,   -- data/hora do host (ISO 8601)
    t_ms        INTEGER,            -- millis() do Nano no envio
    classe      TEXT,               -- saída do modelo (ou da regra, enquanto não houver modelo)
    classe_regra TEXT,              -- classe pela regra de limiar (Figura 3)
    p_normal    REAL, p_atencao REAL, p_critico REAL, p_falha REAL,
    temp        REAL,               -- °C
    umid        REAL,               -- %UR
    press       REAL,               -- hPa
    lat_ms      REAL,               -- latência de inferência informada pelo firmware
    ram_kb      REAL,
    origem_ip   TEXT,
    json_bruto  TEXT
);
CREATE INDEX IF NOT EXISTS idx_recebido ON telemetria(recebido_em);
CREATE TABLE IF NOT EXISTS alertas (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em       TEXT NOT NULL,      -- quando o receptor detectou a mudança de classe
    telemetria_id   INTEGER,            -- mensagem que disparou o alerta
    t_ms            INTEGER,
    classe          TEXT,
    classe_anterior TEXT,
    temp REAL, umid REAL, press REAL, p_anom REAL,
    simulado        INTEGER DEFAULT 0,  -- 1 = mensagem do simular_alertas.py
    status          TEXT,               -- pendente | entregue | parcial | falhou | sem_webhook
    tentativas      INTEGER DEFAULT 0,
    confirmado_em   TEXT,
    email           TEXT,               -- ok | falha (resposta do n8n)
    telegram        TEXT,               -- ok | falha (resposta do n8n)
    execucao_n8n    TEXT,               -- id da execução no n8n (rastreabilidade)
    erro            TEXT
);
"""

trava = threading.Lock()
conexao = sqlite3.connect(BANCO, check_same_thread=False)
conexao.execute("PRAGMA journal_mode=WAL")   # permite leitura simultânea pelo Streamlit
conexao.executescript(ESQUEMA)
colunas = [c[1] for c in conexao.execute("PRAGMA table_info(telemetria)")]
if "classe_regra" not in colunas:                      # migração de bancos antigos
    conexao.execute("ALTER TABLE telemetria ADD COLUMN classe_regra TEXT")
conexao.commit()

estat = {"recebidas": 0, "invalidas": 0, "perdidas_estimadas": 0,
         "ultimo_t": None, "inicio": datetime.now(), "alertas": 0}
estado = {"classe": None}          # última classe recebida (para detectar a entrada em alerta)
fila_alertas: "queue.Queue[int]" = queue.Queue()


def agora():
    return datetime.now().isoformat(timespec="seconds")


def numero(d, chave):
    v = d.get(chave)
    return float(v) if isinstance(v, (int, float)) else None


def gravar(dados: dict, ip: str, bruto: str) -> int:
    p = dados.get("p") if isinstance(dados.get("p"), list) else [None] * 4
    p = (p + [None] * 4)[:4]
    linha = (
        agora(),
        int(dados["t"]) if isinstance(dados.get("t"), (int, float)) else None,
        dados.get("classe"),
        dados.get("classe_regra"),
        *[float(x) if isinstance(x, (int, float)) else None for x in p],
        numero(dados, "temp"), numero(dados, "umid"), numero(dados, "press"),
        numero(dados, "lat_ms"), numero(dados, "ram_kb"),
        ip, bruto,
    )
    with trava:
        cur = conexao.execute(
            "INSERT INTO telemetria (recebido_em, t_ms, classe, classe_regra, p_normal, p_atencao, "
            "p_critico, p_falha, temp, umid, press, lat_ms, ram_kb, origem_ip, json_bruto) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", linha)
        conexao.commit()
        return cur.lastrowid


def atualizar_estatisticas(t):
    estat["recebidas"] += 1
    ultimo = estat["ultimo_t"]
    if isinstance(t, (int, float)):
        if ultimo is not None and t > ultimo:
            faltando = round((t - ultimo) / INTERVALO_MS) - 1
            if faltando > 0:
                estat["perdidas_estimadas"] += faltando
        estat["ultimo_t"] = t   # se t < último, o Nano reiniciou: recomeça a contagem


# ---------------------------------------------------------------- alertas
def verificar_alerta(dados: dict, tel_id: int):
    """Alerta quando a classe entra em Crítico/Falha ou sobe de Crítico para Falha."""
    nova = dados.get("classe")
    anterior = estado["classe"]
    estado["classe"] = nova
    if nova not in ("Crítico", "Falha"):
        return
    if anterior is not None and NIVEL.get(anterior, -1) >= NIVEL[nova]:
        return                                   # continua no mesmo alerta (ou desceu de Falha para Crítico)
    with trava:
        cur = conexao.execute(
            "INSERT INTO alertas (criado_em, telemetria_id, t_ms, classe, classe_anterior, temp, umid, press, "
            "p_anom, simulado, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (agora(), tel_id, dados.get("t"), nova, anterior, numero(dados, "temp"), numero(dados, "umid"),
             numero(dados, "press"), numero(dados, "p_anom"), int(bool(dados.get("simulado"))), "pendente" if WEBHOOK else "sem_webhook"))
        conexao.commit()
        alerta_id = cur.lastrowid
    estat["alertas"] += 1
    print(f"{datetime.now():%H:%M:%S}  ALERTA #{alerta_id}: {anterior} -> {nova}", flush=True)
    if WEBHOOK:
        fila_alertas.put(alerta_id)


def encaminhar(alerta_id: int):
    with trava:
        conexao.row_factory = sqlite3.Row
        a = dict(conexao.execute("SELECT * FROM alertas WHERE id = ?", (alerta_id,)).fetchone())
        conexao.row_factory = None
    corpo = json.dumps({
        "alerta_id": a["id"], "classe": a["classe"], "classe_anterior": a["classe_anterior"],
        "detectado_em": a["criado_em"], "t_ms": a["t_ms"], "temp": a["temp"], "umid": a["umid"],
        "press": a["press"], "p_anom": a["p_anom"], "simulado": bool(a["simulado"]),
    }).encode("utf-8")
    erro = None
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            req = urllib.request.Request(WEBHOOK, data=corpo, method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                resp = json.loads(r.read().decode("utf-8") or "{}")
            email, tg = resp.get("email"), resp.get("telegram")
            status = "entregue" if email == "ok" and tg == "ok" else "parcial" if "ok" in (email, tg) else "falhou"
            with trava:
                conexao.execute(
                    "UPDATE alertas SET status=?, tentativas=?, confirmado_em=?, email=?, telegram=?, "
                    "execucao_n8n=?, erro=NULL WHERE id=?",
                    (status, tentativa, agora(), email, tg, str(resp.get("execucao", "")), alerta_id))
                conexao.commit()
            print(f"{datetime.now():%H:%M:%S}  alerta #{alerta_id} -> n8n: {status} "
                  f"(e-mail {email}, Telegram {tg})", flush=True)
            return
        except (urllib.error.URLError, OSError, ValueError) as e:
            erro = str(e)
            time.sleep(ESPERA_S * tentativa)
    with trava:
        conexao.execute("UPDATE alertas SET status='falhou', tentativas=?, erro=? WHERE id=?",
                        (TENTATIVAS, erro, alerta_id))
        conexao.commit()
    print(f"{datetime.now():%H:%M:%S}  alerta #{alerta_id}: n8n indisponível ({erro})", flush=True)


def trabalhador():
    while True:
        encaminhar(fila_alertas.get())


# ---------------------------------------------------------------- HTTP
class Receptor(BaseHTTPRequestHandler):
    def _responder(self, codigo, texto):
        corpo = texto.encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(corpo)

    def do_POST(self):
        if self.path != "/telemetria":
            return self._responder(404, "rota desconhecida")
        n = int(self.headers.get("Content-Length", 0))
        bruto = self.rfile.read(n).decode("utf-8", errors="replace")
        try:
            dados = json.loads(bruto)
            if not isinstance(dados, dict):
                raise ValueError("JSON não é um objeto")
        except ValueError as e:
            estat["invalidas"] += 1
            print(f"{datetime.now():%H:%M:%S}  JSON inválido: {e} | {bruto!r}", flush=True)
            return self._responder(400, "JSON invalido")
        tel_id = gravar(dados, self.client_address[0], bruto)
        atualizar_estatisticas(dados.get("t"))
        self._responder(200, "OK")                  # responde ao nó antes de tratar o alerta
        print(f"{datetime.now():%H:%M:%S}  #{estat['recebidas']:>5}  "
              f"classe={dados.get('classe')}  temp={dados.get('temp')}  "
              f"umid={dados.get('umid')}  press={dados.get('press')}", flush=True)
        verificar_alerta(dados, tel_id)

    def do_GET(self):   # teste pelo navegador: http://IP:PORTA
        with trava:
            total = conexao.execute("SELECT COUNT(*) FROM telemetria").fetchone()[0]
        self._responder(200, f"Receptor ativo - {total} registros em {BANCO}")

    def log_message(self, *args):
        pass


def resumo():
    dur = (datetime.now() - estat["inicio"]).total_seconds()
    rec, perd = estat["recebidas"], estat["perdidas_estimadas"]
    esperadas = rec + perd
    print("\n===== Resumo da sessão =====")
    print(f"Duração:               {dur / 60:.1f} min")
    print(f"Mensagens recebidas:   {rec}")
    print(f"Perdas estimadas:      {perd}  (saltos no campo t)")
    if esperadas:
        print(f"Taxa de entrega:       {100 * rec / esperadas:.1f}%")
    print(f"JSON inválidos:        {estat['invalidas']}")
    print(f"Alertas detectados:    {estat['alertas']}")
    print(f"Banco:                 {BANCO}")


if __name__ == "__main__":
    # o encaminhamento ao n8n roda numa thread separada, para não atrasar a resposta ao nó
    threading.Thread(target=trabalhador, daemon=True).start()
    servidor = ThreadingHTTPServer(("0.0.0.0", PORTA), Receptor)
    print(f"Receptor ouvindo na porta {PORTA} - gravando em {BANCO} (Ctrl+C para parar)", flush=True)
    print(f"Alertas: {'encaminhados a ' + WEBHOOK if WEBHOOK else 'só registro local (defina N8N_WEBHOOK)'}",
          flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servidor.server_close()
        resumo()
        conexao.close()
