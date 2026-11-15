"""
Dashboard de telemetria do nó TinyML.
Uso: streamlit run dashboard.py
Lê o banco telemetria.db gravado pelo receptor.py e atualiza a cada 5 s.
"""
import json
import sqlite3
import time
from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import streamlit as st

BANCO = "telemetria.db"
ESCALA_MIN = {"temp": 1.0, "umid": 2.0, "press": 1.0}   # amplitude mínima do eixo Y (evita exagerar ruído do sensor)
LACUNA_S = 30          # intervalo sem dados que interrompe as linhas dos gráficos
CORES = {"Normal": "#2e7d32", "Atenção": "#f9a825", "Crítico": "#ef6c00", "Falha": "#c62828"}

st.set_page_config(page_title="Monitoramento TinyML", layout="wide")
st.title("Monitoramento multimodal do data center")

janela_h = st.sidebar.slider("Janela dos gráficos (horas)", 1, 24, 2)
atualizar = st.sidebar.checkbox("Atualização automática (5 s)", value=True)


@st.cache_data(ttl=4)
def carregar(horas):
    desde = (datetime.now() - timedelta(hours=horas)).isoformat(timespec="seconds")
    with sqlite3.connect(BANCO, timeout=5) as c:
        df = pd.read_sql_query(
            "SELECT * FROM telemetria WHERE recebido_em >= ? ORDER BY id", c, params=(desde,))
        total = c.execute("SELECT COUNT(*) FROM telemetria").fetchone()[0]
    df["recebido_em"] = pd.to_datetime(df["recebido_em"])
    # campos do firmware com inferência que não têm coluna própria no banco (ficam no JSON bruto)
    extra = df["json_bruto"].apply(lambda j: json.loads(j) if isinstance(j, str) and j.startswith("{") else {})
    df["p_anom"] = extra.apply(lambda d: d.get("p_anom")).astype(float)
    df["fonte_termica"] = extra.apply(lambda d: d.get("fonte_termica"))
    # novo trecho quando há intervalo > 30 s sem dados (evita ligar períodos distintos com uma reta)
    df["trecho"] = (df["recebido_em"].diff().dt.total_seconds() > LACUNA_S).cumsum()
    return df, total


try:
    df, total = carregar(janela_h)
except Exception as e:
    st.warning(f"Banco ainda não disponível ({e}). Inicie o receptor.py e aguarde o primeiro envio.")
    st.stop()

if df.empty:
    st.info("Nenhum registro na janela selecionada.")
else:
    ult = df.iloc[-1]
    cor = CORES.get(ult["classe"], "#555")
    idade = (datetime.now() - ult["recebido_em"]).total_seconds()
    c1, c2, c3 = st.columns([1, 1, 1])
    c1.markdown(f"<div style='background:{cor};color:white;padding:14px;border-radius:8px;"
                f"text-align:center;font-size:26px;font-weight:700'>{ult['classe']}</div>",
                unsafe_allow_html=True)
    c2.metric("Último envio", f"há {idade:.0f} s")
    if pd.notna(ult["p_anom"]):
        c3.metric("Anomalia acústica (média de 10 s)", f"{100 * ult['p_anom']:.0f} %")
    else:
        c3.metric("Anomalia acústica (média de 10 s)", "—", help="CNN desligada: sem linha de base (envie CAL) ou em calibração")
    if idade > 30:
        st.error("Sem telemetria há mais de 30 s: verifique o nó e a rede.")

    def linha(col, titulo):
        dados = df.dropna(subset=[col])
        if dados.empty:
            return None
        lo, hi = dados[col].min(), dados[col].max()
        falta = ESCALA_MIN[col] - (hi - lo)
        if falta > 0:                                  # garante amplitude mínima centrada nos dados
            lo, hi = lo - falta / 2, hi + falta / 2
        duracao = (dados["recebido_em"].max() - dados["recebido_em"].min()).total_seconds()
        fmt = "%H:%M:%S" if duracao < 600 else "%H:%M"
        return (alt.Chart(dados).mark_line(point=len(dados) < 60)
                .encode(x=alt.X("recebido_em:T", title=None, axis=alt.Axis(format=fmt, tickCount=5)),
                        detail="trecho:N",
                        y=alt.Y(f"{col}:Q", title=titulo, scale=alt.Scale(domain=[lo, hi], nice=False),
                                axis=alt.Axis(format=".1f")),
                        tooltip=[alt.Tooltip("recebido_em:T", title="Horário", format="%H:%M:%S"),
                                 alt.Tooltip(f"{col}:Q", title=titulo, format=".2f")])
                .properties(height=220))

    def mostrar(coluna, grafico):
        if grafico is None:
            coluna.info("Sem dados nesta janela.")
        else:
            coluna.altair_chart(grafico, width="stretch")

    g1, g2, g3 = st.columns(3)               # cada valor atual fica acima do seu gráfico
    g1.metric("Temperatura", f"{ult['temp']:.1f} °C" if pd.notna(ult["temp"]) else "—")
    g2.metric("Umidade", f"{ult['umid']:.1f} %" if pd.notna(ult["umid"]) else "—")
    g3.metric("Pressão", f"{ult['press']:.1f} hPa" if pd.notna(ult["press"]) else "—")
    mostrar(g1, linha("temp", "Temperatura (°C)"))
    mostrar(g2, linha("umid", "Umidade (%)"))
    mostrar(g3, linha("press", "Pressão (hPa)"))

    def barras(dados, titulo_y):
        return (alt.Chart(dados).mark_bar()
                .encode(x=alt.X("classe:N", sort=list(CORES), title=None, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("valor:Q", title=titulo_y),
                        color=alt.Color("classe:N", scale=alt.Scale(domain=list(CORES), range=list(CORES.values())),
                                        legend=None))
                .properties(height=220))

    b1, b2 = st.columns(2)
    probs = ult[["p_normal", "p_atencao", "p_critico", "p_falha"]]
    b1.subheader("Probabilidades do ramo ambiental (LSTM)")
    if not probs.notna().any():
        b1.info("Disponível quando a janela do LSTM tiver 30 passos (5 min após ligar); até lá vale a regra de limiares.")
    else:
        b1.altair_chart(barras(pd.DataFrame({"classe": list(CORES), "valor": probs.values.astype(float)}),
                               "Probabilidade"), width="stretch")
    b2.subheader("Distribuição das classes na janela")
    cont = df["classe"].value_counts().reindex(list(CORES), fill_value=0)
    b2.altair_chart(barras(pd.DataFrame({"classe": cont.index, "valor": cont.values}), "Classificações"),
                    width="stretch")

    st.subheader("Histórico (últimas 100 classificações)")
    colunas = ["recebido_em", "classe", "classe_regra", "temp", "umid", "press", "p_anom", "lat_ms", "t_ms"]
    tabela = df.tail(100).iloc[::-1][[c for c in colunas if c in df.columns]]
    tabela = tabela.dropna(axis=1, how="all")              # oculta colunas ainda sem dados
    st.dataframe(tabela, width="stretch", hide_index=True,
                 column_config={
                     "recebido_em": st.column_config.DatetimeColumn("Recebido em", format="DD/MM/YYYY HH:mm:ss"),
                     "classe": "Classe", "classe_regra": "Classe (regra)",
                     "temp": st.column_config.NumberColumn("Temp. (°C)", format="%.2f"),
                     "umid": st.column_config.NumberColumn("Umid. (%)", format="%.1f"),
                     "press": st.column_config.NumberColumn("Pressão (hPa)", format="%.1f"),
                     "p_anom": st.column_config.NumberColumn("Anomalia acústica", format="%.2f"),
                     "lat_ms": st.column_config.NumberColumn("Latência (ms)", format="%.1f"),
                     "t_ms": st.column_config.NumberColumn("t do nó (ms)", format="%d"),
                 })
    legenda = f"{total} registros no banco · {len(df)} na janela"
    if df["lat_ms"].notna().any():
        legenda += f" · latência média informada: {df['lat_ms'].mean():.1f} ms"
    st.caption(legenda)

if atualizar:
    time.sleep(5)
    st.rerun()
