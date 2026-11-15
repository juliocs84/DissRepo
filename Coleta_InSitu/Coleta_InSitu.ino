/*
  Coleta in situ - áudio contínuo (PDM, 16 kHz, 16 bits) + BME280 (0,5 Hz) no microSD
  Placa: Arduino Nano 33 BLE Sense (Arduino Mbed OS Nano Boards 4.6.0)

  Cada sessão cria uma pasta Snnn/ no cartão com:
    Annnn.WAV  áudio contínuo, arquivos de 10 min (mono, 16 kHz, PCM 16 bits)
    AMB.CSV    leituras do BME280 a cada 2 s
    EVT.CSV    marcações de início/fim de evento feitas pelo botão
    INFO.TXT   parâmetros da sessão
  A coluna "amostra" em AMB.CSV e EVT.CSV é o índice da amostra de áudio no fluxo gravado
  (amostra / 16000 = segundos desde o início da sessão), o que alinha exatamente
  áudio, variáveis ambientais e eventos, sem depender de relógio externo.

  Botão (D7 -> GND, sem resistor):
    toque curto  = marca início/fim de evento (alternados); LED amarelo aceso durante o evento
    3 s pressionado = encerra a sessão com segurança (fecha os arquivos); depois, desligue
  Buzzer, Wi-Fi e telemetria ficam desativados para não contaminar o áudio.
*/
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <PDM.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ---------- Pinos ----------
const uint8_t PIN_SD_CS        = 10;
const uint8_t PIN_BOTAO        = 7;   // botão entre D7 e GND
const uint8_t PIN_LED_EVENTO   = 2;   // amarelo: evento em andamento
const uint8_t PIN_LED_GRAVANDO = 5;   // verde: pisca enquanto grava
const uint8_t PIN_BUZZER       = 6;   // mantido em nível baixo

// ---------- Parâmetros ----------
const char*    VERSAO          = "coleta-1.0";
const uint32_t TAXA            = 16000;           // Hz
const uint8_t  GANHO_PDM       = 20;              // fixo e registrado em INFO.TXT (sem controle automático)
const uint32_t AMOSTRAS_ARQ    = TAXA * 600UL;    // 10 min por arquivo WAV
const unsigned long PERIODO_BME_MS   = 2000;
const unsigned long PERIODO_FLUSH_MS = 5000;
const unsigned long PERIODO_TELA_MS  = 1000;
const unsigned long DESCARTE_INICIAL_MS = 600;    // acomodação do microfone
const unsigned long BOTAO_LONGO_MS   = 3000;
const unsigned long DEBOUNCE_MS      = 50;

// ---------- Buffer circular de áudio (produtor: interrupção do PDM; consumidor: loop) ----------
const uint32_t RING = 32768;                      // amostras (~2 s); potência de 2, múltiplo de 256
int16_t ring[RING];
volatile uint32_t head = 0;                       // total de amostras escritas no buffer
volatile uint32_t tail = 0;                       // total de amostras retiradas para o cartão
volatile uint32_t perdidas = 0;                   // amostras descartadas por buffer cheio
int16_t pdmTmp[512];

void aoReceberPDM() {
  int bytes = PDM.available();
  if (bytes > (int)sizeof(pdmTmp)) bytes = sizeof(pdmTmp);
  PDM.read(pdmTmp, bytes);
  int n = bytes / 2;
  uint32_t h = head;
  for (int i = 0; i < n; i++) {
    if (h - tail >= RING) { perdidas += (n - i); break; }
    ring[h & (RING - 1)] = pdmTmp[i];
    h++;
  }
  head = h;
}

// ---------- Estado ----------
Adafruit_BME280  bme;
Adafruit_SSD1306 display(128, 64, &Wire, -1);
bool oledOk = false, bmeOk = false;

char pasta[5];                // "S001"
char nomeWav[16];             // "S001/A0001.WAV"
File wav;
uint16_t numArquivo = 0;
uint32_t amostrasArquivo = 0; // no arquivo WAV atual
uint32_t amostrasTotal = 0;   // gravadas na sessão (até ~74 h em 32 bits)
uint16_t numEvento = 0;
bool emEvento = false, encerrada = false;
float tUlt = NAN, uUlt = NAN, pUlt = NAN;
unsigned long inicioSessao = 0, proxBme = 0, proxFlush = 0, proxTela = 0;

// =====================================================================
// Utilitários de arquivo
// =====================================================================
void put16(uint8_t* p, uint16_t v) { p[0] = v; p[1] = v >> 8; }
void put32(uint8_t* p, uint32_t v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; }

void escreverCabecalhoWav(File& f, uint32_t nAmostras) {
  uint32_t dados = nAmostras * 2;
  uint8_t h[44];
  memcpy(h, "RIFF", 4);      put32(h + 4, 36 + dados);
  memcpy(h + 8, "WAVEfmt ", 8);
  put32(h + 16, 16);         put16(h + 20, 1);          put16(h + 22, 1);   // PCM, mono
  put32(h + 24, TAXA);       put32(h + 28, TAXA * 2);   put16(h + 32, 2);   put16(h + 34, 16);
  memcpy(h + 36, "data", 4); put32(h + 40, dados);
  f.write(h, 44);
}

void anexarLinha(const char* arquivo, const String& linha) {
  char caminho[16];
  snprintf(caminho, sizeof(caminho), "%s/%s", pasta, arquivo);
  File f = SD.open(caminho, FILE_WRITE);
  if (f) { f.println(linha); f.close(); }
}

void abrirWav() {
  numArquivo++;
  snprintf(nomeWav, sizeof(nomeWav), "%s/A%04u.WAV", pasta, numArquivo);
  wav = SD.open(nomeWav, FILE_WRITE);
  escreverCabecalhoWav(wav, 0);          // tamanho corrigido ao fechar o arquivo
  amostrasArquivo = 0;
}

void fecharWav() {
  wav.close();
  File f = SD.open(nomeWav, O_READ | O_WRITE);  // sem O_APPEND, para regravar o cabeçalho
  if (f) { f.seek(0); escreverCabecalhoWav(f, amostrasArquivo); f.close(); }
}

// Grava no cartão o áudio acumulado, em blocos de 512 bytes
void drenarAudio() {
  while (head - tail >= 256) {
    uint32_t idx = tail & (RING - 1);              // múltiplo de 256: o bloco nunca cruza o fim do buffer
    wav.write((const uint8_t*)&ring[idx], 512);
    tail += 256;
    amostrasArquivo += 256;
    amostrasTotal += 256;
    if (amostrasArquivo >= AMOSTRAS_ARQ) { fecharWav(); abrirWav(); }
  }
}

// Índice da amostra correspondente ao instante atual no fluxo gravado
uint32_t amostraAgora() { return amostrasTotal + (head - tail); }

// =====================================================================
// Botão, eventos e encerramento
// =====================================================================
void marcarEvento(unsigned long tMs, uint32_t amostra) {
  emEvento = !emEvento;
  if (emEvento) numEvento++;
  digitalWrite(PIN_LED_EVENTO, emEvento);
  anexarLinha("EVT.CSV", String(numEvento) + "," + (emEvento ? "inicio" : "fim") + "," + tMs + "," + amostra);
}

void encerrarSessao() {
  drenarAudio();
  PDM.end();
  drenarAudio();
  fecharWav();
  if (emEvento) anexarLinha("EVT.CSV", String(numEvento) + ",fim," + millis() + "," + amostrasTotal);
  anexarLinha("EVT.CSV", String("0,encerramento,") + millis() + "," + amostrasTotal);
  anexarLinha("INFO.TXT", String("amostras_gravadas=") + amostrasTotal + "\namostras_perdidas=" + perdidas +
              "\narquivos_wav=" + numArquivo);
  encerrada = true;
  digitalWrite(PIN_LED_GRAVANDO, LOW);
  digitalWrite(PIN_LED_EVENTO, LOW);
  if (oledOk) {
    display.clearDisplay(); display.setTextSize(2);
    display.setCursor(4, 8);  display.print("SESSAO");
    display.setCursor(4, 28); display.print("ENCERRADA");
    display.setTextSize(1);
    display.setCursor(4, 52); display.print("Pode desligar");
    display.display();
  }
}

void atualizarBotao(unsigned long agora) {
  static bool estavel = HIGH, lido = HIGH, longoTratado = false;
  static unsigned long tMudanca = 0, tPress = 0;
  static uint32_t amostraPress = 0;
  bool l = digitalRead(PIN_BOTAO);
  if (l != lido) { lido = l; tMudanca = agora; }
  if (agora - tMudanca > DEBOUNCE_MS && l != estavel) {
    estavel = l;
    if (estavel == LOW) { tPress = agora; amostraPress = amostraAgora(); longoTratado = false; }
    else if (!longoTratado) marcarEvento(tPress, amostraPress);   // toque curto: registra o instante do aperto
  }
  if (estavel == LOW && !longoTratado && agora - tPress >= BOTAO_LONGO_MS) {
    longoTratado = true;
    encerrarSessao();
  }
}

// =====================================================================
// BME280 e tela
// =====================================================================
void lerBME(unsigned long agora) {
  uint32_t amostra = amostraAgora();
  if (!bmeOk) bmeOk = bme.begin(0x76) || bme.begin(0x77);
  if (bmeOk) {
    tUlt = bme.readTemperature(); uUlt = bme.readHumidity(); pUlt = bme.readPressure() / 100.0F;
    if (isnan(tUlt) || isnan(uUlt) || isnan(pUlt)) bmeOk = false;
  }
  String linha = String(agora) + "," + amostra + ",";
  if (bmeOk) linha += String(tUlt, 2) + "," + String(uUlt, 2) + "," + String(pUlt, 2);
  else       { linha += ",,"; tUlt = uUlt = pUlt = NAN; }
  linha += String(",") + perdidas;
  anexarLinha("AMB.CSV", linha);
}

void atualizarTela(unsigned long agora) {
  digitalWrite(PIN_LED_GRAVANDO, (agora / 500) % 2);   // pisca 1 Hz
  digitalWrite(LEDR, perdidas ? LOW : HIGH);             // RGB vermelho: houve perda de amostras
  if (!oledOk) return;
  unsigned long s = (agora - inicioSessao) / 1000;
  char l1[22], l2[22], l3[22], l4[22];
  snprintf(l1, sizeof(l1), "COLETA %s  A%04u", pasta, numArquivo);
  snprintf(l2, sizeof(l2), "%02lu:%02lu:%02lu", s / 3600, (s / 60) % 60, s % 60);
  snprintf(l3, sizeof(l3), "Evento %u %s", numEvento, emEvento ? "[EM CURSO]" : "");
  snprintf(l4, sizeof(l4), "Perdas: %lu", (unsigned long)perdidas);
  display.clearDisplay(); display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1); display.setCursor(0, 0);  display.print(l1);
  display.setTextSize(2); display.setCursor(0, 12); display.print(l2);
  display.setTextSize(1); display.setCursor(0, 32); display.print(l3);
  display.setCursor(0, 42);
  if (!isnan(tUlt)) { display.print(tUlt, 1); display.print("C  "); display.print(uUlt, 0); display.print("%"); }
  else display.print("BME280 sem leitura");
  display.setCursor(0, 54); display.print(l4);
  display.display();
}

void erroFatal(const char* msg) {
  Serial.println(msg);
  if (oledOk) {
    display.clearDisplay(); display.setTextSize(1); display.setCursor(0, 24);
    display.print(msg); display.display();
  }
  while (true) { digitalWrite(LEDR, LOW); delay(250); digitalWrite(LEDR, HIGH); delay(250); }
}

// =====================================================================
void setup() {
  Serial.begin(115200);                       // apenas depuração; não espera o USB (operação a bateria)
  pinMode(PIN_BUZZER, OUTPUT);       digitalWrite(PIN_BUZZER, LOW);
  pinMode(PIN_LED_EVENTO, OUTPUT);   pinMode(PIN_LED_GRAVANDO, OUTPUT);
  pinMode(3, OUTPUT); pinMode(4, OUTPUT);     // LEDs Crítico e Falha apagados
  pinMode(PIN_BOTAO, INPUT_PULLUP);
  pinMode(LEDR, OUTPUT); digitalWrite(LEDR, HIGH);
  pinMode(LEDG, OUTPUT); digitalWrite(LEDG, HIGH);
  pinMode(LEDB, OUTPUT); digitalWrite(LEDB, HIGH);

  Serial1.begin(115200);
  Serial1.print("AT+CWQAP\r\n");              // desconecta o ESP-01 do Wi-Fi (menos consumo e ruído)

  oledOk = display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  bmeOk = bme.begin(0x76) || bme.begin(0x77);
  if (!SD.begin(PIN_SD_CS)) erroFatal("ERRO: microSD");

  for (int i = 1; i <= 999; i++) {            // próxima pasta de sessão livre
    snprintf(pasta, sizeof(pasta), "S%03d", i);
    if (!SD.exists(pasta)) break;
  }
  if (!SD.mkdir(pasta)) erroFatal("ERRO: criar pasta");

  anexarLinha("INFO.TXT", String("versao=") + VERSAO + "\ntaxa_hz=" + TAXA + "\nbits=16\ncanais=1" +
              "\nganho_pdm=" + GANHO_PDM + "\namostras_por_arquivo=" + AMOSTRAS_ARQ +
              "\nperiodo_bme_ms=" + PERIODO_BME_MS + "\nbme280=" + (bmeOk ? "ok" : "ausente"));
  anexarLinha("AMB.CSV", "t_ms,amostra,temp,umid,press,perdidas");
  anexarLinha("EVT.CSV", "evento,tipo,t_ms,amostra");

  PDM.onReceive(aoReceberPDM);
  PDM.setGain(GANHO_PDM);
  if (!PDM.begin(1, TAXA)) erroFatal("ERRO: microfone PDM");

  delay(DESCARTE_INICIAL_MS);                 // descarta a acomodação inicial do microfone
  noInterrupts();
  tail = head - (head % 256);                 // mantém o índice de leitura alinhado a 256
  perdidas = 0;
  interrupts();

  abrirWav();
  inicioSessao = millis();
  proxBme = inicioSessao;
  proxFlush = inicioSessao + PERIODO_FLUSH_MS;
  proxTela = inicioSessao;
  anexarLinha("EVT.CSV", String("0,inicio_sessao,") + inicioSessao + ",0");
}

void loop() {
  if (encerrada) return;
  drenarAudio();

  unsigned long agora = millis();
  atualizarBotao(agora);
  if (encerrada) return;

  if ((long)(agora - proxBme) >= 0)   { proxBme += PERIODO_BME_MS;     lerBME(agora); }
  if ((long)(agora - proxFlush) >= 0) { proxFlush += PERIODO_FLUSH_MS; wav.flush(); }
  if ((long)(agora - proxTela) >= 0)  { proxTela += PERIODO_TELA_MS;   atualizarTela(agora); }
}
