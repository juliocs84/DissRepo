/*
  Monitoramento multimodal baseado em TinyML - firmware principal (com inferência embarcada)
  Placa: Arduino Nano 33 BLE Sense (Arduino Mbed OS Nano Boards 4.6.0)

  Ciclo (agendado por millis(), sem delay() no loop):
    - contínuo: áudio do PDM (16 kHz) em buffer circular; a cada 10 ms um quadro do espectrograma
      log-mel é calculado e quantizado direto na entrada da CNN; a cada 1 s (98 quadros) executa-se
      a CNN 2D int8 (passo 2 na 1ª convolução), e a probabilidade de anomalia entra na média das últimas 10 janelas;
    - a cada 2 s: leitura do BME280, gravação no microSD (telem.csv), atualização do OLED;
    - a cada 10 s: média das leituras -> novo passo da janela de 30 passos do LSTM (float32);
      fusão (Figura 3) do nível térmico do LSTM com o nível acústico da CNN -> LEDs, buzzer
      e envio HTTP ao receptor (ESP-01, comandos AT), com as probabilidades e a latência.
  Enquanto a janela do LSTM não completa 30 passos (5 min), o nível térmico vem da regra de limiares.
  Sem linha de base acústica (base.csv no microSD), a CNN fica desligada: envie "CAL" pelo Monitor
  Serial com o ventilador em operação normal (cerca de 5 min) para calibrar.

  Erros de dispositivo (sensor, cartão, rede, display, microfone, modelos) são sinalizados no LED RGB
  da placa e no OLED, separados da classe "Falha", que se refere ao estado do ambiente monitorado.

  Ligações:
    D0/D1  Serial1 -> ESP-01 (TX do ESP em D0, RX do ESP em D1)
    A4/A5  I2C     -> BME280 (0x76) e OLED SSD1306 (0x3C)
    D10-D13 SPI    -> microSD (CS em D10)
    D2..D5 LEDs (Atenção, Crítico, Falha, Normal), D6 buzzer
    Microfone MP34DT05 integrado (PDM)

  Arquivos desta pasta: secrets.h; espectro_mel.h, preprocessamento.h, inferencia.h, lstm_c.h;
  gerados pelo notebook 05: modelo_cnn.h, pesos_lstm.h, ops_modelos.h, parametros_modelos.h, tabelas_mel.h
  (modelo_lstm.h só é necessário com USAR_LSTM_C = 0).
*/
#define USAR_CMSIS_DSP 0                 // 1 = arm_rfft_fast_f32 (biblioteca Arduino_CMSIS-DSP)
#define TFLM_CHIRALE   0                 // 1 = Chirale_TensorFlowLite em vez de Harvard_TinyMLx
#define USAR_LSTM_C    1                 // 1 = LSTM direto em C (lstm_c.h); 0 = LSTM no TFLite Micro
#define ARENA_CNN_BYTES  (16 * 1024)     // ajuste com o Teste_Inferencia: arena usada + ~2 KB
#define ARENA_LSTM_BYTES (10 * 1024)     // só usada com USAR_LSTM_C = 0

#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <PDM.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "secrets.h"            // WIFI_SSID, WIFI_PASS, HOST, PORT
#include "espectro_mel.h"
#include "preprocessamento.h"
#include "inferencia.h"

// ---------- Pinos ----------
const uint8_t PIN_LED_ATENCAO = 2;
const uint8_t PIN_LED_CRITICO = 3;
const uint8_t PIN_LED_FALHA   = 4;
const uint8_t PIN_LED_NORMAL  = 5;
const uint8_t PIN_BUZZER      = 6;
const uint8_t PIN_SD_CS       = 10;

// ---------- Tempos ----------
const unsigned long PERIODO_LEITURA_MS   = 2000;   // leitura do BME280
const unsigned long PERIODO_JANELA_MS    = 10000;  // classificação, registro e envio
const unsigned long PERIODO_RECONEXAO_MS = 60000;  // nova tentativa de Wi-Fi
const uint8_t       FALHAS_PARA_RECONECTAR = 3;    // envios seguidos com falha
// false: aciona com tone() a 2,7 kHz, que faz o elemento piezo ressoar e soa mais alto com o pino a 3,3 V;
// true: nível digital contínuo (mais baixo neste buzzer, que fica subalimentado pelo pino).
const bool          BUZZER_ATIVO = false;
const unsigned int  FREQ_BUZZER = 2700;            // Hz, ressonância do buzzer (ajustada em teste)
const bool          ECO_AT = false;                // true: espelha as respostas do ESP no Monitor Serial

// ---------- Ramo acústico ----------
const uint8_t  GANHO_PDM = 20;                     // igual ao firmware de coleta in situ
const bool     USAR_ACUSTICO_NA_FUSAO = true;      // false: CNN só informativa (nível acústico = 0)
const float    LIMIAR_ANOMALIA = 0.5f;             // média das 10 últimas p(anômalo) >= limiar -> nível 1
const uint8_t  N_MEDIA_CNN = 10;                   // janelas de 1 s na média (seção 3.3.2)
const uint16_t CAL_JANELAS = 300;                  // janelas de 1 s na calibração (~5 min)
const char*    ARQUIVO_BASE = "base.csv";          // linha de base: média e desvio dos 40 filtros mel

// ---------- Classes (Figura 3) ----------
enum Classe : uint8_t { NORMAL = 0, ATENCAO = 1, CRITICO = 2, FALHA = 3 };
const char* NOME_CLASSE[] = {"Normal", "Atenção", "Crítico", "Falha"};   // JSON (UTF-8)
const char* NOME_ASCII[]  = {"Normal", "Atencao", "Critico", "Falha"};   // OLED e CSV
const uint8_t LED_CLASSE[] = {PIN_LED_NORMAL, PIN_LED_ATENCAO, PIN_LED_CRITICO, PIN_LED_FALHA};
const float LIMITES_TERMICOS[] = {27.0, 32.0, 35.0};   // °C; o limite pertence à faixa inferior

// Nível térmico pela regra: >27 °C = 1, >32 °C = 2, >35 °C = 3
uint8_t nivelTermico(float t) {
  if (t > LIMITES_TERMICOS[2]) return 3;
  if (t > LIMITES_TERMICOS[1]) return 2;
  if (t > LIMITES_TERMICOS[0]) return 1;
  return 0;
}

// Fusão por regra (Figura 3): nível térmico (LSTM) x nível acústico (CNN)
Classe classePorRegra(uint8_t nTerm, uint8_t nAcust) {
  if (nTerm == 3 || nAcust == 3) return FALHA;
  if (nTerm == 2 && nAcust >= 1) return FALHA;
  return (Classe)max(nTerm, nAcust);
}

// ---------- Erros de dispositivo (máscara de bits) ----------
const uint8_t ERRO_BME  = 1;
const uint8_t ERRO_SD   = 2;
const uint8_t ERRO_REDE = 4;
const uint8_t ERRO_OLED = 8;
const uint8_t ERRO_MIC  = 16;
const uint8_t ERRO_IA   = 32;
uint8_t erros = 0;

// ---------- Objetos e estado ----------
Adafruit_BME280  bme;
Adafruit_SSD1306 display(128, 64, &Wire, -1);
const char* ARQUIVO_CSV = "telem.csv";   // nome 8.3; não mistura com o data.csv antigo

Classe classeAtual = NORMAL;
bool   classeValida = false;              // false até a primeira janela com leitura válida
float  tUlt = NAN, uUlt = NAN, pUlt = NAN; // última leitura (OLED)

struct Janela { float somaT, somaU, somaP; uint8_t n; } jan = {0, 0, 0, 0};

unsigned long proxLeitura = 0, proxJanela = 0, proxReconexao = 0;
uint8_t falhasEnvio = 0;
uint32_t enviosOk = 0, enviosFalha = 0;

// ---------- Estado do ramo ambiental (LSTM) ----------
float histT[LSTM_PASSOS], histU[LSTM_PASSOS], histP[LSTM_PASSOS];
uint8_t nHist = 0;                        // passos válidos na janela (até 30)
float probLSTM[LSTM_CLASSES];
bool  lstmValido = false;                 // true quando o nível térmico veio do LSTM
float latLSTMms = 0;

// ---------- Estado do ramo acústico (PDM + espectrograma + CNN) ----------
const uint32_t RING_N = 4096;             // amostras (256 ms); potência de 2 (cobre a inferência da CNN)
int16_t ring[RING_N];
volatile uint32_t escritos = 0;           // total de amostras recebidas do PDM
int16_t pdmTmp[256];

mel::Espectro espectro;
float logmel[mel::NMEL];
uint32_t inicioJanelaAc = 0;              // índice absoluto da 1ª amostra da janela de 1 s
uint8_t  quadroAtual = 0;                 // 0..97
bool     micOk = false, iaOk = false;

float baseMedia[mel::NMEL], baseInvDesvio[mel::NMEL];
bool  baseOk = false;

bool     calibrando = false;              // modo de calibração da linha de base
uint16_t calJanelas = 0;
uint32_t calN = 0;
double   calMedia[mel::NMEL], calM2[mel::NMEL];

float   histAc[N_MEDIA_CNN];              // p(anômalo) das últimas janelas de 1 s
uint8_t nHistAc = 0, posHistAc = 0;
uint16_t janelasAcCiclo = 0, perdasAcCiclo = 0;   // por ciclo de 10 s
float   latCNNms = 0;

// =====================================================================
// Sensores
// =====================================================================
bool iniciarBME() {
  if (bme.begin(0x76) || bme.begin(0x77)) { erros &= ~ERRO_BME; return true; }
  erros |= ERRO_BME;
  return false;
}

bool lerBME(float& t, float& u, float& p) {
  if ((erros & ERRO_BME) && !iniciarBME()) return false;   // tenta recuperar o sensor
  t = bme.readTemperature();
  u = bme.readHumidity();
  p = bme.readPressure() / 100.0F;                          // Pa -> hPa
  bool ok = !isnan(t) && !isnan(u) && !isnan(p) && t > -40 && t < 85 && p > 300 && p < 1100;
  if (ok) erros &= ~ERRO_BME; else erros |= ERRO_BME;
  return ok;
}

// =====================================================================
// Áudio: PDM -> buffer circular (interrupção) -> quadros log-mel (loop) -> CNN a cada 1 s
// =====================================================================
void aoReceberPDM() {
  int bytes = PDM.available();
  if (bytes > (int)sizeof(pdmTmp)) bytes = sizeof(pdmTmp);
  PDM.read(pdmTmp, bytes);
  int n = bytes / 2;
  uint32_t w = escritos;
  for (int i = 0; i < n; i++) ring[(w + i) & (RING_N - 1)] = pdmTmp[i];
  escritos = w + n;                                   // amostras antigas são sobrescritas
}

void reiniciarJanelaAc() {
  inicioJanelaAc = escritos;
  quadroAtual = 0;
}

double mediaAc() {
  if (nHistAc == 0) return NAN;
  double s = 0;
  for (uint8_t i = 0; i < nHistAc; i++) s += histAc[i];
  return s / nHistAc;
}

uint8_t nivelAcustico() {
  if (!USAR_ACUSTICO_NA_FUSAO || calibrando || !baseOk || nHistAc == 0) return 0;
  return mediaAc() >= LIMIAR_ANOMALIA ? 1 : 0;        // CNN binária: anomalia = nível 1 (Atenção)
}

void acumularCalibracao(const float* lm) {          // média e variância por filtro (Welford)
  calN++;
  for (int m = 0; m < mel::NMEL; m++) {
    double d = lm[m] - calMedia[m];
    calMedia[m] += d / calN;
    calM2[m] += d * (lm[m] - calMedia[m]);
  }
}

void janelaAcCompleta() {
  janelasAcCiclo++;
  if (calibrando) {
    if (++calJanelas >= CAL_JANELAS) concluirCalibracao();
    return;
  }
  if (!baseOk || !iaOk) return;
  uint32_t t0 = micros();
  float p;
  if (!ia::executarCNN(p)) { erros |= ERRO_IA; return; }
  latCNNms = (micros() - t0) / 1000.0f;
  histAc[posHistAc] = p;
  posHistAc = (posHistAc + 1) % N_MEDIA_CNN;
  if (nHistAc < N_MEDIA_CNN) nHistAc++;
}

// Calcula os quadros já disponíveis (no máximo 8 por chamada, para não atrasar o restante do loop).
void processarAudio() {
  if (!micOk || (!calibrando && (!baseOk || !iaOk))) return;
  for (uint8_t k = 0; k < 8; k++) {
    uint32_t ini = inicioJanelaAc + (uint32_t)quadroAtual * mel::SALTO;
    uint32_t e = escritos;
    if ((int32_t)(e - (ini + mel::QUADRO)) < 0) return;   // o quadro ainda não chegou
    if (e - ini > RING_N) {                           // atrasou e as amostras foram sobrescritas
      perdasAcCiclo++;
      reiniciarJanelaAc();
      return;
    }
    espectro.quadroAnel(ring, RING_N - 1, ini, logmel);
    if (escritos - ini > RING_N) {                    // sobrescritas durante o cálculo: descarta a janela
      perdasAcCiclo++;
      reiniciarJanelaAc();
      return;
    }

    if (calibrando) acumularCalibracao(logmel);
    else {
      int8_t* entrada = ia::entradaCNN() + quadroAtual * mel::NMEL;
      for (int m = 0; m < mel::NMEL; m++) entrada[m] = prep::quantizarCNN(logmel[m], baseMedia[m], baseInvDesvio[m]);
    }

    if (++quadroAtual == mel::NQUADROS) {             // 98 quadros: janela de 1 s completa
      inicioJanelaAc += mel::JANELA;                  // próxima janela, sem sobreposição
      quadroAtual = 0;
      janelaAcCompleta();
      return;
    }
  }
}

// ---------- Linha de base (microSD) ----------
void definirBase(const float* media, const float* desvio) {
  for (int m = 0; m < mel::NMEL; m++) { baseMedia[m] = media[m]; baseInvDesvio[m] = 1.0f / desvio[m]; }
  baseOk = true;
}

bool carregarBase() {
  if (erros & ERRO_SD) return false;
  File f = SD.open(ARQUIVO_BASE, FILE_READ);
  if (!f) return false;
  float media[mel::NMEL], desvio[mel::NMEL];
  int lidos = 0;
  f.readStringUntil('\n');                            // cabeçalho
  while (f.available() && lidos < mel::NMEL) {
    String s = f.readStringUntil('\n');
    int a = s.indexOf(','), b = s.indexOf(',', a + 1);
    if (a < 0 || b < 0) continue;
    media[lidos] = s.substring(a + 1, b).toFloat();
    desvio[lidos] = s.substring(b + 1).toFloat();
    if (desvio[lidos] <= 0) break;
    lidos++;
  }
  f.close();
  if (lidos != mel::NMEL) return false;
  definirBase(media, desvio);
  return true;
}

bool salvarBase() {
  if ((erros & ERRO_SD) && !iniciarSD()) return false;
  if (SD.exists(ARQUIVO_BASE)) SD.remove(ARQUIVO_BASE);
  File f = SD.open(ARQUIVO_BASE, FILE_WRITE);
  if (!f) { erros |= ERRO_SD; return false; }
  f.println("filtro,media,desvio");
  for (int m = 0; m < mel::NMEL; m++) {
    f.print(m); f.print(',');
    f.print(baseMedia[m], 6); f.print(',');
    f.println(1.0f / baseInvDesvio[m], 6);
  }
  f.close();
  return true;
}

void iniciarCalibracao() {
  if (!micOk) { Serial.println(">> Calibração impossível: microfone indisponível"); return; }
  calibrando = true;
  calJanelas = 0; calN = 0;
  for (int m = 0; m < mel::NMEL; m++) { calMedia[m] = 0; calM2[m] = 0; }
  nHistAc = 0; posHistAc = 0;
  reiniciarJanelaAc();
  Serial.print(">> Calibração da linha de base: mantenha o ventilador em operação NORMAL por cerca de ");
  Serial.print(CAL_JANELAS / 60.0, 1); Serial.println(" min");
}

void concluirCalibracao() {
  float media[mel::NMEL], desvio[mel::NMEL];
  for (int m = 0; m < mel::NMEL; m++) {
    media[m] = calMedia[m];
    desvio[m] = sqrt(calM2[m] / calN) + 1e-6;         // desvio populacional + 1e-6, como no notebook 03 v2
  }
  definirBase(media, desvio);
  calibrando = false;
  bool salvo = salvarBase();
  reiniciarJanelaAc();
  Serial.print(">> Linha de base concluída: "); Serial.print(calN); Serial.print(" quadros; ");
  Serial.println(salvo ? "salva em base.csv" : "NÃO salva (microSD indisponível; vale até desligar)");
}

void imprimirBase() {
  if (!baseOk) { Serial.println(">> Sem linha de base"); return; }
  Serial.println("filtro,media,desvio");
  for (int m = 0; m < mel::NMEL; m++) {
    Serial.print(m); Serial.print(','); Serial.print(baseMedia[m], 4); Serial.print(',');
    Serial.println(1.0f / baseInvDesvio[m], 4);
  }
}

void verificarComandos() {                            // Monitor Serial: CAL, BASE
  static String cmd;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      cmd.trim(); cmd.toUpperCase();
      if (cmd == "CAL") iniciarCalibracao();
      else if (cmd == "BASE") imprimirBase();
      else if (cmd.length()) Serial.println(">> Comandos: CAL (calibrar a linha de base), BASE (mostrar)");
      cmd = "";
    } else if (cmd.length() < 16) cmd += c;
  }
}

// =====================================================================
// Ramo ambiental: janela de 30 passos de 10 s -> LSTM
// =====================================================================
void adicionarPassoAmbiental(float t, float u, float p) {
  if (nHist == LSTM_PASSOS) {                         // desliza a janela
    memmove(histT, histT + 1, sizeof(float) * (LSTM_PASSOS - 1));
    memmove(histU, histU + 1, sizeof(float) * (LSTM_PASSOS - 1));
    memmove(histP, histP + 1, sizeof(float) * (LSTM_PASSOS - 1));
    nHist--;
  }
  histT[nHist] = t; histU[nHist] = u; histP[nHist] = p;
  nHist++;
}

// Retorna o nível térmico: LSTM com a janela completa; regra de limiares enquanto ela se forma.
uint8_t nivelTermicoAtual(float t) {
  lstmValido = false;
  if (!iaOk || nHist < LSTM_PASSOS) return nivelTermico(t);
  float x[LSTM_PASSOS][LSTM_ATRIBUTOS];
  uint32_t t0 = micros();
  prep::atributosLSTM(histT, histU, histP, x);
  if (!ia::executarLSTM(x, probLSTM)) { erros |= ERRO_IA; return nivelTermico(t); }
  latLSTMms = (micros() - t0) / 1000.0f;
  lstmValido = true;
  uint8_t arg = 0;
  for (uint8_t c = 1; c < LSTM_CLASSES; c++) if (probLSTM[c] > probLSTM[arg]) arg = c;
  return arg;
}

// =====================================================================
// Interface: LEDs, buzzer e OLED
// =====================================================================
void atualizarLEDs() {
  for (uint8_t i = 0; i < 4; i++)
    digitalWrite(LED_CLASSE[i], classeValida && classeAtual == i);
}

// LED RGB da placa (ativo em nível baixo): vermelho piscando = erro de dispositivo
void atualizarLedErro(unsigned long agora) {
  bool aceso = erros && ((agora / 500) % 2 == 0);
  digitalWrite(LEDR, aceso ? LOW : HIGH);
}

// Padrões não bloqueantes: bipe (ms), pausa (ms), número de bipes, período do ciclo (ms)
struct Padrao { uint16_t bipe, pausa; uint8_t n; uint16_t ciclo; };
const Padrao PADRAO[] = {
  {0,   0,   0, 0},     // Normal: silêncio
  {0,   0,   0, 0},     // Atenção: apenas LED
  {200, 0,   1, 2000},  // Crítico: pulsos de 200 ms a cada 2 s (seção 7.2.3)
  {100, 0,   1, 500}    // Falha: pulsos de 100 ms a cada 500 ms (seção 7.2.3)
};

// Liga/desliga o buzzer. No núcleo Mbed, noTone() pode deixar o pino em estado residual
// (apito baixo contínuo); por isso o desligamento sempre devolve o pino ao modo digital em nível baixo.
void buzzerLiga() {
  if (BUZZER_ATIVO) digitalWrite(PIN_BUZZER, HIGH);
  else tone(PIN_BUZZER, FREQ_BUZZER);
}

void buzzerDesliga() {
  if (!BUZZER_ATIVO) noTone(PIN_BUZZER);
  pinMode(PIN_BUZZER, OUTPUT);                       // devolve o pino ao modo digital
  digitalWrite(PIN_BUZZER, LOW);
}

void atualizarBuzzer(unsigned long agora) {
  static unsigned long inicioCiclo = 0;
  static bool tocando = false;
  const Padrao& pd = PADRAO[classeValida ? classeAtual : NORMAL];
  bool deveTocar = false;
  if (pd.n > 0) {
    if (agora - inicioCiclo >= pd.ciclo) inicioCiclo = agora;
    unsigned long dt = agora - inicioCiclo;
    unsigned long passo = pd.bipe + pd.pausa;
    deveTocar = dt < passo * pd.n && (dt % passo) < pd.bipe;
  }
  if (deveTocar && !tocando)      { buzzerLiga();    tocando = true; }
  else if (!deveTocar && tocando) { buzzerDesliga(); tocando = false; }
}

void bipe(unsigned int ms) {      // apenas na inicialização
  buzzerLiga();
  delay(ms);
  buzzerDesliga();
}

void desenharGauge(float valor, float minVal, float maxVal) {
  const int cx = 64, cy = 48, r = 32;

  for (int g = 180; g <= 360; g += 3) {                      // arco
    float a = g * DEG_TO_RAD;
    display.drawPixel(cx + cos(a) * r, cy + sin(a) * r, SSD1306_WHITE);
  }
  for (uint8_t i = 0; i < 3; i++) {                          // marcas dos limites 27/32/35 °C
    float a = PI + PI * (LIMITES_TERMICOS[i] - minVal) / (maxVal - minVal);
    display.drawLine(cx + cos(a) * (r - 4), cy + sin(a) * (r - 4),
                     cx + cos(a) * (r + 1), cy + sin(a) * (r + 1), SSD1306_WHITE);
  }
  // ponteiro: interpolação em ponto flutuante (map() trabalha com inteiros)
  float frac = (constrain(valor, minVal, maxVal) - minVal) / (maxVal - minVal);
  float a = PI + PI * frac;
  display.drawLine(cx, cy, cx + cos(a) * (r - 6), cy + sin(a) * (r - 6), SSD1306_WHITE);
  display.fillCircle(cx, cy, 3, SSD1306_WHITE);
}

// Coluna da esquerda: estado dos dois ramos de IA
void desenharEstadoIA() {
  display.setCursor(0, 14);                                  // ramo acústico
  if (erros & ERRO_MIC)      display.print("MIC!");
  else if (calibrando)       display.print("CAL");
  else if (erros & ERRO_IA)  display.print("IA!");
  else if (!baseOk)          display.print("BASE?");
  else if (nHistAc == 0)     display.print("A --");
  else { display.print("A"); display.print((int)round(100 * mediaAc())); display.print("%"); }

  display.setCursor(0, 24);                                  // ramo ambiental
  if (calibrando) { display.print(100 * calJanelas / CAL_JANELAS); display.print("%"); }
  else if (lstmValido) display.print("LSTM");
  else { display.print("R"); display.print(nHist); }         // regra enquanto a janela se forma
}

void atualizarTela() {
  if (erros & ERRO_OLED) return;
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  display.setCursor(13, 0);
  display.print("ESTADO DO SISTEMA");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  if (!isnan(tUlt)) {
    desenharGauge(tUlt, 20.0, 40.0);
  } else {
    display.setCursor(31, 30);
    display.print("SEM SENSOR");
  }
  desenharEstadoIA();

  // Rodapé: temperatura | erros de dispositivo | classe
  display.setCursor(0, 56);
  if (!isnan(tUlt)) { display.print(tUlt, 1); display.print("C"); }
  else display.print("--.-C");

  display.setCursor(38, 56);
  if (erros & ERRO_SD)   display.print("SD! ");
  if (erros & ERRO_REDE) display.print("NET!");

  const char* rotulo = classeValida ? NOME_ASCII[classeAtual] : "...";
  display.setCursor(128 - strlen(rotulo) * 6, 56);
  display.print(rotulo);

  display.display();
}

// =====================================================================
// Registro no microSD
// =====================================================================
bool iniciarSD() {
  if (SD.begin(PIN_SD_CS)) {
    erros &= ~ERRO_SD;
    if (!SD.exists(ARQUIVO_CSV)) {
      File f = SD.open(ARQUIVO_CSV, FILE_WRITE);
      if (f) { f.println("t_ms,temp,umid,press,classe,erros"); f.close(); }
    }
    return true;
  }
  erros |= ERRO_SD;
  return false;
}

void gravarSD(const String& linha) {
  if ((erros & ERRO_SD) && !iniciarSD()) return;   // tenta remontar o cartão
  File f = SD.open(ARQUIVO_CSV, FILE_WRITE);
  if (f) { f.println(linha); f.close(); }
  else erros |= ERRO_SD;
}

// =====================================================================
// Rede: ESP-01 com firmware AT 1.3.0
// =====================================================================
bool waitFor(const char* expect, unsigned long timeout) {
  String buf;
  buf.reserve(128);
  unsigned long t0 = millis();
  while (millis() - t0 < timeout) {
    while (Serial1.available()) {
      char c = Serial1.read();
      if (ECO_AT) Serial.write(c);
      buf += c;
      if (buf.length() > 120) buf.remove(0, 60);   // mantém só o final da resposta
      if (buf.indexOf(expect) >= 0) return true;
      if (buf.indexOf("ERROR") >= 0 || buf.indexOf("FAIL") >= 0) return false;
    }
  }
  return false;
}

bool sendCmd(const String& cmd, const char* expect, unsigned long timeout) {
  while (Serial1.available()) Serial1.read();
  Serial1.print(cmd);
  Serial1.print("\r\n");
  return waitFor(expect, timeout);
}

bool wifiConnect(uint8_t tentativas) {
  sendCmd("AT+CWMODE=1", "OK", 2000);
  sendCmd("AT+CIPMUX=0", "OK", 2000);
  for (uint8_t i = 1; i <= tentativas; i++) {
    Serial.print(">> Wi-Fi: tentativa "); Serial.println(i);
    if (sendCmd(String("AT+CWJAP=\"") + WIFI_SSID + "\",\"" + WIFI_PASS + "\"", "WIFI GOT IP", 20000)) {
      waitFor("OK", 3000);
      erros &= ~ERRO_REDE;
      return true;
    }
  }
  erros |= ERRO_REDE;
  return false;
}

bool httpPost(const char* path, const String& json) {
  String req = String("POST ") + path + " HTTP/1.1\r\n"
               "Host: " + HOST + "\r\n"
               "Content-Type: application/json\r\n"
               "Content-Length: " + String(json.length()) + "\r\n"
               "Connection: close\r\n\r\n" + json;

  sendCmd("AT+CIPCLOSE", "OK", 300);                 // ERROR aqui é normal
  if (!sendCmd(String("AT+CIPSTART=\"TCP\",\"") + HOST + "\"," + String(PORT), "OK", 5000)) return false;
  if (!sendCmd("AT+CIPSEND=" + String(req.length()), ">", 2000)) return false;
  Serial1.print(req);
  bool ok = waitFor("SEND OK", 3000);
  waitFor("CLOSED", 1500);
  return ok;
}

void enviarTelemetria(const String& json, unsigned long agora) {
  if ((erros & ERRO_REDE) && falhasEnvio >= FALHAS_PARA_RECONECTAR) {
    if ((long)(agora - proxReconexao) < 0) return;   // aguarda a próxima janela de reconexão
    proxReconexao = agora + PERIODO_RECONEXAO_MS;
    if (!wifiConnect(1)) return;
    falhasEnvio = 0;
  }
  if (httpPost("/telemetria", json)) {
    enviosOk++;
    falhasEnvio = 0;
    erros &= ~ERRO_REDE;
  } else {
    enviosFalha++;
    if (++falhasEnvio >= FALHAS_PARA_RECONECTAR) {
      erros |= ERRO_REDE;
      proxReconexao = millis();                      // reconecta já no próximo ciclo
    }
  }
}

// =====================================================================
// Ciclos
// =====================================================================
void cicloLeitura() {
  float t, u, p;
  bool ok = lerBME(t, u, p);
  if (ok) {
    tUlt = t; uUlt = u; pUlt = p;
    jan.somaT += t; jan.somaU += u; jan.somaP += p; jan.n++;
  } else {
    tUlt = uUlt = pUlt = NAN;
  }

  // Registro a 0,5 Hz: leitura bruta + classe vigente (decidida no último ciclo de 10 s)
  String linha = String(millis()) + ",";
  if (ok) linha += String(t, 2) + "," + String(u, 2) + "," + String(p, 2) + ",";
  else    linha += ",,,";
  linha += String(classeValida ? NOME_ASCII[classeAtual] : "") + "," + erros;
  gravarSD(linha);

  atualizarTela();
}

void cicloJanela(unsigned long agora) {
  String json, csv;
  uint8_t nAc = nivelAcustico();
  double pAc = mediaAc();
  bool acValido = !calibrando && baseOk && iaOk && nHistAc > 0;

  if (jan.n > 0) {
    float t = jan.somaT / jan.n, u = jan.somaU / jan.n, p = jan.somaP / jan.n;
    adicionarPassoAmbiental(t, u, p);
    uint8_t nTerm = nivelTermicoAtual(t);
    classeAtual = classePorRegra(nTerm, nAc);
    Classe regra = classePorRegra(nivelTermico(t), 0);  // referência: só limiares de temperatura
    classeValida = true;

    json = String("{\"t\":") + agora +
           ",\"classe\":\"" + NOME_CLASSE[classeAtual] + "\"" +
           ",\"classe_regra\":\"" + NOME_CLASSE[regra] + "\"" +
           ",\"temp\":" + String(t, 2) + ",\"umid\":" + String(u, 2) + ",\"press\":" + String(p, 2) +
           ",\"n\":" + jan.n;
    if (lstmValido) {
      json += ",\"p\":[";
      for (uint8_t c = 0; c < LSTM_CLASSES; c++) json += String(probLSTM[c], 4) + (c < 3 ? "," : "]");
    }
    if (acValido) json += ",\"p_anom\":" + String(pAc, 4) + ",\"nivel_ac\":" + nAc;
    json += ",\"janelas_ac\":" + String(janelasAcCiclo) + ",\"perdas_ac\":" + perdasAcCiclo;
    if (lstmValido || acValido)
      json += ",\"lat_ms\":" + String((lstmValido ? latLSTMms : 0) + (acValido ? latCNNms : 0), 1);
    json += ",\"fonte_termica\":\"" + String(lstmValido ? "lstm" : "regra") + "\",\"erros\":" + erros + "}";

    csv = String(agora) + "," + String(t, 2) + "," + String(u, 2) + "," + String(p, 2) + "," +
          jan.n + "," + NOME_ASCII[classeAtual] + "," + erros + "," +
          (acValido ? String(pAc, 3) : String("")) + "," + (lstmValido ? "lstm" : "regra");
  } else {
    classeValida = false;                              // sem leitura: não é a classe Falha
    nHist = 0;                                         // a janela do LSTM recomeça
    json = String("{\"t\":") + agora + ",\"classe\":\"Sem leitura\",\"n\":0,\"erros\":" + erros + "}";
    csv = String(agora) + ",,,,0,Sem leitura," + erros + ",,";
  }
  jan = {0, 0, 0, 0};
  janelasAcCiclo = 0;
  perdasAcCiclo = 0;

  atualizarLEDs();
  Serial.println(csv);                                 // linha CSV para depuração/coleta via USB
  enviarTelemetria(json, agora);
}

// =====================================================================
void setup() {
  NRF_NVMC->ICACHECNF = NVMC_ICACHECNF_CACHEEN_Msk;   // cache de instruções do nRF52840 (desligado após o reset)
  Serial.begin(115200);
  Serial1.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 3000);             // não trava sem USB

  for (uint8_t i = 0; i < 4; i++) pinMode(LED_CLASSE[i], OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  digitalWrite(PIN_BUZZER, LOW);
  pinMode(LEDR, OUTPUT); digitalWrite(LEDR, HIGH);     // LED RGB é ativo em nível baixo
  pinMode(LEDG, OUTPUT); digitalWrite(LEDG, HIGH);
  pinMode(LEDB, OUTPUT); digitalWrite(LEDB, HIGH);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) erros |= ERRO_OLED;
  if (!(erros & ERRO_OLED)) {
    display.clearDisplay(); display.setTextSize(1); display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);  display.print("Iniciando...");
    display.setCursor(0, 16); display.print("Modelos e Wi-Fi");
    display.display();
  }
  if (!iniciarBME()) Serial.println(">> BME280 não encontrado (0x76/0x77)");
  if (!iniciarSD())  Serial.println(">> microSD indisponível; seguindo sem registro local");

  // Modelos (TensorFlow Lite Micro) e espectrograma
  const char* erroIA = ia::iniciar();
  iaOk = (erroIA == nullptr) && espectro.iniciar();
  if (!iaOk) { erros |= ERRO_IA; Serial.print(">> Modelos não inicializados: "); Serial.println(erroIA ? erroIA : "FFT"); }
  else {
    Serial.print(">> Modelos prontos | arena CNN "); Serial.print(ia::arenaUsadaCNN() / 1024.0, 1);
    Serial.print(" KB, LSTM "); Serial.print(ia::arenaUsadaLSTM() / 1024.0, 1);
    Serial.print(" KB | FFT: "); Serial.println(espectro.backend());
  }
  if (carregarBase()) Serial.println(">> Linha de base acústica carregada (base.csv)");
  else Serial.println(">> Sem linha de base acústica: CNN desligada. Envie CAL com o ventilador em operação normal.");

  // Microfone
  PDM.onReceive(aoReceberPDM);
  PDM.setGain(GANHO_PDM);
  micOk = PDM.begin(1, mel::FS);
  if (!micOk) { erros |= ERRO_MIC; Serial.println(">> Microfone PDM não inicializado"); }

  delay(1000);                                          // ESP-01 inicializando
  if (sendCmd("AT", "OK", 1000) && wifiConnect(3)) Serial.println(">> Wi-Fi conectado");
  else { erros |= ERRO_REDE; falhasEnvio = FALHAS_PARA_RECONECTAR; Serial.println(">> Sem rede; nova tentativa em 60 s"); }

  bipe(150);
  Serial.println("t_ms,temp,umid,press,n,classe,erros,p_anom,fonte_termica");

  unsigned long agora = millis();
  proxLeitura = agora;
  proxJanela = agora + PERIODO_JANELA_MS;
  proxReconexao = agora + PERIODO_RECONEXAO_MS;
  reiniciarJanelaAc();
}

void loop() {
  processarAudio();
  verificarComandos();

  unsigned long agora = millis();
  if ((long)(agora - proxLeitura) >= 0) {
    proxLeitura += PERIODO_LEITURA_MS;                 // agenda fixa: sem deriva acumulada
    cicloLeitura();
  }
  if ((long)(agora - proxJanela) >= 0) {
    proxJanela += PERIODO_JANELA_MS;
    cicloJanela(agora);
  }
  // após um bloqueio longo (ex.: reconexão Wi-Fi), retoma a agenda sem rajada de ciclos atrasados
  agora = millis();
  if ((long)(agora - proxLeitura) > (long)PERIODO_LEITURA_MS) proxLeitura = agora;
  if ((long)(agora - proxJanela)  > (long)PERIODO_JANELA_MS)  proxJanela  = agora + PERIODO_LEITURA_MS;

  atualizarBuzzer(millis());
  atualizarLedErro(millis());
}
