/*
  Teste de inferência no Arduino Nano 33 BLE Sense (Tabela 12 e paridade com o Python)

  1. Paridade do espectrograma: áudio de teste (paridade.npz) -> log-mel na placa x log-mel do Python.
  2. CNN: entrada quantizada do Python -> saída int8 na placa x saída do TFLite no Python;
     e a cadeia completa (áudio -> espectrograma -> CNN) na placa.
  3. LSTM: janelas reais do Bristol (valores brutos) -> atributos e probabilidades x Python.
  4. Memória: arena usada por modelo.
  5. Latência: 1000 execuções de cada etapa (média, desvio-padrão e máximo) e da decisão no pior caso
     (último quadro do espectrograma + CNN + LSTM + fusão), que é o caminho real no firmware: os outros
     97 quadros são calculados enquanto o áudio chega.

  Arquivos .h necessários nesta pasta: gerados pelo notebook 05 (modelo_cnn.h, modelo_lstm.h, ops_modelos.h,
  parametros_modelos.h, pesos_lstm.h, tabelas_mel.h, vetores_teste.h) + espectro_mel.h, preprocessamento.h,
  inferencia.h, lstm_c.h.
  Abra o Monitor Serial a 115200 baud. Duração: cerca de 5 minutos.
*/
#define USAR_CMSIS_DSP 0      // 1 = arm_rfft_fast_f32 (requer a biblioteca Arduino_CMSIS-DSP)
#define TFLM_CHIRALE   0      // 1 = Chirale_TensorFlowLite em vez de Harvard_TinyMLx
#define USAR_LSTM_C    1      // 1 = LSTM direto em C (lstm_c.h); 0 = LSTM no TFLite Micro
#define ARENA_CNN_BYTES  (24 * 1024)
#define ARENA_LSTM_BYTES (12 * 1024)
#define LIGAR_CACHE    1      // 1 = liga o cache de instruções do nRF52840 (desligado após o reset)

#include "espectro_mel.h"
#include "preprocessamento.h"
#include "inferencia.h"
#include "vetores_teste.h"

const int N_EXECUCOES = 1000;
static mel::Espectro esp;
static float quadroX[mel::QUADRO];
static float logmel[mel::NMEL];
static float logmelPlaca[mel::NQUADROS * mel::NMEL];

struct Estat {                      // média, desvio e máximo (Welford)
  uint32_t n = 0; double media = 0, m2 = 0, max = 0;
  void add(double v) { n++; double d = v - media; media += d / n; m2 += d * (v - media); if (v > max) max = v; }
  double dp() const { return n > 1 ? sqrt(m2 / (n - 1)) : 0; }
};

// Resultados guardados para o resumo final (repetido a cada 30 s no loop)
double gEmax[2], gEmed[2]; int gDifQ[2], gDifQmax[2];
int8_t gCnnQ[2][2]; float gPCadeia[2];
double gLstmEa[8], gLstmEp[8]; int gLstmArg[8];
bool okEsp = true, okCnn = true, okLstm = true, terminou = false;
Estat eEsp, eQuadro, eCnn, eLstm, eDecisao, eTotal;

void linha() { Serial.println(F("------------------------------------------------------------")); }

uint32_t gUltimoQuadroUs = 0;   // tempo do último quadro da janela (inclui a quantização)

// Espectrograma de uma janela de 1 s; se entrada != nullptr, quantiza direto na entrada da CNN
void espectrograma(const int16_t* audio, const float* media, const float* desvio, int8_t* entrada, float* saidaLM) {
  for (int f = 0; f < mel::NQUADROS; f++) {
    uint32_t tq = micros();
    const int16_t* s = audio + f * mel::SALTO;
    for (int n = 0; n < mel::QUADRO; n++) quadroX[n] = s[n] / 32768.0f;
    esp.quadro(quadroX, logmel);
    for (int m = 0; m < mel::NMEL; m++) {
      if (saidaLM) saidaLM[f * mel::NMEL + m] = logmel[m];
      if (entrada) entrada[f * mel::NMEL + m] = prep::quantizarCNN(logmel[m], media[m], 1.0f / desvio[m]);
    }
    if (f == mel::NQUADROS - 1) gUltimoQuadroUs = micros() - tq;
  }
}

void lstmJanela(int k, float x[][LSTM_ATRIBUTOS]) {
  float T[LSTM_PASSOS], U[LSTM_PASSOS], P[LSTM_PASSOS];
  for (int t = 0; t < LSTM_PASSOS; t++) {
    const float* v = TESTE_LSTM_BRUTO + (k * LSTM_PASSOS + t) * 3;
    T[t] = v[0]; U[t] = v[1]; P[t] = v[2];
  }
  prep::atributosLSTM(T, U, P, x);
}

void setup() {
#if LIGAR_CACHE
  NRF_NVMC->ICACHECNF = NVMC_ICACHECNF_CACHEEN_Msk;   // código lido da Flash passa pelo cache
#endif
  Serial.begin(115200);
  while (!Serial);
  // espera o Monitor Serial: só começa depois de receber Enter (assim nada se perde)
  while (!Serial.available()) {
    Serial.println(F(">> Envie Enter (campo de mensagem do Monitor Serial) para iniciar o teste"));
    delay(2000);
  }
  while (Serial.available()) Serial.read();
  linha();
  Serial.println(F("Teste de inferencia - Nano 33 BLE Sense"));
  Serial.print(F("FFT: ")); Serial.println(esp.backend());
  Serial.print(F("TFLM: ")); Serial.println(TFLM_CHIRALE ? "Chirale_TensorFlowLite" : "Harvard_TinyMLx");
  Serial.print(F("LSTM: ")); Serial.println(USAR_LSTM_C ? "em C (lstm_c.h)" : "TFLite Micro");
  Serial.print(F("Cache de instrucoes: ")); Serial.println((NRF_NVMC->ICACHECNF & NVMC_ICACHECNF_CACHEEN_Msk) ? "ligado" : "desligado");
  if (!esp.iniciar()) { Serial.println(F("ERRO: FFT nao inicializou")); while (true); }
  const char* erro = ia::iniciar();
  if (erro) { Serial.print(F("ERRO: ")); Serial.println(erro); while (true); }

  // ---------- 1. Paridade do espectrograma ----------
  linha(); Serial.println(F("1. Paridade do espectrograma (placa x Python)"));
  for (int j = 0; j < N_TESTE_AUDIO; j++) {
    espectrograma(TESTE_AUDIO + j * mel::JANELA, TESTE_BASE_MEDIA + j * 40, TESTE_BASE_DESVIO + j * 40,
                  ia::entradaCNN(), logmelPlaca);
    double emax = 0, emed = 0; int difQ = 0, difQmax = 0;
    for (int i = 0; i < mel::NQUADROS * mel::NMEL; i++) {
      double e = fabs(logmelPlaca[i] - TESTE_LOGMEL[j * 3920 + i]);
      if (e > emax) emax = e;
      emed += e;
      int d = abs(ia::entradaCNN()[i] - TESTE_CNN_ENTRADA[j * 3920 + i]);
      if (d) difQ++;
      if (d > difQmax) difQmax = d;
    }
    Serial.print(F("  janela ")); Serial.print(j);
    Serial.print(F(": log-mel erro max ")); Serial.print(emax, 7);
    Serial.print(F(" | medio ")); Serial.print(emed / 3920, 8);
    Serial.print(F(" | entrada int8 diferente em ")); Serial.print(difQ);
    Serial.print(F("/3920 (max ")); Serial.print(difQmax); Serial.println(F(")"));
    gEmax[j] = emax; gEmed[j] = emed / 3920; gDifQ[j] = difQ; gDifQmax[j] = difQmax;
    if (emax > 1e-3 || difQmax > 1) okEsp = false;
  }
  Serial.println(okEsp ? F("  -> PARIDADE OK") : F("  -> ATENCAO: diferenca acima do esperado"));

  // ---------- 2. CNN ----------
  linha(); Serial.println(F("2. CNN int8 (placa x TFLite no Python)"));
  for (int j = 0; j < N_TESTE_AUDIO; j++) {
    memcpy(ia::entradaCNN(), TESTE_CNN_ENTRADA + j * 3920, 3920);
    float p; int8_t q[2];
    ia::executarCNN(p, q);
    int d = max(abs(q[0] - TESTE_CNN_SAIDA[j * 2]), abs(q[1] - TESTE_CNN_SAIDA[j * 2 + 1]));
    Serial.print(F("  janela ")); Serial.print(j);
    Serial.print(F(": saida placa [")); Serial.print(q[0]); Serial.print(','); Serial.print(q[1]);
    Serial.print(F("] | Python [")); Serial.print(TESTE_CNN_SAIDA[j * 2]); Serial.print(',');
    Serial.print(TESTE_CNN_SAIDA[j * 2 + 1]); Serial.print(F("] | p_anomalia ")); Serial.println(p, 4);
    if (d > 1) okCnn = false;
    gCnnQ[j][0] = q[0]; gCnnQ[j][1] = q[1];
    // cadeia completa: áudio -> espectrograma da placa -> CNN
    espectrograma(TESTE_AUDIO + j * mel::JANELA, TESTE_BASE_MEDIA + j * 40, TESTE_BASE_DESVIO + j * 40,
                  ia::entradaCNN(), nullptr);
    ia::executarCNN(p, q);
    Serial.print(F("           cadeia completa na placa: p_anomalia ")); Serial.println(p, 4);
    gPCadeia[j] = p;
  }
  Serial.println(okCnn ? F("  -> CNN OK") : F("  -> ATENCAO: saida diferente do Python"));

  // ---------- 3. LSTM ----------
  linha(); Serial.println(USAR_LSTM_C ? F("3. LSTM float32 em C (placa x Python)") : F("3. LSTM float32 no TFLite Micro (placa x Python)"));
  float x[LSTM_PASSOS][LSTM_ATRIBUTOS], prob[LSTM_CLASSES];
  for (int k = 0; k < N_TESTE_LSTM; k++) {
    lstmJanela(k, x);
    double ea = 0, ep = 0;
    for (int i = 0; i < LSTM_PASSOS * LSTM_ATRIBUTOS; i++) ea = fmax(ea, fabs((&x[0][0])[i] - TESTE_LSTM_ATRIB[k * 90 + i]));
    ia::executarLSTM(x, prob);
    int arg = 0;
    for (int c = 0; c < LSTM_CLASSES; c++) {
      ep = fmax(ep, fabs(prob[c] - TESTE_LSTM_PROB[k * 4 + c]));
      if (prob[c] > prob[arg]) arg = c;
    }
    Serial.print(F("  janela ")); Serial.print(k); Serial.print(F(" (classe real ")); Serial.print(TESTE_LSTM_CLASSE[k]);
    Serial.print(F(", predita ")); Serial.print(arg); Serial.print(F("): atributos erro max ")); Serial.print(ea, 7);
    Serial.print(F(" | probabilidades erro max ")); Serial.println(ep, 7);
    if (ea > 1e-4 || ep > 1e-4) okLstm = false;
    if (k < 8) { gLstmEa[k] = ea; gLstmEp[k] = ep; gLstmArg[k] = arg; }
  }
  Serial.println(okLstm ? F("  -> LSTM OK") : F("  -> ATENCAO: diferenca acima do esperado"));

  // ---------- 4. Memória ----------
  linha(); Serial.println(F("4. Memoria"));
  Serial.print(F("  Arena CNN usada:  ")); Serial.print(ia::arenaUsadaCNN() / 1024.0, 1);
  Serial.print(F(" KB (reservada ")); Serial.print(ARENA_CNN_BYTES / 1024); Serial.println(F(" KB)"));
  Serial.print(F("  Arena LSTM usada: ")); Serial.print(ia::arenaUsadaLSTM() / 1024.0, 1);
  Serial.print(F(" KB (reservada ")); Serial.print(ARENA_LSTM_BYTES / 1024); Serial.println(F(" KB)"));
  Serial.print(F("  Arena total usada: ")); Serial.print((ia::arenaUsadaCNN() + ia::arenaUsadaLSTM()) / 1024.0, 1);
  Serial.println(F(" KB"));

  // ---------- 5. Latência ----------
  linha(); Serial.print(F("5. Latencia (")); Serial.print(N_EXECUCOES); Serial.println(F(" execucoes; aguarde ~5 min)"));
  float p;
  for (int i = 0; i < N_EXECUCOES; i++) {
    int j = i % N_TESTE_AUDIO, k = i % N_TESTE_LSTM;
    uint32_t t0 = micros();
    espectrograma(TESTE_AUDIO + j * mel::JANELA, TESTE_BASE_MEDIA + j * 40, TESTE_BASE_DESVIO + j * 40,
                  ia::entradaCNN(), nullptr);
    uint32_t t1 = micros();
    ia::executarCNN(p);
    uint32_t t2 = micros();
    lstmJanela(k, x);                                // pré-processamento + 30 passos + fusão por regra
    ia::executarLSTM(x, prob);
    int nTerm = 0;
    for (int c = 1; c < LSTM_CLASSES; c++) if (prob[c] > prob[nTerm]) nTerm = c;
    volatile int classe = (nTerm == 3 || (nTerm == 2 && p >= 0.5f)) ? 3 : max(nTerm, p >= 0.5f ? 1 : 0);
    (void)classe;
    uint32_t t3 = micros();
    eEsp.add((t1 - t0) / 1000.0); eQuadro.add(gUltimoQuadroUs / 1000.0); eCnn.add((t2 - t1) / 1000.0);
    eLstm.add((t3 - t2) / 1000.0); eTotal.add((t3 - t0) / 1000.0);
    eDecisao.add((gUltimoQuadroUs + (t3 - t1)) / 1000.0);
    if ((i + 1) % 100 == 0) { Serial.print(F("  ")); Serial.print(i + 1); Serial.println(F(" execucoes")); }
  }
  terminou = true;
  imprimirResumo();
}

void imprimirResumo() {
  linha(); Serial.println(F("RESUMO COMPLETO"));
  Serial.print(F("FFT: ")); Serial.print(esp.backend());
  Serial.print(F(" | TFLM: ")); Serial.print(TFLM_CHIRALE ? "Chirale" : "Harvard_TinyMLx");
  Serial.print(F(" | cache: ")); Serial.println((NRF_NVMC->ICACHECNF & NVMC_ICACHECNF_CACHEEN_Msk) ? "ligado" : "desligado");
  Serial.println(F("Validacao (placa x Python):"));
  for (int j = 0; j < N_TESTE_AUDIO; j++) {
    Serial.print(F("  espectrograma janela ")); Serial.print(j); Serial.print(F(": erro max ")); Serial.print(gEmax[j], 7);
    Serial.print(F(", medio ")); Serial.print(gEmed[j], 8); Serial.print(F(" | int8 diferentes ")); Serial.print(gDifQ[j]);
    Serial.print(F("/3920 (max ")); Serial.print(gDifQmax[j]); Serial.println(F(")"));
  }
  for (int j = 0; j < N_TESTE_AUDIO; j++) {
    Serial.print(F("  CNN janela ")); Serial.print(j); Serial.print(F(": placa [")); Serial.print(gCnnQ[j][0]); Serial.print(',');
    Serial.print(gCnnQ[j][1]); Serial.print(F("] Python [")); Serial.print(TESTE_CNN_SAIDA[j * 2]); Serial.print(',');
    Serial.print(TESTE_CNN_SAIDA[j * 2 + 1]); Serial.print(F("] | cadeia completa p_anomalia ")); Serial.println(gPCadeia[j], 4);
  }
  for (int k = 0; k < N_TESTE_LSTM && k < 8; k++) {
    Serial.print(F("  LSTM janela ")); Serial.print(k); Serial.print(F(" (real ")); Serial.print(TESTE_LSTM_CLASSE[k]);
    Serial.print(F(", predita ")); Serial.print(gLstmArg[k]); Serial.print(F("): atributos erro ")); Serial.print(gLstmEa[k], 7);
    Serial.print(F(" | prob. erro ")); Serial.println(gLstmEp[k], 7);
  }
  Serial.print(F("  -> espectrograma ")); Serial.print(okEsp ? F("OK") : F("ATENCAO"));
  Serial.print(F(" | CNN ")); Serial.print(okCnn ? F("OK") : F("ATENCAO"));
  Serial.print(F(" | LSTM ")); Serial.println(okLstm ? F("OK") : F("ATENCAO"));
  Serial.print(F("Arena usada: CNN ")); Serial.print(ia::arenaUsadaCNN() / 1024.0, 1);
  Serial.print(F(" KB | LSTM ")); Serial.print(ia::arenaUsadaLSTM() / 1024.0, 1);
  Serial.print(F(" KB | total ")); Serial.print((ia::arenaUsadaCNN() + ia::arenaUsadaLSTM()) / 1024.0, 1); Serial.println(F(" KB"));

  const char* nomes[] = {"Espectrograma mel, 1 quadro (10 ms)", "Espectrograma mel, janela de 1 s (98 quadros)",
                         "Inferencia CNN 2D", "Inferencia LSTM + fusao",
                         "Decisao, pior caso (ultimo quadro + CNN + LSTM + fusao)",
                         "Janela inteira sequencial (98 quadros + CNN + LSTM)"};
  Estat* e[] = {&eQuadro, &eEsp, &eCnn, &eLstm, &eDecisao, &eTotal};
  Serial.print(F("Latencia (")); Serial.print(eTotal.n); Serial.println(F(" execucoes; media +- desvio; maximo) em ms:"));
  for (int i = 0; i < 6; i++) {
    Serial.print(F("  ")); Serial.print(nomes[i]); Serial.print(F(": "));
    Serial.print(e[i]->media, 2); Serial.print(F(" +- ")); Serial.print(e[i]->dp(), 2);
    Serial.print(F("; max ")); Serial.println(e[i]->max, 2);
  }
  Serial.print(F("  Decisao (pior caso) < 200 ms (RNF2): "));
  Serial.println(eDecisao.max < 200 ? F("ATENDE") : F("NAO ATENDE"));
  Serial.print(F("  Ocupacao do processador em regime (98 quadros + CNN por segundo + LSTM a cada 10 s): "));
  Serial.print((eEsp.media + eCnn.media + eLstm.media / 10.0) / 10.0, 1); Serial.println(F(" %"));
  Serial.println(F("Flash e RAM totais: ver a mensagem de compilacao do firmware principal."));
  linha();
}

void loop() {                                       // repete o resumo para poder copiar a qualquer momento
  static unsigned long ultimo = 0;
  if (terminou && millis() - ultimo > 30000) { ultimo = millis(); imprimirResumo(); }
}
