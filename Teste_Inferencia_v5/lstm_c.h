#pragma once
/*
  LSTM ambiental executado diretamente em C, sem o interpretador do TensorFlow Lite Micro.
  Mesmos pesos e mesma conta do modelo treinado (notebook 04), exportados pelo notebook 05 em pesos_lstm.h:
    z = x Wk + h Wr + b;  i, f, o = sigmoide;  g = tanh;  c' = f c + i g;  h' = o tanh(c')
    saída = softmax( relu(h' Wd + bd) Wo + bo ), após os 30 passos (estado zerado no início da janela).
  Motivo: no interpretador, cada passo levava ~4 ms, quase todo de sobrecarga das 21 operações do grafo.
*/
#include <math.h>
#include <string.h>
#include "parametros_modelos.h"
#include "pesos_lstm.h"

#pragma GCC push_options
#pragma GCC optimize("O2")

namespace lstmc {

const int U = LSTM_UNIDADES, G = 4 * LSTM_UNIDADES, D = 16;

inline float sigmoide(float v) { return 1.0f / (1.0f + expf(-v)); }

// x: janela de 30 passos com os atributos normalizados; prob: 4 probabilidades (Normal, Atenção, Crítico, Falha)
inline void executar(const float x[][LSTM_ATRIBUTOS], float prob[LSTM_CLASSES]) {
  float h[U], c[U], z[G];
  memset(h, 0, sizeof(h));
  memset(c, 0, sizeof(c));
  for (int t = 0; t < LSTM_PASSOS; t++) {
    for (int j = 0; j < G; j++) z[j] = LSTM_B[j];
    for (int a = 0; a < LSTM_ATRIBUTOS; a++) {
      const float xa = x[t][a];
      const float* w = LSTM_WK + a * G;
      for (int j = 0; j < G; j++) z[j] += xa * w[j];
    }
    for (int k = 0; k < U; k++) {
      const float hk = h[k];
      const float* w = LSTM_WR + k * G;
      for (int j = 0; j < G; j++) z[j] += hk * w[j];
    }
    for (int k = 0; k < U; k++) {
      float i = sigmoide(z[k]), f = sigmoide(z[U + k]), g = tanhf(z[2 * U + k]), o = sigmoide(z[3 * U + k]);
      c[k] = f * c[k] + i * g;
      h[k] = o * tanhf(c[k]);
    }
  }
  float d[D], s[LSTM_CLASSES];
  for (int j = 0; j < D; j++) {
    float v = LSTM_BD[j];
    for (int k = 0; k < U; k++) v += h[k] * LSTM_WD[k * D + j];
    d[j] = v > 0 ? v : 0;
  }
  float m = -1e30f;
  for (int j = 0; j < LSTM_CLASSES; j++) {
    float v = LSTM_BO[j];
    for (int k = 0; k < D; k++) v += d[k] * LSTM_WO[k * LSTM_CLASSES + j];
    s[j] = v;
    if (v > m) m = v;
  }
  float soma = 0;
  for (int j = 0; j < LSTM_CLASSES; j++) { s[j] = expf(s[j] - m); soma += s[j]; }
  for (int j = 0; j < LSTM_CLASSES; j++) prob[j] = s[j] / soma;
}

}  // namespace lstmc

#pragma GCC pop_options
