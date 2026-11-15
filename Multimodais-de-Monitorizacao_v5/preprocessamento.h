#pragma once
// Pré-processamento das entradas dos modelos, idêntico ao dos notebooks 02 e 03 v2.
#include <math.h>
#include <stdint.h>
#include "parametros_modelos.h"

namespace prep {

// CNN: z = (log-mel - média da linha de base) / desvio, por filtro mel; depois quantização int8 da entrada.
inline int8_t quantizarCNN(float logmel, float media, float invDesvio) {
  float z = (logmel - media) * invDesvio;
  long q = lroundf(z / CNN_ESCALA_ENTRADA) + CNN_ZERO_ENTRADA;
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return (int8_t)q;
}

inline float minmax(float v, const float faixa[2]) {
  float r = (v - faixa[0]) / (faixa[1] - faixa[0]);
  return r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
}

// LSTM: janela de 30 passos [temp, umid, press] -> [temp, umid, press - média da janela] em [0, 1].
inline void atributosLSTM(const float* temp, const float* umid, const float* press, float x[][LSTM_ATRIBUTOS]) {
  float media = 0.0f;
  for (int t = 0; t < LSTM_PASSOS; t++) media += press[t];
  media /= LSTM_PASSOS;
  for (int t = 0; t < LSTM_PASSOS; t++) {
    x[t][0] = minmax(temp[t], FAIXA_TEMP);
    x[t][1] = minmax(umid[t], FAIXA_UMID);
    x[t][2] = minmax(press[t] - media, FAIXA_DPRESS);
  }
}

}  // namespace prep
