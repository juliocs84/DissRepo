#pragma once
// Gerado por 05b_exportar_firmware_v5.ipynb (sessões sintéticas) — não editar
#include <stdint.h>

// Camada de fusão treinada (notebook 11): z = x W + b; classe = argmax(softmax(z)).
// x = [p_LSTM(Normal, Atenção, Crítico, Falha), média de 10 janelas de p_CNN(nível 0, 1, 2, 3)]; W[i][j] = FUSAO_W[i * 4 + j].
const int FUSAO_ENTRADAS = 8, FUSAO_CLASSES = 4;
const float FUSAO_W[32] = {
  2.791013f, -0.883359909f, -0.772376418f, -8.97092724f,
  -4.54119635f, 3.729182f, 1.15861118f, -14.6312532f,
  -34.2866554f, -6.89055014f, 2.67231083f, 5.17172813f,
  -24.6961479f, -38.3085709f, -5.2055645f, 11.1098442f,
  6.36405039f, 0.904263616f, -2.85142136f, -18.284565f,
  -2.04724884f, 5.03360891f, -1.28025508f, -2.69391608f,
  -10.8326569f, -7.16079712f, 4.54042244f, 6.01195288f,
  -32.0413857f, -19.0518475f, -9.5457716f, 20.0440617f
};
const float FUSAO_B[4] = {
  0.0691873431f, 0.0443073995f, -0.31601873f, 0.750728428f
};
