#pragma once
// Gerado por 05b_exportar_firmware_v5.ipynb (sessões sintéticas) — não editar
#include <stdint.h>

// CNN 2D int8 ajustada (notebook 11): entrada 98 x 40 x 1, saída = 4 níveis acústicos (0 a 3)
const int   CNN_QUADROS = 98, CNN_FILTROS_MEL = 40, CNN_CLASSES = 4;
const float CNN_ESCALA_ENTRADA = 0.0565933213f;
const int   CNN_ZERO_ENTRADA   = -27;
const float CNN_ESCALA_SAIDA   = 0.00390625f;
const int   CNN_ZERO_SAIDA     = -128;

// LSTM (float32, executado em C): 30 passos de 10 s, 3 atributos, 32 unidades, 4 classes
const int LSTM_PASSOS = 30, LSTM_ATRIBUTOS = 3, LSTM_UNIDADES = 32, LSTM_CLASSES = 4;
const int LSTM_ENT_X = 0, LSTM_ENT_H = 1, LSTM_ENT_C = 2, LSTM_SAI_H = 0, LSTM_SAI_C = 1, LSTM_SAI_P = 2;   // só com USAR_LSTM_C = 0

// Normalização min-max dos atributos (notebook 02): temperatura, umidade, pressão - média da janela
const float FAIXA_TEMP[2] = {10.0f, 45.0f};
const float FAIXA_UMID[2] = {0.0f, 100.0f};
const float FAIXA_DPRESS[2] = {-5.0f, 5.0f};

const unsigned int ARENA_MINIMA_CNN = 14897, ARENA_MINIMA_LSTM = 0;
