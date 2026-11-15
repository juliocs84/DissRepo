#pragma once
// Gerado por 05_exportar_firmware.ipynb — não editar
#include <stdint.h>

// CNN 2D (int8; semente 3): entrada 98 x 40 x 1, saída [normal, anômalo]
const int   CNN_QUADROS = 98, CNN_FILTROS_MEL = 40;
const float CNN_ESCALA_ENTRADA = 0.0796889141f;
const int   CNN_ZERO_ENTRADA   = 23;
const float CNN_ESCALA_SAIDA   = 0.00390625f;
const int   CNN_ZERO_SAIDA     = -128;

// LSTM de passo único (float32; semente 2): índices na ordem do modelo
const int LSTM_PASSOS = 30, LSTM_ATRIBUTOS = 3, LSTM_UNIDADES = 32, LSTM_CLASSES = 4;
const int LSTM_ENT_X = 2, LSTM_ENT_H = 1, LSTM_ENT_C = 0;
const int LSTM_SAI_H = 1, LSTM_SAI_C = 0, LSTM_SAI_P = 2;

// Normalização min-max dos atributos (notebook 02): temperatura, umidade, pressão - média da janela
const float FAIXA_TEMP[2] = {10.0f, 45.0f};
const float FAIXA_UMID[2] = {0.0f, 100.0f};
const float FAIXA_DPRESS[2] = {-5.0f, 5.0f};

// Arena mínima medida no TFLite Micro atual (o firmware reserva uma folga sobre estes valores)
const unsigned int ARENA_MINIMA_CNN = 14897, ARENA_MINIMA_LSTM = 6650;
