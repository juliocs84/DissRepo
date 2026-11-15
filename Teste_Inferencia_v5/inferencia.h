#pragma once
/*
  Execução dos modelos com o TensorFlow Lite Micro.
    - CNN 2D int8 (espectrograma 98 x 40 -> probabilidades dos CNN_CLASSES níveis acústicos; v5: 4 níveis);
    - LSTM em float32, 30 passos por janela (estado h, c zerado no início). Com USAR_LSTM_C = 1 (padrão),
      o LSTM roda direto em C (lstm_c.h), sem interpretador; com 0, roda o modelo de passo único no TFLite Micro.

  Biblioteca: Harvard_TinyMLx (padrão, Tabela 9). Para usar a Chirale_TensorFlowLite (TFLM recente),
  defina TFLM_CHIRALE como 1 ANTES de incluir este arquivo. Instale apenas uma das duas.
  Tamanho das arenas: ARENA_CNN_BYTES e ARENA_LSTM_BYTES (defina antes de incluir para ajustar).
*/
#include <Arduino.h>
#include <string.h>

#ifndef TFLM_CHIRALE
#define TFLM_CHIRALE 0
#endif
#if TFLM_CHIRALE
#include <Chirale_TensorFlowLite.h>
#else
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/version.h"
#endif
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "parametros_modelos.h"
#include "ops_modelos.h"
#include "modelo_cnn.h"
#ifndef USAR_LSTM_C
#define USAR_LSTM_C 1
#endif
#if USAR_LSTM_C
#include "lstm_c.h"
#else
#include "modelo_lstm.h"
#endif

#ifndef ARENA_CNN_BYTES
#define ARENA_CNN_BYTES (24 * 1024)
#endif
#ifndef ARENA_LSTM_BYTES
#define ARENA_LSTM_BYTES (12 * 1024)
#endif

namespace ia {

alignas(16) static uint8_t arenaCNN[ARENA_CNN_BYTES];
#if !USAR_LSTM_C
alignas(16) static uint8_t arenaLSTM[ARENA_LSTM_BYTES];
#endif

static tflite::MicroInterpreter* cnn = nullptr;
static tflite::MicroInterpreter* lstm = nullptr;
static TfLiteTensor* cnnEntrada = nullptr;
static TfLiteTensor* cnnSaida = nullptr;
static TfLiteTensor* lstmEnt[3] = {nullptr, nullptr, nullptr};
static TfLiteTensor* lstmSai[3] = {nullptr, nullptr, nullptr};

// Inicializa os dois interpretadores. Retorna nullptr se tudo certo; senão, a mensagem de erro.
inline const char* iniciar() {
  static tflite::MicroMutableOpResolver<N_OPS_MODELOS> resolver;
  if (!registrarOps(resolver)) return "falha ao registrar as operacoes";

  const tflite::Model* mc = tflite::GetModel(g_modelo_cnn);
  if (mc->version() != TFLITE_SCHEMA_VERSION) return "versao do esquema TFLite incompativel";
#if TFLM_CHIRALE
  static tflite::MicroInterpreter interpCNN(mc, resolver, arenaCNN, ARENA_CNN_BYTES);
#else
  static tflite::MicroErrorReporter relator;
  static tflite::MicroInterpreter interpCNN(mc, resolver, arenaCNN, ARENA_CNN_BYTES, &relator);
#endif
  cnn = &interpCNN;
  if (cnn->AllocateTensors() != kTfLiteOk) return "CNN: arena insuficiente (aumente ARENA_CNN_BYTES)";
  cnnEntrada = cnn->input(0);
  cnnSaida = cnn->output(0);
  if (cnnEntrada->type != kTfLiteInt8 || cnnEntrada->bytes != (size_t)(CNN_QUADROS * CNN_FILTROS_MEL))
    return "CNN: entrada diferente de 98 x 40 int8";
  if (cnnSaida->type != kTfLiteInt8 || cnnSaida->bytes != (size_t)CNN_CLASSES)
    return "CNN: numero de saidas diferente de CNN_CLASSES (gere os .h com o notebook 05b)";

#if !USAR_LSTM_C
  const tflite::Model* ml = tflite::GetModel(g_modelo_lstm);
  if (ml->version() != TFLITE_SCHEMA_VERSION) return "versao do esquema TFLite incompativel";
#if TFLM_CHIRALE
  static tflite::MicroInterpreter interpLSTM(ml, resolver, arenaLSTM, ARENA_LSTM_BYTES);
#else
  static tflite::MicroInterpreter interpLSTM(ml, resolver, arenaLSTM, ARENA_LSTM_BYTES, &relator);
#endif
  lstm = &interpLSTM;
  if (lstm->AllocateTensors() != kTfLiteOk) return "LSTM: arena insuficiente (aumente ARENA_LSTM_BYTES)";
  for (int i = 0; i < 3; i++) { lstmEnt[i] = lstm->input(i); lstmSai[i] = lstm->output(i); }
  if (lstmEnt[LSTM_ENT_X]->bytes != LSTM_ATRIBUTOS * sizeof(float) ||
      lstmEnt[LSTM_ENT_H]->bytes != LSTM_UNIDADES * sizeof(float) ||
      lstmSai[LSTM_SAI_P]->bytes != LSTM_CLASSES * sizeof(float))
    return "LSTM: entradas/saidas fora do esperado";
#endif
  return nullptr;
}

inline size_t arenaUsadaCNN()  { return cnn ? cnn->arena_used_bytes() : 0; }
inline size_t arenaUsadaLSTM() { return lstm ? lstm->arena_used_bytes() : 0; }

// Buffer da entrada da CNN (98 x 40 int8): o firmware escreve aqui, quadro a quadro.
inline int8_t* entradaCNN() { return cnnEntrada->data.int8; }

// Executa a CNN sobre a entrada já preenchida. prob: probabilidade de cada nível acústico (0 a CNN_CLASSES-1).
inline bool executarCNN(float prob[CNN_CLASSES], int8_t saidaQ[CNN_CLASSES] = nullptr) {
  if (cnn->Invoke() != kTfLiteOk) return false;
  const int8_t* s = cnnSaida->data.int8;
  for (int c = 0; c < CNN_CLASSES; c++) {
    if (saidaQ) saidaQ[c] = s[c];
    prob[c] = (s[c] - CNN_ZERO_SAIDA) * CNN_ESCALA_SAIDA;
  }
  return true;
}

// Executa o LSTM de passo único 30 vezes. x: atributos normalizados. prob: 4 probabilidades.
inline bool executarLSTM(const float x[][LSTM_ATRIBUTOS], float prob[LSTM_CLASSES]) {
#if USAR_LSTM_C
  lstmc::executar(x, prob);
  return true;
#else
  float h[LSTM_UNIDADES], c[LSTM_UNIDADES];
  memset(h, 0, sizeof(h));
  memset(c, 0, sizeof(c));
  for (int t = 0; t < LSTM_PASSOS; t++) {
    memcpy(lstmEnt[LSTM_ENT_X]->data.f, x[t], sizeof(float) * LSTM_ATRIBUTOS);
    memcpy(lstmEnt[LSTM_ENT_H]->data.f, h, sizeof(h));
    memcpy(lstmEnt[LSTM_ENT_C]->data.f, c, sizeof(c));
    if (lstm->Invoke() != kTfLiteOk) return false;
    // copia as saídas antes de escrever as entradas: o planejador de memória pode sobrepô-las
    memcpy(h, lstmSai[LSTM_SAI_H]->data.f, sizeof(h));
    memcpy(c, lstmSai[LSTM_SAI_C]->data.f, sizeof(c));
  }
  memcpy(prob, lstmSai[LSTM_SAI_P]->data.f, sizeof(float) * LSTM_CLASSES);
  return true;
#endif
}

}  // namespace ia
