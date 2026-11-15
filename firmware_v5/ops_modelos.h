#pragma once
// Gerado por 05b_exportar_firmware_v5.ipynb (sessões sintéticas) — não editar
#include <stdint.h>

// Operações usadas pela CNN: CONV_2D, FULLY_CONNECTED, MAX_POOL_2D, MEAN, SOFTMAX (o LSTM roda em C, sem interpretador)
#define N_OPS_MODELOS 5

template <typename R>
bool registrarOps(R& r) {
  return
    r.AddConv2D() == kTfLiteOk &&
    r.AddFullyConnected() == kTfLiteOk &&
    r.AddMaxPool2D() == kTfLiteOk &&
    r.AddMean() == kTfLiteOk &&
    r.AddSoftmax() == kTfLiteOk;
}
