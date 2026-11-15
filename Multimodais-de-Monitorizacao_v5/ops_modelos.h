#pragma once
// Gerado por 05_exportar_firmware.ipynb — não editar
#include <stdint.h>

// Operações usadas pela CNN e pelo LSTM: ADD, CONV_2D, FULLY_CONNECTED, LOGISTIC, MAX_POOL_2D, MEAN, MUL, RESHAPE, SOFTMAX, STRIDED_SLICE, TANH
#define N_OPS_MODELOS 11

template <typename R>
bool registrarOps(R& r) {
  return
    r.AddAdd() == kTfLiteOk &&
    r.AddConv2D() == kTfLiteOk &&
    r.AddFullyConnected() == kTfLiteOk &&
    r.AddLogistic() == kTfLiteOk &&
    r.AddMaxPool2D() == kTfLiteOk &&
    r.AddMean() == kTfLiteOk &&
    r.AddMul() == kTfLiteOk &&
    r.AddReshape() == kTfLiteOk &&
    r.AddSoftmax() == kTfLiteOk &&
    r.AddStridedSlice() == kTfLiteOk &&
    r.AddTanh() == kTfLiteOk;
}
