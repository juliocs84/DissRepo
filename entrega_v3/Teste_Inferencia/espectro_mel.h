#pragma once
/*
  Espectrograma log-mel no firmware, idêntico ao do notebook 01 (core_mimii.log_mel):
    quadro de 400 amostras (25 ms) x janela de Hann periódica -> FFT real de 512 pontos (zero-padding)
    -> potência |X|^2 nos 257 bins -> 40 filtros mel HTK (20-8000 Hz) -> ln(energia + 1e-6).
  Uma janela de 1 s (16 000 amostras) gera 98 quadros com salto de 160 amostras (10 ms).

  FFT: por padrão, implementação própria (FFT complexa radix-2 de 256 pontos + pós-processamento
  para sinal real), com tabelas em memória Flash (tabelas_mel.h). Com USAR_CMSIS_DSP = 1 usa
  arm_rfft_fast_f32 da biblioteca Arduino_CMSIS-DSP. O sketch Teste_Inferencia mede a paridade
  das duas opções com o Python.
*/
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "tabelas_mel.h"

#ifndef USAR_CMSIS_DSP
#define USAR_CMSIS_DSP 0
#endif
#if USAR_CMSIS_DSP
#include <arm_math.h>
#endif

// O núcleo Arduino compila com -Os (tamanho); o laço da FFT ganha com otimização de velocidade.
#pragma GCC push_options
#pragma GCC optimize("O2")

namespace mel {

const int FS = 16000, QUADRO = 400, SALTO = 160, NFFT = 512, NBINS = NFFT / 2 + 1;
const int NMEL = 40, NQUADROS = 98, JANELA = 16000;
const float EPS = 1e-6f;

class Espectro {
 public:
  bool iniciar() {
#if USAR_CMSIS_DSP
    return arm_rfft_fast_init_f32(&inst_, NFFT) == ARM_MATH_SUCCESS;
#else
    return true;
#endif
  }

  const char* backend() const { return USAR_CMSIS_DSP ? "CMSIS-DSP (arm_rfft_fast_f32)" : "FFT propria (radix-2)"; }

  // x: 400 amostras em [-1, 1] (int16 / 32768), sem janela. Saída: 40 log-energias mel.
  void quadro(const float* x, float* logmel) {
    for (int n = 0; n < QUADRO; n++) a_[n] = x[n] * HANN[n];
    finalizar(logmel);
  }

  // Mesmo cálculo, lendo 400 amostras int16 de um buffer circular (máscara = tamanho - 1) a partir de ini.
  // s * HANN / 32768 é bit a bit igual a (s / 32768) * HANN, porque dividir por 2^15 é exato.
  void quadroAnel(const int16_t* anel, uint32_t mascara, uint32_t ini, float* logmel) {
    for (int n = 0; n < QUADRO; n++) a_[n] = ((float)anel[(ini + n) & mascara] * HANN[n]) * (1.0f / 32768.0f);
    finalizar(logmel);
  }

 private:
  void finalizar(float* logmel) {
    for (int n = QUADRO; n < NFFT; n++) a_[n] = 0.0f;
    potencia(pot_);
    for (int m = 0; m < NMEL; m++) {
      const float* w = MEL_PESOS + MEL_DESLOC[m];
      const float* p = pot_ + MEL_INICIO[m];
      float s = 0.0f;
      for (int j = 0; j < MEL_TAM[m]; j++) s += p[j] * w[j];
      logmel[m] = logf(s + EPS);
    }
  }

  float a_[NFFT];       // entrada da FFT (destruída pela transformada)
  float b_[NFFT];       // saída intermediária
  float pot_[NBINS];    // |X[k]|^2, k = 0..256

#if USAR_CMSIS_DSP
  arm_rfft_fast_instance_f32 inst_;
  void potencia(float* P) {
    arm_rfft_fast_f32(&inst_, a_, b_, 0);       // saída: [X0.re, X256.re, X1.re, X1.im, ...]
    P[0] = b_[0] * b_[0];
    P[NFFT / 2] = b_[1] * b_[1];
    for (int k = 1; k < NFFT / 2; k++) P[k] = b_[2 * k] * b_[2 * k] + b_[2 * k + 1] * b_[2 * k + 1];
  }
#else
  // z[n] = x[2n] + j x[2n+1] (256 pontos complexos); Z = FFT256(z); depois separa as partes par e ímpar.
  void potencia(float* P) {
    const int M = NFFT / 2;                     // 256
    float* re = b_;                             // b_[0..255]   parte real
    float* im = b_ + M;                         // b_[256..511] parte imaginária
    for (int n = 0; n < M; n++) {               // reordenação por inversão de bits
      int r = FFT_REV[n];
      re[r] = a_[2 * n];
      im[r] = a_[2 * n + 1];
    }
    for (int tam = 2; tam <= M; tam <<= 1) {    // estágios radix-2 (decimação no tempo)
      int meio = tam >> 1, passo = M / tam;
      for (int i = 0; i < M; i += tam) {
        for (int k = 0; k < meio; k++) {
          float wr = FFT_COS[k * passo], wi = FFT_SIN[k * passo];   // e^{-j 2 pi k / tam}
          int p = i + k, q = p + meio;
          float tr = wr * re[q] - wi * im[q];
          float ti = wr * im[q] + wi * re[q];
          re[q] = re[p] - tr; im[q] = im[p] - ti;
          re[p] += tr;        im[p] += ti;
        }
      }
    }
    // X[k] = E[k] + W^k O[k], com E = (Z[k] + conj Z[M-k]) / 2 e O = -j (Z[k] - conj Z[M-k]) / 2
    P[0] = (re[0] + im[0]) * (re[0] + im[0]);
    P[M] = (re[0] - im[0]) * (re[0] - im[0]);
    for (int k = 1; k < M; k++) {
      float ar = re[k], ai = im[k], br = re[M - k], bi = -im[M - k];
      float er = 0.5f * (ar + br), ei = 0.5f * (ai + bi);
      float or_ = 0.5f * (ai - bi), oi = -0.5f * (ar - br);
      float wr = RFFT_COS[k], wi = RFFT_SIN[k];                     // e^{-j 2 pi k / 512}
      float xr = er + wr * or_ - wi * oi;
      float xi = ei + wr * oi + wi * or_;
      P[k] = xr * xr + xi * xi;
    }
  }
#endif
};

}  // namespace mel

#pragma GCC pop_options
