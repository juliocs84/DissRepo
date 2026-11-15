# Detecção Embarcada de Anomalias em Data Centers por Fusão de Dados Ambientais e Acústicos

**Desenvolvimento e Avaliação de um Artefato TinyML**

Repositório de replicação da dissertação de mestrado de **Júlio César da Silva**, Master of Science in Artificial Intelligence, AGTU University (Florida, EUA), sob orientação do **Prof. Dr. William Aparecido Celestino Lopes** (2026).

O artefato classifica o estado de um ambiente de data center em quatro classes (**Normal, Atenção, Crítico e Falha**) diretamente em um microcontrolador ARM Cortex-M4 (Arduino Nano 33 BLE Sense). Ele combina, por fusão tardia, um ramo **LSTM** para as séries de temperatura, umidade e pressão (BME280) e um ramo **CNN 2D** para espectrogramas log-mel do áudio do microfone embarcado. Uma camada orientada a MLOps faz telemetria, dashboard e alertas por e-mail e Telegram.

> **Natureza dos dados.** A avaliação quantitativa da classificação usa **sessões sintéticas** (SIM001 a SIM006), compostas a partir dos datasets públicos MIMII e Bristol segundo o protocolo da seção 3.3.1 da dissertação. Elas **não são medições feitas no nó nem em um data center em operação**. As medições do protótipo físico verificam a execução embarcada, a latência, a memória, a sinalização e a comunicação.

---

## Principais resultados

| Indicador | Valor | Origem |
|---|---|---|
| F1-score macro, fusão multimodal | 0,921 | Sessão sintética de teste SIM006 (Python) |
| F1-score macro, melhor unimodal (CNN) | 0,673 | Idem |
| Diferença (IC 95%, bootstrap pareado) | 0,248 (0,131 a 0,457) | Idem, 2.000 reamostragens de 25 grupos |
| Acurácia / recall da classe Falha | 93,1% / 0,964 | Idem |
| Decisão no pior caso | 191,9 ± 0,1 ms (máx. 192,2 ms) | Placa, bancada, 1.000 execuções |
| Decisão em operação | média de 185,8 ms; máx. 199,6 ms; 5.932 ciclos | Placa, firmware v5 |
| Flash / RAM estática / arena | 239,1 KB / 77,9 KB / 13,4 KB | Relatório de compilação e `arena_used_bytes()` |
| Paridade placa × Python | diferenças ≈ 10⁻⁵; classes idênticas | `Teste_Inferencia_v5` |

Os resultados de classificação são compatíveis com as hipóteses **nas condições examinadas**. A superioridade da fusão (H1) tem caráter exploratório, porque uma configuração inicial foi avaliada na SIM006 antes da busca de hiperparâmetros (seção 9.4). A confirmação independente exige novas sessões de teste e aquisição multimodal contínua em operação real.

---

## Estrutura do repositório

```
firmware/
  Coleta_InSitu/                     firmware de coleta (áudio 16 kHz + BME280 no microSD)
  Multimodais-de-Monitorizacao_v5/   firmware final (CNN int8 de 4 níveis, LSTM float32 em C, fusão densa)
  Teste_Inferencia_v5/               paridade com o Python, arena e latência (Apêndice D)
  secrets.h.example                  modelo de credenciais (o secrets.h real NÃO é versionado)
notebooks/                           preparação, treino, geração de sessões, avaliação e manifesto (ordem abaixo)
modelos/
  cnn_v3/                            CNN pré-treinada (binária) e comparação float32 × int8
  sim/                               modelos finais v2 (CNN int8, LSTM, fusao.json) e busca_hiperparametros.csv
resultados_sim/                      predições, métricas, IC por bootstrap e hipóteses (notebook 12)
manifesto/                           hashes SHA-256, eventos da SIM006, reprodução da Tabela 13
  origem/                            origem de cada clipe MIMII e trecho Bristol por sessão e partição
mlops/
  receptor/                          serviço receptor da telemetria (Python) e simular_alertas.py
  dashboard/                         dashboard Streamlit
  n8n/workflow_alertas_tinyml.json   fluxo de alertas (as credenciais ficam só no n8n)
hardware/                            esquemático (Figura 8) e mapeamento de pinos (Tabela 6)
questionario/                        instrumento de validação na versão aplicada (sem respostas)
```

> Ajuste esta árvore aos nomes reais das pastas no momento da publicação.

---

## Requisitos

**Python (treinamento e avaliação)**
- Python 3.13, TensorFlow 2.21.0, Keras 3.13.2, NumPy, pandas, scikit-learn, soundfile, matplotlib
- Google Colab (recomendado) com cerca de 15 GB livres para o MIMII e as sessões

**Firmware**
- Arduino IDE 2.3.8, pacote *Arduino Mbed OS Nano Boards* 4.6.0 (inclui a biblioteca PDM)
- Bibliotecas e versões listadas no Apêndice B.5 da dissertação

**Hardware**
- Arduino Nano 33 BLE Sense (nRF52840, Cortex-M4, 1 MB de flash, 256 KB de RAM)
- BME280, módulo ESP8266 (firmware AT 1.3.0), microSD, display, LEDs e buzzer. Componentes e ligações: Tabela 5, Figura 8 e Tabela 6

---

## Dados de origem (não incluídos)

| Dataset | Uso | Obtenção e licença |
|---|---|---|
| **MIMII Dataset** (Purohit et al., 2019), ventiladores, 6 dB | Pré-treinamento da CNN e composição do áudio das sessões | Zenodo, registro 3384388, CC BY-SA 4.0 |
| **Multi-sensor, Multi-device Smart Building Indoor Environmental Dataset** (Erol et al., 2023), Universidade de Bristol | Pré-treinamento do LSTM e base ambiental das sessões | Data in Brief / repositório da Universidade de Bristol, Non-Commercial Government Licence v2 |

Os arquivos de áudio das sessões sintéticas (cerca de 6,5 GB) não são versionados. Eles podem ser regenerados pelo notebook 07b com a semente registrada.

---

## Ordem de execução

Defina a pasta base no Google Drive (`dissertacao_tinyml/`) e coloque nela os datasets de origem.

| Etapa | Notebook | Saída principal |
|---|---|---|
| 1 | `01_*` a `04_*`: preparação do MIMII (partição por gravação) e do Bristol (blocos cronológicos 70/15/15), pré-treinamento e quantização | `cnn_v3/`, `lstm/` |
| 2 | `05_exportar_firmware`, `05b_exportar_firmware_v5` | arquivos `.h` do firmware |
| 3 | `07b_gerar_sessoes_sinteticas` (semente 2026) | `coleta_sim/SIM001` a `SIM006`, `planilha_eventos.csv`, `parametros_geracao.json` |
| 4 | `08_verificar_sessao`, `09_linha_base_acustica` | conferência das sessões e `base.csv` |
| 5 | `10_rotular_sessoes` | `coleta_sim/rotulado/` (SIM006: 1.998 decisões, 12 eventos) |
| 6 | `11b_busca_hiperparametros` | modelos finais v2 em `modelos/sim/` |
| 7 | `12_avaliacao_H1_H2` | Tabela 13, Figura 19, IC por bootstrap |
| 8 | `14_manifesto_SIM006` | `manifesto/` (hashes e reprodução da Tabela 13) |
| 9 | `07b_verificacao` | `manifesto/origem/` (origem dos trechos e comparação de hashes) |

Os notebooks `06`, `07` e `13` geram figuras e o áudio de fundo da demonstração, sem novo treino. A relação completa de arquivos está na Tabela C1 (Apêndice C).

**Firmware:** abra `Multimodais-de-Monitorizacao_v5.ino`, copie `secrets.h.example` para `secrets.h`, preencha as credenciais da rede e compile para o Nano 33 BLE Sense. Para verificar paridade e latência, carregue `Teste_Inferencia_v5.ino` e acompanhe o Monitor Serial (Apêndice D, Figura D1).

---

## Reprodutibilidade da sessão de teste (SIM006)

- **Geração:** notebook 07b, semente 2026. Cada partição das sessões usa só a partição correspondente do MIMII e do Bristol.
- **Verificação:** a regeneração produziu **361 de 361 arquivos idênticos** (SHA-256) às sessões usadas na dissertação. Nenhuma gravação ou trecho foi usado fora da sua partição nem compartilhado entre treino, validação e teste.
- **Limitação:** dentro de cada sessão, os clipes do MIMII são reutilizados com início e ganho aleatórios. Na SIM006 isso dá, em média, 14,9 usos por clipe normal e 9,0 por clipe anômalo.
- **Avaliação:** o bootstrap é pareado, por eventos e trechos entre eventos, com 2.000 reamostragens, semente 0 e método percentil. A Tabela 13 é recalculada pelo manifesto a partir das predições salvas.
- **Hiperparâmetros selecionados (validação):** taxa da CNN 3 × 10⁻⁴; ponderação das classes inversa; 4.000 janelas normais por sessão; ponderação da fusão pela raiz; δ = 0,75 no viés da classe Normal.

---

## Segurança e privacidade

- `secrets.h`, credenciais de Wi-Fi, token do Telegram e senha de app do Gmail **não são versionados**. As credenciais de alerta ficam apenas no n8n.
- O questionário é publicado só como instrumento. As respostas dos participantes não fazem parte do repositório.

---

## Como citar

```
SILVA, Júlio César da. Detecção Embarcada de Anomalias em Data Centers por Fusão de Dados
Ambientais e Acústicos: Desenvolvimento e Avaliação de um Artefato TinyML. 2026.
Dissertação (Master of Science in Artificial Intelligence) – AGTU University, Florida, 2026.
```

## Licença

| Conteúdo | Licença |
|---|---|
| Código-fonte (firmware, notebooks, receptor, dashboard e scripts) | [MIT](https://opensource.org/license/mit) |
| Textos, figuras, esquemáticos e questionário | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.pt-br) |
| Modelos treinados e dados derivados (manifesto, predições, rótulos) | Sujeitos às licenças dos datasets de origem; uso **não comercial** |

Os modelos e os dados derivados foram produzidos a partir do MIMII (CC BY-SA 4.0) e do conjunto de Bristol (Non-Commercial Government Licence v2). Por isso, seguem as condições dessas licenças: atribuição aos autores originais, compartilhamento pela mesma licença no caso do MIMII e vedação de uso comercial no caso do Bristol. Os datasets de origem não são redistribuídos neste repositório.

```
MIT License

Copyright (c) 2026 Júlio César da Silva

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

O texto acima também deve ser salvo como arquivo `LICENSE` na raiz do repositório.
