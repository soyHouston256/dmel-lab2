# DMEL · Laboratorio 2 · UTEC

Predicción de **48 horas de caudal específico (mm/h)** a partir de **336 horas de historia y 12 variables**. Implementación propia en PyTorch, adaptada de [Wang et al. (2026)](https://doi.org/10.1016/j.asoc.2026.114792): CEEMDAN, agrupación por entropía y combinación de Informer con LSTM.

**Estado al 03/10/2026:** LSTM y DMEL entrenados y evaluados en Khipu; comparación y presentación disponibles. El respaldo completo de ambas corridas está verificado localmente y se transfiere por separado de GitHub.

**Entregables:** [PowerPoint UTEC · 22 diapositivas](outputs/presentacion/DMEL-UTEC-comparacion-LSTM-DMEL.pptx) · [Guion](outputs/presentacion/guion-dmel-comparacion.txt) · [Resultados y evidencia](outputs/README.md).

## Inicio rápido

Desde la raíz del repositorio, con Python **3.12**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m dmel.cli --help
python -m pytest tests -q
python -m scripts.smoke --out runs/synthetic-smoke
```

Las pruebas y el ejemplo `smoke` usan datos sintéticos: funcionan sin GPU, acceso a Khipu ni dataset del laboratorio. El ejemplo exige un directorio de salida nuevo. Ejecutar los comandos desde la raíz; el proyecto no requiere `pip install -e .`.

Para elegir el entorno CPU/CUDA, ver [instalación](docs/instalacion.md). Para usar los datos y pesos existentes en Linux, seguir [migración a la workstation](docs/migracion-workstation.md).

## Resultados disponibles

Validación de **18 142 ventanas × 48 horizontes = 870 816 pares**, en **508 cuencas**. Ambos modelos usaron semilla 42, completaron 20 épocas por parada temprana y seleccionaron la época 10 (`epoch=9`).

| Métrica | LSTM | DMEL |
|---|---:|---:|
| RMSE global (mm/h) ↓ | **0.102283** | 0.105666 |
| MAE global (mm/h) ↓ | **0.021275** | 0.022268 |
| NSE global ↑ | **0.644696** | 0.620803 |
| KGE 2009 global ↑ | **0.722057** | 0.718808 |
| Predicciones negativas | 3.2556 % | 1.0839 % |
| Cuencas con menor RMSE | 256 | 252 |

DMEL obtuvo un RMSE global **3.31 % mayor** que LSTM; LSTM tuvo menor RMSE en los 48 horizontes. DMEL redujo la fracción de predicciones negativas. Las [métricas LSTM](outputs/khipu-lstm-53987-completed/metrics.json), [métricas DMEL](outputs/khipu-dmel-54181-completed/metrics.json) y [comparación detallada](outputs/khipu-dmel-54181-completed/analysis.json) conservan el desglose completo.

Son resultados de una semilla sobre la validación empleada para seleccionar los checkpoints, sin recortar predicciones a cero. La agregación reúne pares muestra/horizonte, no una serie cronológica reconstruida. Las medias de NSE por cuenca contienen valores negativos extremos; consultar también las medianas y los casos individuales.

## Estructura

```text
dmel/                 Modelos, datos, descomposición, entrenamiento, métricas y CLI
configs/              Configuraciones finales y diagnósticas documentadas
dataset/              Contrato y lector del dataset; binarios fuera de Git
scripts/              Prueba sintética, gráficos, benchmark y transferencia
slurm/                Jobs y envío de experimentos en Khipu
tests/                Pruebas de causalidad, inferencia, reanudación y Slurm simulado
docs/                 Instalación, metodología, migración y registro de Khipu
outputs/              Métricas, informes, figuras y presentación seleccionados
.github/workflows/    Pruebas automáticas en CPU
```

`cache/`, `runs/`, `logs/` y `migration/` son almacenamiento local ignorado. GitHub contiene las fuentes y los entregables seleccionados; los HDF5, pesos, entornos y respaldos se mantienen aparte. Ver [datos](dataset/README.md) y [preparación de GitHub](docs/github.md).

## Documentación

- [Índice de guías](docs/README.md): instalación, migración y ejecución.
- [Implementación y alcance científico](docs/implementation.md).
- [CEEMDAN, SampEn y simplificación de MRS](docs/decomposition.md).
- [Configuraciones de los experimentos](configs/README.md).
- [Scripts disponibles](scripts/README.md) y [ejecución con Slurm](slurm/README.md).
- [Catálogo de resultados e informes históricos](outputs/README.md).

## Alcance y pendientes

Esta es una **adaptación al dataset horario del laboratorio**: utiliza dos grupos por ventana y una pérdida conjunta sobre caudal. No afirma reproducir las agrupaciones ni las cifras del artículo. La normalización se ajusta únicamente con train y se respetan los flags oficiales. `test_targets.csv` no participa del entrenamiento ni de la selección.

Quedan por completar la referencia de persistencia, la ablación MRS–Informer, el análisis de predicciones individuales, las predicciones finales de test y el video de exposición. El repositorio conserva los resultados obtenidos hasta ahora; la prueba sintética solo verifica el funcionamiento del pipeline.
