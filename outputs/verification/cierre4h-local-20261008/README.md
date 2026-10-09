# Cierre experimental verificado · 8 de octubre de 2026

La ablación y las predicciones del test real están completas y verificadas en el Mac. El usuario ejecutó los trabajos y la descarga desde Khipu; esta auditoría solo leyó los archivos locales. El [resumen estructurado](summary.json) reúne resultados, hashes y procedencia. Las fechas UTC de los informes son del 9 de octubre; corresponden a la noche del 8 en Lima.

## Ablación de especialización

Dos entrenamientos **desde cero, de tres épocas cada uno**: control DMEL (Informer en alta, LSTM en baja) y MRS–Informer (Informer en ambas). Se conservaron representación CEEMDAN/MRS, suma de ramas, 254 000 muestras de entrenamiento, 18 142 de validación, normalización, semilla 42, batch 32 y Adam a 1e-4. Ambos mejores checkpoints corresponden a `epoch=2`, la tercera época. No se compara la variante de tres épocas con el DMEL original de veinte épocas.

| Resultado de validación | DMEL, control | MRS–Informer |
|---|---:|---:|
| RMSE (mm/h), menor es mejor | 0.110024 | 0.111464 |
| MAE (mm/h), menor es mejor | 0.024031 | 0.027545 |
| NSE, mayor es mejor | 0.588882 | 0.578049 |
| KGE 2009, mayor es mejor | 0.695402 | 0.708342 |
| Sesgo (mm/h) | −0.002579 | +0.007133 |
| Predicciones negativas | 8.65 % | 3.02 % |
| Parámetros | 3 648 561 | 5 666 818 |
| Tiempo acumulado de épocas (min) | 31.46 | 33.71 |

Sustituir LSTM por Informer aumentó el RMSE en **0.00144012 mm/h**, equivalente a **1.31 % respecto al control DMEL**. MAE y NSE favorecen al control; KGE y la fracción de negativos favorecen a la variante. La diferencia no establece superioridad en todas las métricas ni significancia estadística. Solo hay una semilla, cambia la capacidad y tres épocas no demuestran convergencia. La validación se usó para seleccionar los checkpoints. Se agregan 870 816 pares muestra/horizonte, no una serie cronológica reconstruida.

El [informe completo de ablación](ablation.json) recalcula métricas globales, por horizonte y por cuenca desde los NPZ. También contrasta configuraciones, hashes de datos/código/caché, particiones, objetivos, cuencas, normalizador, estados y carga estricta de pesos. Las configuraciones, historiales y metadatos se conservan en [dmel-3epochs/](dmel-3epochs/) y [mrs_informer-3epochs/](mrs_informer-3epochs/), copiados sin modificar desde la descarga.

## Test completo

| Exportación | Muestras | Horizontes | Valores | Verificación |
|---|---:|---:|---:|---|
| LSTM original, mejor época 10 | 27 983 | 48 | 1 343 184 | [Completa](test-lstm.json) |
| DMEL original, mejor época 10 | 27 983 | 48 | 1 343 184 | [Completa](test-dmel.json) |

Se verificaron las 49 columnas `Id,q_01,…,q_48`, los IDs 0–27 982 en orden y sin duplicados, correspondencia de cuencas, valores finitos, igualdad numérica CSV/NPZ y hash del HDF5 fuente. Ambos CSV coinciden con los SHA-256 reportados por el usuario desde Khipu. Los datos están en mm/h y se conservaron sin clipping. Son predicciones de los modelos originales, no de los nuevos controles de tres épocas.

**No son métricas de test.** No se usó `test_targets.csv` para esta verificación ni para ajustar o seleccionar modelos. El formato CSV está documentado por el proyecto; confirmar si la entrega requiere otro esquema oficial.

Archivos de entrega locales, excluidos de Git junto con el respaldo:

- CSV LSTM: `migration/khipu-cierre/runs/cierre-4h-seed42-20261008/test-lstm/predictions.csv` (desde la raíz local).
- CSV DMEL: `migration/khipu-cierre/runs/cierre-4h-seed42-20261008/test-dmel/predictions.csv` (desde la raíz local).

El [manifiesto de descarga](download-manifest.json) registra 40 archivos (474 561 757 bytes), incluidos checkpoints y caché de test. La caché está completa, cubre los 27 983 IDs, contiene componentes finitos `[27983,336,2]` y corresponde al hash de test. Los hashes del manifiesto preservan la identidad de los archivos locales; no existe un manifiesto remoto de todos los archivos para cotejarlos uno por uno.

## Ejecución y límites operativos

El usuario compartió `sacct` con `COMPLETED 0:0` para LSTM test (54931, 32 s), caché de test (54932, 2 h 11 min 57 s) y DMEL test (54935, 43 s). El job de ablación 54933 dejó ambos estados y artefactos completos y el log `CONTROL Y ABLACION TERMINADOS`. No se recibió su `sacct` final; no se inventa su código de salida ni duración total. Los 65.17 minutos de entrenamiento son la suma de tiempos registrados en épocas y excluyen cola y otros pasos.

La [auditoría actualizada de la guía](../laboratorio-2-cierre-verificado-20261008.json) registra requisitos experimentales y video completados localmente; la publicación y entrega quedan a cargo del propietario. La presentación vigente es [DMEL-UTEC-lab2.pptx](../../presentacion/DMEL-UTEC-lab2.pptx): conserva 16–19, añade ablación en 20 y test en 21, actualiza conclusiones en 22; referencias en 23–24. El [video final](../../video-dmel/README.md) contiene las cuatro intervenciones y el cierre, con una duración de 17:49.84.

## Repetir solo la auditoría local

Desde la raíz, con el entorno del proyecto y los respaldos locales presentes:

```bash
python -m scripts.lab2_closure compare-ablation \
  --reference migration/khipu-cierre/runs/cierre-4h-seed42-20261008/dmel-3epochs \
  --ablation migration/khipu-cierre/runs/cierre-4h-seed42-20261008/mrs_informer-3epochs \
  --out outputs/verification/cierre4h-local-20261008/ablation.json

python -m scripts.lab2_closure verify-test \
  --data migration/khipu/dataset/test.h5 \
  --run migration/khipu/runs/lstm-seed42-fd-fix \
  --predictions migration/khipu-cierre/runs/cierre-4h-seed42-20261008/test-lstm \
  --out outputs/verification/cierre4h-local-20261008/test-lstm.json

python -m scripts.lab2_closure verify-test \
  --data migration/khipu/dataset/test.h5 \
  --run migration/khipu/runs/dmel-seed42 \
  --predictions migration/khipu-cierre/runs/cierre-4h-seed42-20261008/test-dmel \
  --out outputs/verification/cierre4h-local-20261008/test-dmel.json
```

No hace falta volver a entrenar ni regenerar las exportaciones verificadas.

La [verificación de la PPT](presentation-qa.json) registra 24 diapositivas renderizadas, revisión visual, seis gráficos y libros originales preservados, y cero hallazgos de estructura o geometría. No se realizó una prueba nativa de editar/guardar/reabrir en PowerPoint.
