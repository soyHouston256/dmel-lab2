# Implementación del Laboratorio 2

Estado 08/10/2026: corridas principales LSTM 53987 y DMEL 54181 terminadas; control DMEL y ablación MRS–Informer entrenados durante tres épocas; exportaciones completas de test para los dos modelos originales. Los nuevos artefactos se descargaron y [verificaron localmente](../outputs/verification/cierre4h-local-20261008/README.md). El [catálogo de evidencia](../outputs/README.md) enlaza resultados y PPT actualizada. Es una **adaptación de DMEL**, no una reproducción de sus cifras.

## Datos y causalidad

Cada muestra usa X `[336,12]` y produce 48 caudales en mm/h. Canal 11: caudal; 0–10: meteorología histórica. Se respetan flags 0=train, 1=validation. `basin_id` se conserva para evaluación; no se inventan fechas desde Id. HDF5 se abre por proceso y se lee por filas. La normalización usa exclusivamente X de train, con momentos en float64. `y_aux` y `test_targets.csv` no se usan como entradas ni para seleccionar modelos.

CEEMDAN recibe solo `X[i,:,11]`; no concatena filas o cuencas. La caché conserva Id, componentes, diagnósticos, SHA-256 del HDF5, parámetros, hash de implementación y versiones NumPy/SciPy. Un único escritor HDF5 recibe resultados de una cola acotada a 2×workers. Las cachés `.partial` se reanudan si sus firmas coinciden; un archivo corrupto requiere revisión y no se garantiza recuperación frente a cualquier fallo de almacenamiento.

## Decisiones frente al artículo

| Elemento | Implementación | Alcance |
|---|---|---|
| CEEMDAN | Propio basado en Torres 2011; 100 realizaciones, ruido relativo 0.2 | Bordes/sifting/ruido por etapa son decisiones explícitas |
| SampEn | m=2, r=0.2×SD, Chebyshev, pares sin autocoincidencias | Degenerados con estado explícito |
| Agrupación | Dos sumas: IMFs con entropía mayor que la señal y resto+residuo | Simplifica MRS; omite fusión intermedia en varias RIMFs por similitud |
| Rama alta | ProbSparse, reducción temporal del encoder y decoder causal | Implementación propia de mecanismos principales de Informer |
| Rama baja | LSTM dos capas + cabeza de 48 pasos | Cabeza directa elegida para el laboratorio |
| Entradas | 11 meteorológicas históricas + componente de la rama | Adaptación multivariada horaria |
| Objetivo | Suma de salidas y MSE conjunta frente al caudal | Contribuciones individuales no identificables; no se inventan etiquetas futuras por IMF |
| Persistencia | Cada cabeza corrige el último valor de su entrada | Modificación compartida con baselines |
| Posición | Sinusoidal relativa y proyección lineal | Sin calendario, porque no hay fechas |
| Optimización | Adam 1e-4, batch 32, máximo 100 épocas, FP32 | Paciencia 10, clipping 1 y dropout 0.1 son decisiones adicionales |

La rama Informer conserva atención ProbSparse, reducción del encoder y salida multihorizonte en un paso descritas por [Zhou et al.](https://arxiv.org/abs/2012.07436). La importancia de queries se estima por máximo menos media de scores muestreados. En evaluación el muestreo es fijo, independiente del orden de batches. La selección discreta se calcula por bloques sin gradiente para controlar memoria. El decoder de una capa recibe últimas 48 entradas conocidas y 48 vectores de ceros, con self-attention causal y cross-attention completa.

Se sustituyen embedding convolucional y normalización del bloque distilling originales por proyección lineal y Conv1D/ELU/MaxPool. No se importa código descargado de Informer o PyEMD. PyTorch aporta operadores y capa decoder; el ensamble, ProbSparse, descomposición, agrupación y pipeline están en este proyecto. [decomposition.md](decomposition.md) documenta CEEMDAN/SampEn y sus fuentes. Alta/baja significa complejidad, no banda física fija.

La escala conserva la identidad `high/σ + (low−μ)/σ = (q−μ)/σ`; se multiplica la salida sumada por σ y se añade μ **una sola vez**. No hay clipping: las métricas registran predicciones negativas. Componentes individuales pueden ser negativos.

## Experimentos y presupuestos

| Configuración | Pregunta | Parámetros |
|---|---|---:|
| Persistencia | ¿Superamos repetir último caudal? | 0 |
| `lstm.json` | ¿Superamos LSTM multivariada cruda? | 815152 |
| `dmel.json` | ¿Sirve combinar grupos y predictores? | 3648561 |
| `mrs_informer.json` | ¿Sirve especializar la rama baja con LSTM? | 5666818 |

La comparación principal LSTM–DMEL comparte máximo 100 épocas y paciencia 10; ambos completaron 20 y seleccionaron la décima. La ablación ejecutada utiliza `dmel-3epochs.json` y `mrs_informer-3epochs.json`: dos entrenamientos nuevos desde cero, máximo 3 épocas y paciencia 3; ambos seleccionaron la tercera. Dentro de cada comparación se mantienen split, seed 42, normalización y loss. No comparar sus presupuestos como si fueran iguales entre los dos experimentos. La ablación conserva representación y suma, pero cambia capacidad. Una semilla y tres épocas no demuestran robustez ni convergencia; ampliar requiere configs y nombres de corrida nuevos. Persistencia queda como extensión. Las configuraciones `smoke` son diagnósticas, no resultados publicables.

## Evaluación y artefactos

Se selecciona `best.pt` por RMSE global de validación en mm/h. Se exportan RMSE, MAE, sesgo, NSE y KGE2009: global, cada horizonte y cada cuenca; también media por cuenca y cantidad de métricas definidas. Casos indefinidos son `null`. La agregación cuenta pares muestra/horizonte; sin fechas no es una serie temporal continua ni permite auditar el solapamiento original. La métrica oficial sigue pendiente de confirmar.

Cada corrida guarda configuración efectiva, hashes de código/datos/cache, versiones, normalización, ids, historial, `best.pt` y `last.pt`. Reanudar restaura Adam, época, paciencia y RNG PyTorch/CUDA/DataLoader; rechaza cambios de configuración, código o datos. No promete igualdad bit a bit entre CPU/GPU. CSV: Id original y `q_01`…`q_48`, formato propuesto pendiente de confirmación oficial. Test sin etiquetas solo genera predicciones. Resultados sintéticos se guardan bajo `runs/synthetic-*`.

La ablación arroja RMSE 0.110024 para el control y 0.111464 mm/h para MRS–Informer (+1.31 %); KGE y la fracción de negativos favorecen a la variante. Se verificaron 27 983 × 48 predicciones de test por modelo original, sin calcular precisión en test. El [video grupal](../outputs/video-dmel/README.md) está terminado y verificado localmente, con duración de 17:49.84. Persistencia, más semillas y análisis detallado de eventos son extensiones; la ausencia de fechas limita la auditoría de solapamiento temporal. Las pruebas del código no acreditan por sí mismas ventaja hidrológica.
