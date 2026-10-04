# Resultados, informes y entregables

Este directorio contiene evidencia seleccionada para GitHub. Se conservan las rutas originales porque están citadas desde los informes y las notas de la presentación. Las ejecuciones completas, datos y pesos viven en `runs/` o `migration/`, fuera de Git.

## Entrega vigente

| Artefacto | Contenido |
|---|---|
| [Presentación UTEC](presentacion/DMEL-UTEC-comparacion-LSTM-DMEL.pptx) | 22 diapositivas: metodología, dataset y comparación LSTM–DMEL |
| [Guion de exposición](presentacion/guion-dmel-comparacion.txt) | Notas y referencias de la presentación vigente |
| [LSTM 53987](khipu-lstm-53987-completed/metrics.json) | Validación global, 48 horizontes y 508 cuencas |
| [DMEL 54181](khipu-dmel-54181-completed/metrics.json) | Mismo desglose para DMEL |
| [Comparación de modelos](khipu-dmel-54181-completed/analysis.json) | Diferencias, conteos por horizonte/cuenca y limitaciones |
| [Curva de LSTM](khipu-lstm-53987-learning-20261003.png) | Aprendizaje y duración por época; también disponible en PDF |
| [Verificación del repositorio](verification/README.md) | Respaldo, integridad y pruebas de preparación |

Las carpetas `*-completed/` incluyen `status.json`, `sacct.txt`, `provenance.json` y análisis. Sus registros de procedencia describen lo conocido al extraer las salidas del usuario. Campos como `checkpoints_downloaded: false` son históricos: la descarga posterior está documentada en [la verificación del respaldo](verification/khipu-backup-20261003.json). Las métricas se cotejaron con esa copia sin sustituir los originales.

## Informes de investigación y diseño

| Documento | Alcance |
|---|---|
| [Revisión del laboratorio](revision-laboratorio-2.md) | Requisitos de la entrega y contrato horario |
| [Revisión de DMEL](revision-dmel.md) | Arquitectura, artículo y límites de la evidencia |
| [Fuentes de implementación](fuentes-implementacion.md) | Repositorios y fuentes consultadas |
| [Especificación inicial](especificacion-implementacion.md) | Diseño previo; el contrato horario vigente está en `docs/implementation.md` |
| [Auditoría de reproducibilidad](auditoria-reproducibilidad.md) | Decisiones e incógnitas de la adaptación |
| [Configuración del artículo](dmel-paper-config.json) | Parámetros publicados y no especificados |
| [Revisión del dataset](revision-dataset-lab2.json) | Inspección inicial |
| [Procedencia de datos](dataset-source.json) | Fuente del laboratorio y descarga inicial |
| [Desarrollo inicial](desarrollo-v1.md) | Estado del pipeline al finalizar esa etapa |
| [Plan de Khipu](plan-khipu.md) | Dimensionamiento y planificación previos a las corridas |

Los archivos `*.provenance.md` acompañan los informes correspondientes. Las fechas de estos documentos importan: no interpretar sus pendientes como estado actual.

## Evidencia histórica

- `khipu-precompute-complete.json`: caché de 272 142 ventanas terminada.
- `khipu-lstm-failed-53814.json`: fallo de inferencia corregido posteriormente.
- `khipu-lstm-53987-epoch*.json`: capturas de avance.
- `khipu-lstm-53987-history-20261003.json` y `*-analysis-20261003.json`: historial y análisis recibidos.
- `khipu-dmel-54181-submitted.json`: constancia de envío, reemplazada como estado por `khipu-dmel-54181-completed/`.
- `tests-*.xml`: reportes de etapas anteriores; `tests-inference-workers.xml` procede de un proceso interrumpido tras sus aserciones, como explica la [recuperación](../docs/recuperacion-lstm-53814.md).
- [Versiones de la presentación](presentacion/README.md): versiones previas conservadas con su alcance.

`sources/` contiene material original extraído y está ignorado. Para nuevas ejecuciones usar `runs/`; incorporar a `outputs/` solo los resultados seleccionados con fecha, procedencia y alcance.
