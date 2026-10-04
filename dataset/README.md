# Dataset del Laboratorio 2

Git conserva [metadata.json](metadata.json) y el [lector proporcionado](leer_datos.py). Los HDF5 y CSV de datos se distribuyen por separado; `.gitignore` los excluye.

| Archivo | Uso | Incluido en Git |
|---|---|---|
| `train.h5` | 254 000 muestras train y 18 142 validation, según flags | No |
| `test.h5` | 27 983 muestras para predicción final | No |
| `test_targets.csv` | No se usa para entrenamiento ni selección | No |
| `metadata.json` | Contrato del dataset | Sí |
| `leer_datos.py` | Lector de referencia del laboratorio | Sí |

Cada entrada tiene forma `[336, 12]`; el canal 11 contiene caudal específico y los anteriores, meteorología histórica. La salida tiene 48 pasos horarios en mm/h. Los flags son `0=train`, `1=validation`; los IDs de cuenca son anónimos. No hay fechas suficientes para reconstruir una cronología a partir del ID de fila. La normalización del modelo se ajusta únicamente con train.

## Obtener y comprobar los archivos

La [carpeta del laboratorio](https://drive.google.com/drive/folders/1crKbJBhLHQEVJGG-agOxMnKLDnBP2x4w) figura en el [registro de procedencia](../outputs/dataset-source.json). Ese JSON conserva el estado de la descarga inicial del 26/09; no representa la disponibilidad actual.

Al 03/10, el respaldo local verificado contiene `migration/khipu/dataset/train.h5`, `migration/khipu/dataset/test.h5` y `migration/khipu/cache/train.h5`. En un clon nuevo estos archivos no aparecen: copiar el respaldo siguiendo la [guía de migración](../docs/migracion-workstation.md).

```bash
python -m dmel.cli inspect --data migration/khipu/dataset/train.h5
python -m dmel.cli inspect --data migration/khipu/dataset/test.h5
```

Firmas confirmadas y verificadas en el respaldo:

| Artefacto | SHA-256 |
|---|---|
| `dataset/train.h5` | `38a1e2dee5f9aa61f5e1543dc6f7c22689f6ee3059511498022df31f01730078` |
| `cache/train.h5` | `e8d304ca15b9f33eb5eb01bba82c32caffa004bc94bef2c183b038c57d7987e1` |

La caché completa contiene las 272 142 ventanas supervisadas. Se puede reutilizar para DMEL y MRS–Informer con la misma descomposición; LSTM y persistencia usan la señal original. La caché de test es un artefacto distinto y aún debe generarse si se elige un modelo que la necesite.
