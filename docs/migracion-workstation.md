# Traslado a la workstation Linux

El código y los entregables viajan por GitHub. Los datos, cachés y checkpoints viajan en el respaldo `migration/khipu/`, excluido de Git.

## 1. Copiar el respaldo

El respaldo local contiene 68 archivos inventariados (aproximadamente 6.51 GiB), más logs diagnósticos. Incluye `dataset/train.h5`, `dataset/test.h5`, `cache/train.h5` y las corridas completas `lstm-seed42-fd-fix` y `dmel-seed42`. Se conserva también una copia de las fuentes usadas en Khipu.

Copiar la carpeta `migration/khipu/` completa al mismo lugar dentro del clon en Linux, mediante disco externo o una transferencia ejecutada por el usuario. No copiar la antigua `.venv`. Conservar `manifest.json` y `VERIFIED.json`; el primero describe el contenido y el segundo registra su última verificación.

Después de copiar, desde la raíz del clon:

```bash
python3 scripts/pull_khipu.py --verify-only --dest migration/khipu
```

Esta orden solo lee la copia local y renueva su recibo de verificación; no abre SSH. Las corridas incompletas `lstm-seed42` y `pilot-full` fueron omitidas y constan en el inventario. El respaldo no acredita esas corridas.

El [registro de comprobación](../outputs/verification/khipu-backup-20261003.json) documenta el cotejo de las fuentes y las métricas con los archivos recibidos anteriormente.

## 2. Preparar el entorno

Seguir [instalación](instalacion.md), elegir CPU o NVIDIA/CUDA según el equipo y ejecutar las pruebas. Para CUDA, comprobar tanto `environment` como una operación real en la GPU. No se ha medido todavía el rendimiento de esta workstation.

Se pueden usar directamente las rutas del respaldo, sin duplicar los HDF5:

```bash
DMEL_BACKUP="$PWD/migration/khipu"
python -m dmel.cli inspect --data "$DMEL_BACKUP/dataset/train.h5"
```

Las órdenes siguientes usan esa variable en la misma terminal y un entorno con CUDA verificada. Para comprobar solo en CPU, sustituir `--device cuda` por `--device cpu`.

## 3. Usar los checkpoints existentes

Ambas corridas ya terminaron. Para volver a evaluar sus mejores pesos sin sobrescribir la evidencia original:

```bash
python -m dmel.cli evaluate --data "$DMEL_BACKUP/dataset/train.h5" --run "$DMEL_BACKUP/runs/lstm-seed42-fd-fix" --out runs/workstation-lstm-validation --device cuda
python -m dmel.cli evaluate --data "$DMEL_BACKUP/dataset/train.h5" --run "$DMEL_BACKUP/runs/dmel-seed42" --cache "$DMEL_BACKUP/cache/train.h5" --out runs/workstation-dmel-validation --device cuda
```

`evaluate` lee `best.pt`, la normalización y la configuración guardadas. La evaluación de validación comprueba el hash del dataset. El entorno de otro equipo puede introducir pequeñas diferencias numéricas; conservar la evidencia original y registrar la nueva por separado.

**Reanudar entrenamiento es distinto de evaluar.** La comprobación de `--resume` exige la misma configuración efectiva, incluidos los hashes y la ruta absoluta del dataset. Mover una corrida desde `/home/max.ramirez/dmel` puede hacerla incompatible para reanudar. No editar `config.json` ni sus hashes para eludirlo. Esta migración conserva las corridas terminadas y permite evaluarlas; una continuación de entrenamiento entre rutas diferentes requeriría un cambio explícito del mecanismo de reanudación.

## 4. Completar experimentos pendientes

Persistencia usa CPU y no necesita caché:

```bash
python -m dmel.cli persistence --data "$DMEL_BACKUP/dataset/train.h5" --out runs/persistence-workstation/validation
```

La ablación puede entrenarse en una carpeta nueva, reutilizando la caché completa:

```bash
python -m dmel.cli train --data "$DMEL_BACKUP/dataset/train.h5" --cache "$DMEL_BACKUP/cache/train.h5" --config configs/mrs_informer.json --out runs/mrs-informer-workstation-seed42 --device cuda
```

En Linux local no hace falta Slurm ni reencolar después de cada época: la CLI usa `--epochs-per-cycle 0` por defecto y mantiene los checkpoints. Estas órdenes son instrucciones para el usuario; no se han ejecutado como parte de la preparación de GitHub.

Para predecir test con LSTM basta `predict` y su checkpoint. DMEL o MRS–Informer necesitan primero una caché de **test** construida con `configs/decomposition.json`; `cache/train.h5` no sirve para test. El formato de entrega final debe confirmarse antes de generar el archivo para enviar al curso.
