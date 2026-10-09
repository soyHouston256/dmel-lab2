# Cierre del Laboratorio 2

**Cierre completado y verificado el 08/10/2026:** control DMEL y ablación MRS–Informer, tres épocas cada uno; test real completo de los dos modelos originales. El usuario descargó los artefactos y la auditoría local pasó. Consultar [resultados, límites y verificación](../outputs/verification/cierre4h-local-20261008/README.md) y [PPT actualizada](../outputs/presentacion/DMEL-UTEC-lab2.pptx).

No hace falta ejecutar nuevamente los entrenamientos ni las predicciones. El [video grupal final](../outputs/video-dmel/README.md) ya integra las cuatro intervenciones y el cierre, dura 17:49.84 y conserva su respaldo editable de Resolve. La publicación de los cambios y la entrega corresponden al propietario; ver [GitHub](github.md).

## Plan original conservado como referencia

Los comandos siguientes pertenecen al plan largo previo, no son los jobs finalmente ejecutados ni pasos pendientes. El plazo de cuatro horas hizo que se sustituyera por el [plan reducido para Khipu](cierre-khipu-4h.md): control y ablación de tres épocas cada uno. Una ejecución futura con presupuesto largo debe usar rutas nuevas y verificarse por separado. El usuario ejecuta los comandos remotos; el agente no abre SSH.

## Dos entregables concretos

1. **Ablación de especialización:** entrenar `mrs_informer` desde cero y comparar su mejor checkpoint con DMEL sobre las mismas 18 142 ventanas de validación. Se cambia la rama baja de LSTM a Informer. Permanecen CEEMDAN/MRS, datos, semilla 42, batch 32, Adam a 1e-4, 100 épocas máximas y paciencia 10. DMEL tiene 3 648 561 parámetros; la ablación, 5 666 818. Esa diferencia de capacidad limita la atribución del efecto a la arquitectura.
2. **Test completo:** exportar 27 983 filas, cada una con `Id,q_01,...,q_48` en mm/h, sin clipping. Generar DMEL y también LSTM, que obtuvo menor RMSE de validación. Elegir cualquier modelo de entrega por validación, nunca por `test_targets.csv`. La guía no especifica un esquema externo de envío; este es el formato documentado por nuestra CLI.

El informe `outputs/verification/cierre-lab2-preflight-20261008.json` verifica hashes de train y caché, particiones, código, versiones de descomposición y carga estricta de ambos checkpoints. No acredita entrenamiento nuevo.

## Workstation Linux

Trasladar el repositorio **y por separado** `migration/khipu/`, siguiendo [migración](migracion-workstation.md). GitHub no contiene los HDF5 ni los pesos. Activar el entorno PyTorch/CUDA descrito en [instalación](instalacion.md) y ejecutar desde la raíz:

```bash
source .venv/bin/activate
python -m dmel.cli environment
python -m scripts.lab2_closure preflight --out outputs/verification/preflight-workstation.json

# Primera entrega rápida: LSTM no necesita descomposición de test.
bash scripts/run_lab2_closure.sh test-lstm

# Ablación real: comienza desde cero; repetir reanuda desde last.pt.
bash scripts/run_lab2_closure.sh ablation

# Nueva caché de las 27 983 historias y predicciones del DMEL ya entrenado.
DMEL_PRECOMPUTE_WORKERS=8 bash scripts/run_lab2_closure.sh test-dmel
```

Las etapas se pueden ejecutar en sesiones separadas. Ablación y descomposición comparten CPU/RAM; ajustar el número de workers con una medición en ese equipo. No aumentar batch, cambiar semilla ni modificar el modelo para acelerar esta comparación. El número de workers de precompute no cambia el experimento.

Variables opcionales: `DMEL_PYTHON`, `DMEL_INPUT_ROOT` (directorio con dataset/cache/runs originales), `DMEL_CLOSURE_TAG` (nombre nuevo de la entrega), `DMEL_DEVICE` (`cuda` por defecto, `cpu` solo si se eligió CPU). Los logs quedan en `logs/closure-seed42-20261008-*.log`.

Para ciclos interrumpibles, `DMEL_EPOCHS_PER_CYCLE=1` pausa tras una época; `DMEL_PRECOMPUTE_MAX_SECONDS=1800` pone un presupuesto suave a la caché. Ambos terminan con código **75** mientras falte trabajo. Repetir la misma etapa reanuda; conservar rutas, entorno y código. No mover una corrida parcial entre equipos esperando reanudación idéntica. La finalización se confirma por los informes, no por la existencia de un checkpoint.

## Khipu: ejecución manual del usuario

El agente no abre SSH. El usuario copia las nuevas fuentes/scripts a Khipu y ejecuta allí desde `/home/max.ramirez/dmel`, en su entorno habitual. Allí los originales ya están en `dataset/`, `cache/` y `runs/`; usar `--root .` en preflight.

```bash
source .venv/bin/activate
python -m scripts.lab2_closure preflight --root . --out outputs/verification/preflight-khipu-cierre.json

# Ablación con los wrappers y mecanismo de requeue existentes.
sbatch --job-name=lab2_ablation --gres=shard:rtxa6000:4 --time=08:00:00 \
  --export=ALL,DATA=dataset/train.h5,CONFIG=configs/mrs_informer.json,OUT=runs/mrs-informer-closure-seed42-20261008,CACHE=cache/train.h5,RESUME=0,EPOCHS_PER_CYCLE=1,AUTO_REQUEUE=1,MAX_RESTARTS=99 slurm/train.sbatch

# Exportación LSTM: emplea el modelo ya entrenado.
sbatch --job-name=lab2_test_lstm \
  --export=ALL,DATA=dataset/test.h5,RUN=runs/lstm-seed42-fd-fix,CACHE= slurm/predict.sbatch

# Descomposición completa de test (CPU). LIMIT=0 es esencial.
sbatch --job-name=lab2_test_cache --time=08:00:00 \
  --export=ALL,DATA=dataset/test.h5,CONFIG=configs/decomposition.json,OUT=cache/test-closure-20261008.h5,SPLIT=test,LIMIT=0,SELECTION_CONFIG=,MAX_SECONDS=24000,AUTO_REQUEUE=1,MAX_RESTARTS=99 slurm/precompute.sbatch
```

Consultar los logs y `squeue --me`. El precompute puede abarcar varios ciclos. **Cuando exista la caché final `.h5`**, enviar DMEL:

```bash
sbatch --job-name=lab2_test_dmel \
  --export=ALL,DATA=dataset/test.h5,RUN=runs/dmel-seed42,CACHE=cache/test-closure-20261008.h5 slurm/predict.sbatch
```

Los jobs de predicción existentes escriben en `RUN/test/`; comprobar antes que no haya una entrega anterior que se quiera conservar. Los cuatro shards solicitados no equivalen por sí mismos a una cantidad de VRAM. No se presupone disponibilidad inmediata de recursos.

## Verificación y evidencia para la PPT

El wrapper de workstation genera automáticamente:

- `runs/closure-seed42-20261008/mrs-informer/`: checkpoints, historial, métricas y predicciones de validación.
- `runs/closure-seed42-20261008/test-lstm/` y `test-dmel/`: CSV, NPZ y metadatos de test.
- `outputs/verification/closure-seed42-20261008/ablation.json`: comparación emparejada, parámetros y métricas recalculadas.
- `outputs/verification/closure-seed42-20261008/test-*.json`: verifica cardinalidad, 48 horizontes, orden de IDs y cuencas, finitud, igualdad CSV/NPZ y hashes. El hash del checkpoint registra el archivo presente durante la verificación; no reemplaza el log de generación.

En Khipu ejecutar al terminar:

```bash
python -m scripts.lab2_closure compare-ablation \
  --reference runs/dmel-seed42 --ablation runs/mrs-informer-closure-seed42-20261008 \
  --out outputs/verification/cierre-ablation.json
python -m scripts.lab2_closure verify-test --data dataset/test.h5 \
  --run runs/dmel-seed42 --predictions runs/dmel-seed42/test \
  --out outputs/verification/cierre-test-dmel.json
python -m scripts.lab2_closure verify-test --data dataset/test.h5 \
  --run runs/lstm-seed42-fd-fix --predictions runs/lstm-seed42-fd-fix/test \
  --out outputs/verification/cierre-test-lstm.json
```

Después, incorporar la tabla de ablación, curvas y exportación verificada en informe/PPT. Una semilla no permite afirmar significancia. Los resultados son de validación usada para seleccionar checkpoints; no son métricas de test. La persistencia adicional y múltiples semillas serían extensiones, no sustituyen estos dos requisitos.
