# DMEL en Khipu

**Estado al 03/10/2026:** caché train, LSTM 53987 y DMEL 54181 completados. Esta guía describe la operación en el clúster; para usar el respaldo existente en otro equipo seguir [migración a Linux](../docs/migracion-workstation.md). Los envíos nuevos deben usar una carpeta de corrida nueva. El helper `submit_experiments.sh` conserva nombres fijos como `runs/dmel-seed42`: no volver a enviarlo sobre una corrida existente.

Las corridas finales recibidas usaron 8 CPU, 32 GB de RAM y 4 shards. El helper solicita 8 shards por defecto; `TRAIN_GRES=shard:rtxa6000:4` lo cambia. La disponibilidad depende de la cuota y del estado del clúster. Las estimaciones y pilotos siguientes son referencias del procedimiento.

El usuario eligió Khipu y ejecuta los comandos desde su sesión SSH, en `~/dmel`, separado de Sonidos. Sus logs confirman datos y piloto CPU/GPU correctos. `myaccount` confirma `postgrado`, 32 CPU y 8 horas por job; el detalle de recursos GPU/RAM de esa salida está truncado. [Estado y evidencia](../docs/estado-khipu.md).

`standard`, `gpu` y `shard:rtxa6000:N` funcionaron en Sonidos; confirmar disponibilidad y cuotas actuales. Un shard no garantiza VRAM aislada o fija, según [Slurm](https://slurm.schedmd.com/gres.html#Sharding). Se conserva el uso de `sbatch` del flujo anterior.

## Transferencia y entorno

Desde la laptop, con `train.h5` disponible y Python con dependencias:

```bash
python -m dmel.cli inspect --data dataset/train.h5
python -m dmel.cli inspect --data dataset/test.h5
bash scripts/sync_khipu.sh --check
bash scripts/sync_khipu.sh --code --dry-run
bash scripts/sync_khipu.sh --code
bash scripts/sync_khipu.sh --data
```

Autenticar la cuenta por el procedimiento normal en la terminal, sin guardar contraseñas en archivos. `KHIPU_REMOTE`/`KHIPU_DIR` cambian destino. `--code` usa carpetas explícitas; `--data` exige train/test y excluye `test_targets.csv`. No hay `rsync --delete`; se exige destino vacío o marcador `.dmel-project`. Comparar SHA-256 remotos con los de `inspect` local.

En el login remoto, dentro de `~/dmel`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-khipu.txt
source .venv/bin/activate
mkdir -p logs cache runs
python -m dmel.cli environment
sinfo -s
sinfo -p gpu -N -o '%N %G'
scontrol show partition gpu
```

Entorno independiente de Sonidos, conservando `torch==2.7.1+cu126`; no requiere librerías de audio. El piloto 53146 confirmó CUDA 12.6, RTX A6000 y una operación GPU real.

## Piloto

Preparar exactamente las 16 filas train y 8 validation del config smoke. Un límite aleatorio global no asegura cubrir ambos subconjuntos.

```bash
sbatch --export=ALL,LIMIT=0,SELECTION_CONFIG=configs/smoke.json,CONFIG=configs/decomposition-smoke.json,OUT=cache/smoke.h5 slurm/precompute.sbatch
# Una vez completado:
sbatch --export=ALL,CACHE=cache/smoke.h5,CONFIG=configs/smoke.json,OUT=runs/smoke-gpu slurm/train.sbatch
# Medir descomposición final en 32 ventanas:
sbatch --export=ALL,LIMIT=32,CONFIG=configs/decomposition.json,OUT=cache/cost-pilot.h5 slurm/precompute.sbatch
```

Revisar pérdida, predicciones, memoria y tiempos. El precompute solicita 16 CPU/16GB/6h; son límites iniciales. Cada worker usa un hilo BLAS. Para medir GPU final, crear otra configuración con dimensiones de `dmel.json` y límites de muestras/épocas; el smoke pequeño no certifica la memoria del modelo final. No copiar datos a `/tmp` remoto sin verificar capacidad y si usa RAM.

## Corridas completas

Tras dimensionar el piloto:

```bash
sbatch --job-name=dmel_train_pre --time=08:00:00 --cpus-per-task=16 --mem=16G --export=ALL,LIMIT=0,SELECTION_CONFIG=,CONFIG=configs/decomposition.json,OUT=cache/train.h5,MAX_SECONDS=25200,AUTO_REQUEUE=1,MAX_RESTARTS=4 slurm/precompute.sbatch
```

Cada ciclo deja de enviar ventanas después de 7 horas, drena las pendientes y cierra HDF5. La CLI devuelve 75 si quedan filas; el batch solicita `scontrol requeue` solo en ese caso. Se permiten 4 reinicios (5 ciclos en total), conservando el mismo job y anexando los logs. Los demás errores detienen el job para inspección. Khipu documenta este uso en su [guía de reinicio](https://docs.khipu.utec.edu.pe/tutoriales/reinicio-job-pytorch/).

Repetir el mismo comando puede reanudar un `.partial` compatible; una cache terminada no se sobrescribe. El presupuesto es suave: no protege de un SIGKILL/fallo de nodo; la hora de margen permite drenar las tareas normales antes del límite de Slurm. Si se reduce `--time`, también hay que ajustar `MAX_SECONDS` dejando margen; cuando está disponible, `SLURM_JOB_END_TIME` limita el presupuesto para reservar al menos 10 minutos de cierre. No cambiar código/configuración durante los ciclos ni lanzar dos jobs sobre la misma cache.

La cache de train ya está terminada: job 53249 `COMPLETED 0:0`, 272142 ventanas. No repetir el precompute. Subir la actualización del entrenamiento antes de lanzar los modelos: [guía GPU](../docs/entrenamiento-khipu.md). Para una ejecución nueva de todos los experimentos:

```bash
sbatch --partition=standard --cpus-per-task=2 --mem=4G --time=00:30:00 --output=logs/persistence_%j.out --wrap='.venv/bin/python -m dmel.cli persistence --data dataset/train.h5 --out runs/persistence/validation'
bash slurm/submit_experiments.sh --dry-run
bash slurm/submit_experiments.sh --submit
```

La cadena LSTM→DMEL→MRS-Informer usa `afterok`; por defecto 8 shards/8h y `EPOCHS_PER_CYCLE=1`, `AUTO_REQUEUE=1`, `MAX_RESTARTS=99`. Cada época guarda modelo, optimizador y estados RNG. La pausa devuelve 75, y el batch reencola el mismo job; solo devuelve éxito cuando termina el presupuesto científico o el early stopping y exporta validación. Las dependencias esperan ese éxito final. Los errores reales no se reencolan deliberadamente.

Para empezar con un solo modelo: `bash slurm/submit_experiments.sh --model lstm --submit`; luego usar `--model dmel` o `--model mrs_informer`. No volver a enviar toda la cadena si ya se inició un modelo. El helper comprueba que batch y CLI conocen `--epochs-per-cycle` antes de enviar. `TRAIN_TIME`, `TRAIN_GRES`, `TRAIN_EPOCHS_PER_CYCLE`, `TRAIN_MAX_RESTARTS` y `CACHE` son ajustables dentro de la cuota; la opción por ciclo no cambia las 100 épocas totales. Ids en `logs/submissions.tsv`.

La primera época completa todavía debe medirse en Khipu: cada época tiene que caber en 8 horas. El guardado es al terminar la época, no a mitad de un batch. Un TIMEOUT requiere revisar rendimiento y `last.pt`; si existe, reanudar con `RESUME=1` y reparar las dependencias afectadas. No entrenar ni preprocesar el dataset completo en el nodo login.

## Reanudar, predecir y recuperar

```bash
squeue -u "$USER" -o '%.10i %.28j %.9P %.8T %.10M %R'
sacct -j JOBID --format=JobID,JobName,State,Elapsed,MaxRSS,ReqTRES%40
# Config/código/datos deben ser idénticos al checkpoint:
sbatch --job-name=lab2_dmel --time=08:00:00 --gres=shard:rtxa6000:8 --export=ALL,CONFIG=configs/dmel.json,CACHE=cache/train.h5,OUT=runs/dmel-seed42,RESUME=1,EPOCHS_PER_CYCLE=1,AUTO_REQUEUE=1,MAX_RESTARTS=99 slurm/train.sbatch
# Precompute test final:
sbatch --export=ALL,DATA=dataset/test.h5,SPLIT=test,LIMIT=0,CONFIG=configs/decomposition.json,OUT=cache/test.h5 slurm/precompute.sbatch
# Tras completarlo y disponer de best.pt:
sbatch --export=ALL,RUN=runs/dmel-seed42,CACHE=cache/test.h5 slurm/predict.sbatch
```

Las solicitudes de reanudación también pueden ajustarse con `--time`/`--gres`; por defecto `train.sbatch` pide 2 shards/1h para el piloto. Test conserva todas las 27,983 filas. Usar el checkpoint elegido solo con validación. Desde laptop:

```bash
bash scripts/sync_khipu.sh --pull --dry-run
bash scripts/sync_khipu.sh --pull
```

Ese modo recupera `best.pt`, JSON/CSV/NPZ y logs; excluye `last.pt` y cachés. Para un respaldo completo usar `bash scripts/descargar_khipu.sh`, ejecutado por el usuario desde la laptop. La opción `--verify-only` comprueba la copia local sin SSH. Ver [utilidades de transferencia](../scripts/README.md). Comprobar estado final, exit status y checksums; un job encolado no es un resultado completado.
