# Paso a paso: preparar y ejecutar DMEL en Khipu

Usar dos terminales: **A en tu Mac**, **B conectada a Khipu**. El destino es `/home/max.ramirez/dmel`, separado de Sonidos. Ejecutar cada paso cuando el anterior termine sin errores. Si el servidor rechaza recursos o paquetes, guardar el error antes de cambiar versiones/cuotas.

**Estado al 03/10/2026:** precompute, LSTM 53987 y DMEL 54181 completados. Esta guía conserva el procedimiento inicial; para un traslado seguir [migración a Linux](migracion-workstation.md). Ver [evidencia y límites](estado-khipu.md). No repetir el precompute ni los entrenamientos terminados. Las indicaciones de archivos faltantes en los pasos siguientes corresponden a la preparación inicial.

## 1. Mac: comprobar conexión

```bash
ssh max.ramirez@khipu.utec.edu.pe
```

Autenticarse normalmente. Para volver al Mac, `exit`. No hace falta compartir contraseñas en el chat. El script `--check` usa autenticación por clave y no pide contraseña: para esta primera conexión usar `ssh` directamente.

## 2. Mac: subir código y datos disponibles

```bash
cd /Users/maxhoustonramirezmartel/code/utec/5/deep/dmel
bash scripts/sync_khipu.sh --code
ssh max.ramirez@khipu.utec.edu.pe 'mkdir -p /home/max.ramirez/dmel/dataset /home/max.ramirez/dmel/logs /home/max.ramirez/dmel/cache /home/max.ramirez/dmel/runs'
rsync -avP --partial dataset/test.h5 dataset/metadata.json dataset/leer_datos.py max.ramirez@khipu.utec.edu.pe:/home/max.ramirez/dmel/dataset/
```

Puede solicitar contraseña varias veces. No usar `--data` todavía: ese modo exige que train también exista localmente. Si `--code` encuentra un directorio remoto ocupado sin marcador DMEL, se detiene para protegerlo; revisar su contenido antes de continuar.

## 3. Khipu: comprobar recursos y crear entorno

En terminal B:

```bash
ssh max.ramirez@khipu.utec.edu.pe
cd /home/max.ramirez/dmel
df -h .
myaccount
sinfo -s
sinfo -p gpu -N -o '%N %G'
scontrol show partition gpu
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-khipu.txt
```

Los scripts usan los recursos observados en Sonidos: `standard`, `gpu`, `shard:rtxa6000`. Confirmar que sigan disponibles. Los datos ocupan alrededor de 5.5 GB; además hay que reservar espacio/cuota para entorno, caches y checkpoints. Si no existe Python3.12 o falla la instalación, enviar la salida exacta para ajustar el entorno.

## 4. Khipu: descargar train directamente

```bash
python -m pip install gdown
gdown 1Ou6LEqX0eflNhf0AA_qp-lXd0SPlm6GG -O dataset/train.h5 --continue
wc -c < dataset/train.h5
```

El tamaño esperado es **5076559230 bytes**. `--continue` permite retomar una descarga interrumpida; [documentación de gdown](https://github.com/wkentaro/gdown). La descarga y el esquema ya fueron confirmados en Khipu por el usuario: no repetir este paso si el archivo validado está presente.

Si gdown falla por permisos, cuota de Drive o acceso de red, descargar [train.h5 desde el navegador](https://drive.google.com/file/d/1Ou6LEqX0eflNhf0AA_qp-lXd0SPlm6GG/view?usp=drivesdk) al Mac y transferirlo desde terminal A:

```bash
rsync -avP --partial ~/Downloads/train.h5 max.ramirez@khipu.utec.edu.pe:/home/max.ramirez/dmel/dataset/train.h5
```

Ajustar la ruta si se guardó con otro nombre. El `wc` confirma tamaño, no integridad científica; el paso siguiente verifica HDF5 y calcula checksum.

## 5. Khipu: validar datos y pruebas en CPU

Desde `/home/max.ramirez/dmel`, con `.venv` activado:

```bash
sbatch --job-name=dmel_check --partition=standard --cpus-per-task=2 --mem=8G --time=00:20:00 --output=logs/%x_%j.out --wrap='set -e; .venv/bin/python -m dmel.cli inspect --data dataset/train.h5; .venv/bin/python -m pytest tests -q'
```

Guardar el número de job que devuelve `Submitted batch job ...`. Sustituir **JOBID** por ese número:

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS
cat logs/dmel_check_JOBID.out
```

Continuar si termina `COMPLETED`, `ExitCode=0:0`, HDF5 tiene el esquema esperado y pasan las pruebas. El código inspecciona shape/flags y rechaza valores no finitos cuando se leen para entrenar; esto no sustituye una auditoría estadística completa de train.

## 6. Khipu: piloto con arquitectura final

Primero descomponer las filas seleccionadas del piloto (256 train + 64 validation), usando los parámetros CEEMDAN finales:

```bash
sbatch --job-name=dmel_pilot_pre --time=01:00:00 --export=ALL,LIMIT=0,SELECTION_CONFIG=configs/pilot-full.json,CONFIG=configs/decomposition.json,OUT=cache/pilot-full.h5 slurm/precompute.sbatch
```

Cuando ese job termine `COMPLETED`:

```bash
sbatch --job-name=dmel_pilot_gpu --time=00:30:00 --gres=shard:rtxa6000:8 --export=ALL,CONFIG=configs/pilot-full.json,CACHE=cache/pilot-full.h5,OUT=runs/pilot-full slurm/train.sbatch
```

Cuando termine, revisar:

```bash
cat runs/pilot-full/history.json
cat runs/pilot-full/environment.json
cat runs/pilot-full/validation/metrics.json
```

Guardar también `logs/dmel_pilot_gpu_JOBID.out` y el estado `sacct`. Este piloto comprueba CUDA, batch 32, memoria pico y pipeline. Su métrica usa pocos datos y no es un resultado final. Si falla, corregir la causa antes de ejecutar el dataset completo.

## 7. Khipu: precompute completo y tres experimentos

Después de verificar el piloto y ajustar tiempo/recursos si hiciera falta:

```bash
sbatch --job-name=dmel_train_pre --time=08:00:00 --cpus-per-task=16 --mem=16G --export=ALL,LIMIT=0,SELECTION_CONFIG=,CONFIG=configs/decomposition.json,OUT=cache/train.h5,MAX_SECONDS=25200,AUTO_REQUEUE=1,MAX_RESTARTS=4 slurm/precompute.sbatch
```

El precompute ya terminó en el job 53249. El comando anterior queda como referencia para una ejecución desde cero; no volver a enviarlo sobre la cache final existente.

El precompute cierra su cache parcial después de 7 horas más el drenaje de las ventanas en curso y se reencola hasta 4 veces (5 ciclos máximos). Solo reencola pausas limpias, no fallos. Los logs se anexan al mismo archivo. Al reanudarse, valida fuente/configuración e ignora filas ya procesadas. El resultado final `cache/train.h5` aparece únicamente cuando todas las filas están completas. Mantener código, configuración y datos sin cambios durante los ciclos. [Procedimiento oficial de ciclos](https://docs.khipu.utec.edu.pe/tutoriales/reinicio-job-pytorch/).

Revisar `squeue --me` y el log; se puede pasar de `R` a `PD` entre ciclos. Esperar `COMPLETED` y la cache final antes de entrenar. Luego:

```bash
sbatch --job-name=dmel_persistence --partition=standard --cpus-per-task=2 --mem=4G --time=00:30:00 --output=logs/%x_%j.out --wrap='.venv/bin/python -m dmel.cli persistence --data dataset/train.h5 --out runs/persistence/validation'
bash slurm/submit_experiments.sh --dry-run
bash slurm/submit_experiments.sh --submit
```

Encola LSTM, DMEL y ablación MRS-Informer en serie, con dependencias `afterok`. Por defecto solicita 8 shards/32GB/8h, con una época completa por ciclo y hasta 99 reinicios, respetando el máximo de 100 épocas y early stopping de cada configuración. Un fallo real detiene la cadena. No lanzar la cadena completa si alguno de esos modelos ya se inició por separado; para arrancar solo LSTM usar `bash slurm/submit_experiments.sh --model lstm --submit`. [Guía de entrenamiento y sincronización](entrenamiento-khipu.md). Jobs registrados en `logs/submissions.tsv`. Tras enviar con sbatch se puede cerrar SSH: [Slurm ejecuta los jobs en el cluster](https://slurm.schedmd.com/sbatch.html).

No cambiar código/configuración de una corrida iniciada. Cada época debe caber en las 8 horas: medir la primera y revisar rendimiento si se excede. Ante un TIMEOUT se conserva la última época completamente guardada, si existe; usar `RESUME=1` solo después de revisar la causa y las dependencias. El reinicio entre épocas usa automáticamente el mismo checkpoint, datos, configuración y código.

## 8. Khipu: test del modelo elegido con validación

```bash
sbatch --job-name=dmel_test_pre --export=ALL,DATA=dataset/test.h5,SPLIT=test,LIMIT=0,CONFIG=configs/decomposition.json,OUT=cache/test.h5 slurm/precompute.sbatch
```

Al terminar, si la corrida elegida es DMEL:

```bash
sbatch --job-name=dmel_test --export=ALL,RUN=runs/dmel-seed42,CACHE=cache/test.h5 slurm/predict.sbatch
```

Salida: `runs/dmel-seed42/test/predictions.csv`, con Id y 48 caudales para 27,983 muestras. Si se elige LSTM, usar `RUN=runs/lstm-seed42` y omitir CACHE; si MRS-Informer, usar su carpeta y la misma cache test. No usar `test_targets.csv` para elegir el modelo.

## 9. Mac: recuperar resultados

Desde terminal A:

```bash
cd /Users/maxhoustonramirezmartel/code/utec/5/deep/dmel
bash scripts/sync_khipu.sh --pull
```

Trae best.pt, métricas, predicciones y logs. No hace falta descargar caches ni train al Mac. Para la primera revisión, compartir salida de `sinfo`, log del job de comprobación y log/historial del piloto GPU; así podemos dimensionar la ejecución completa con datos observados.
