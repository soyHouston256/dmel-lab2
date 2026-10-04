# Entrenamiento GPU después del precompute

**Registro histórico del procedimiento. Estado al 03/10/2026:** LSTM 53987 y DMEL 54181 terminaron con `COMPLETED 0:0`, 20 épocas y parada temprana. Los comandos siguientes documentan cómo se lanzaron los experimentos; no es necesario repetirlos para la migración. Ver [resultados vigentes](../outputs/README.md) y [workstation Linux](migracion-workstation.md). La reanudación por ciclos GPU quedó confirmada en los logs. Las menciones a comprobaciones pendientes describen la situación en la fecha de cada paso.

**Actualización 30/09:** el job 53814 falló durante la primera validación por retención de almacenamiento compartido. Para ese intento seguir [la recuperación específica](recuperacion-lstm-53814.md), que conserva la carpeta fallida y utiliza la corrección local de `infer()`.

El usuario confirmó `cache/train.h5` completo, 272142 ventanas, en el job 53249. No se vuelve a calcular. La actualización del 29/09 añade ciclos por época al entrenamiento; debe subirse antes de iniciar una corrida nueva.

## 1. Mac: subir el código actualizado

En una terminal local, fuera de SSH:

```bash
cd /Users/maxhoustonramirezmartel/code/utec/5/deep/dmel
bash scripts/sync_khipu.sh --code
```

El modo `--code` transfiere código/configuración/documentación; la cache completa sigue en Khipu. La implementación CEEMDAN conserva su hash `852b59039123cb3f59370f5c1183e78ce4826315d3c4ee81342ec42304605f86`.

## 2. Khipu: iniciar LSTM

```bash
cd /home/max.ramirez/dmel
bash slurm/submit_experiments.sh --model lstm --dry-run
bash slurm/submit_experiments.sh --model lstm --submit
squeue --me
```

La primera línea imprime el comando sin enviarlo. La segunda comprueba que Python y el batch tienen la opción nueva y solicita el job. Esta comprobación usa `python -S -m dmel.cli train --help`, sin PyTorch ni otras dependencias científicas. Si falla una comprobación, corregir la sincronización antes de reenviar. El helper antiguo rechaza `--model` como primera opción, por lo que no puede enviar silenciosamente una cadena antigua con esta invocación.

Durante el envío se muestran, en orden, `Comprobando opciones de entrenamiento (sin cargar PyTorch)...`, `Enviando lstm a Slurm...` y `lstm: job <ID>`. El entrenamiento empieza dentro del job GPU, después de obtener recursos. Un `KeyboardInterrupt` en imports de PyTorch durante la comprobación `--help` indica que se interrumpió la versión anterior de la CLI; subir la corrección de importación diferida antes de repetir el envío. En ese punto de la comprobación todavía no se ha llamado a sbatch.

Se solicitan 8 CPU, 32 GB, 8 shards RTX A6000 y hasta 8 horas por ciclo. Se entrena una época completa, valida y guarda `last.pt` (pesos, Adam, época y RNG), además de `best.pt` si mejora. Si faltan épocas, se reencola el mismo job hasta 99 veces; se conserva el máximo de 100 épocas y paciencia 10. Cada época recorre todo el train y toda la validación definidos por los flags oficiales. `EPOCHS_PER_CYCLE` es un límite operativo, no reduce el dataset ni cambia los hiperparámetros.

La primera época medirá el costo real con todos los datos. Debe caber en 8 h; el ciclo por época no garantiza eso hasta medirlo. Si ocurre TIMEOUT antes del primer checkpoint, detener y revisar rendimiento; no enviar un bucle de reintentos. Si hay un checkpoint completo, se puede reanudar bajo la misma configuración después de revisar la causa. No modificar código, datos ni configuración durante una corrida iniciada.

## 3. Consultar el avance

Sustituir JOBID por el identificador devuelto:

```bash
squeue --me
tail -n 20 logs/lab2_lstm_JOBID.out
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
```

La cabecera debe mostrar `epochs_per_cycle=1`, `restart=0`, `resume=0`. Primero verifica/checksumea datos y ajusta la normalización con train; puede tardar antes del mensaje `training_start`. Este mensaje indica las cantidades reales de train/validation. Cada 500 batches se registra `train_progress`. Al cerrar la primera época se imprime `epoch: 0`, MSE train, RMSE validation, segundos y memoria GPU máxima. Tras el reinicio se espera `restart=1`, `resume=1`, `start_epoch=1`.

Después de la primera época existe:

```bash
cat runs/lstm-seed42/status.json
cat runs/lstm-seed42/history.json
```

`complete: false` corresponde a entrenamiento parcial; `complete: true`, validación exportada y `COMPLETED 0:0` identifican una corrida terminada. La solicitud de requeue deja al job esperando recursos entre épocas. Este procedimiento sigue el esquema de [ciclos de Khipu](https://docs.khipu.utec.edu.pe/tutoriales/reinicio-job-pytorch/); su duración total depende de las épocas necesarias y la cola. La reanudación GPU de esta versión aún debe observarse remotamente; las pruebas locales verifican equivalencia exacta en CPU.

## 4. Referencia de persistencia y modelos siguientes

La referencia de persistencia repite el último caudal observado para los 48 horizontes. Puede ejecutarse con un job CPU:

```bash
sbatch --job-name=dmel_persistence --partition=standard --cpus-per-task=2 --mem=4G --time=00:30:00 --output=logs/%x_%j.out --wrap='.venv/bin/python -m dmel.cli persistence --data dataset/train.h5 --out runs/persistence/validation'
```

Después de revisar el entrenamiento LSTM, iniciar DMEL y luego la ablación, cada uno por separado:

```bash
bash slurm/submit_experiments.sh --model dmel --submit
# Después de completar/revisar DMEL:
bash slurm/submit_experiments.sh --model mrs_informer --submit
```

LSTM usa señal original; DMEL y la ablación usan la misma `cache/train.h5`. Comparar `runs/*/validation/metrics.json` y métricas por horizonte/cuenca. El CSV de test se genera después de seleccionar mediante validación; `test_targets.csv` permanece fuera del ajuste y selección.
