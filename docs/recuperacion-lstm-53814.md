# Recuperación tras el fallo de validación de LSTM

**Cierre al 03/10:** estos pasos dieron lugar al job **53987**, que completó 20 épocas con reanudación GPU y `COMPLETED 0:0`. La mejor validación alcanzó RMSE 0.1022830620 mm/h. El respaldo local incluye sus checkpoints y predicciones. Los pasos y las comprobaciones pendientes que siguen son el registro histórico de la recuperación, no una indicación de enviar otra copia. [Estado actual](estado-khipu.md).

El usuario compartió `FAILED 1:0` para el job 53814, con 00:11:13 de duración. La ejecución empezó el 30/09/2026 a las 18:27:04 -05 en g002 con una RTX A6000 y 4 shards. El smoke CUDA pasó. El traceback sitúa el fallo en la primera validación, después del bucle de entrenamiento de la época 0.

## Causa y corrección

`infer()` acumulaba `batch["Id"].numpy()`, `batch["basin_id"].numpy()` y `batch["y"].numpy()` hasta concatenarlos al final. Esas vistas mantenían vivo el almacenamiento recibido de los workers. La estrategia `file_descriptor` de PyTorch en Linux conserva descriptores para el almacenamiento compartido. La retención crece con los lotes y es consistente con el error de Khipu. El límite remoto exacto de descriptores no fue proporcionado.

La corrección añade `.copy()` a los tres arrays acumulados. Sus valores, orden, tipos y métricas se mantienen. Las predicciones ya son arrays independientes porque se calculan por operaciones aritméticas o `repeat`. No se cambia la arquitectura, los hiperparámetros, la cantidad de workers ni la estrategia global de PyTorch.

Fuentes: [PyTorch 2.7, memoria compartida y descriptores](https://docs.pytorch.org/docs/2.7/multiprocessing.html#file-descriptor-file-descriptor), código local `dmel/engine.py` y log proporcionado por el usuario.

Verificación local: las cuatro variantes de la regresión de propiedad del almacenamiento fallaron con el código anterior y pasaron con la corrección. Cubren modelo/persistencia y datos con/sin etiquetas. Se añadió una integración sintética con 18142 filas, 567 lotes y 2 workers `spawn`. En macOS el sandbox bloqueó inicialmente `torch_shm_manager`; fuera del sandbox autorizado, sus comparaciones pasaron y Pytest informó 5 pruebas aprobadas, pero el proceso quedó detenido al salir y se cerró con SIGINT (salida 130). Por ello ese ensayo no se considera una ejecución completa y limpia del comando, aunque conserva [el reporte de sus aserciones](../outputs/tests-inference-workers.xml). La integración se reserva ahora a plataformas que admiten `file_descriptor`, como Linux/Khipu, y queda pendiente allí. El Mac usa `file_system`; las 4 regresiones portables y las 81 pruebas previas forman la [suite local](../outputs/tests-inference-resources.xml).

El checkpoint se escribe después de terminar la validación. Este intento tenía `restart=0`, `resume=0` y `start_epoch=0`, por lo que el flujo mostrado no llegó al primer guardado. No se ha inspeccionado por SSH la carpeta remota. Se propone una carpeta nueva para conservar íntegra la corrida fallida y registrar los hashes del código corregido. No usar `--resume` sobre ella ni editar a mano sus hashes para forzar compatibilidad.

## 1. Subir desde el Mac

La suite local final terminó correctamente con **85 pruebas aprobadas y 1 omitida en 20.44 s**. La omitida requiere la estrategia de memoria compartida de Linux.

En una terminal local, fuera de SSH:

```bash
cd /Users/maxhoustonramirezmartel/code/utec/5/deep/dmel
bash scripts/sync_khipu.sh --code
```

## 2. Comprobar y enviar desde Khipu

```bash
cd /home/max.ramirez/dmel
sha256sum dmel/engine.py
```

Hash de la versión corregida: `8afa1d67bd584b7b5399f3c9cf9d1790df596173f1bb6536c8f2a6130271b290`.

Una vez confirmada la subida, enviar **una sola vez**:

```bash
sbatch --job-name=lab2_lstm_fix \
  --partition=gpu --gres=shard:rtxa6000:4 \
  --cpus-per-task=8 --mem=32G --time=08:00:00 \
  --export=ALL,CONFIG=configs/lstm.json,DATA=dataset/train.h5,OUT=runs/lstm-seed42-fd-fix,CACHE=,RESUME=0,AUTO_REQUEUE=1,EPOCHS_PER_CYCLE=1,MAX_RESTARTS=99 \
  slurm/train.sbatch
```

Conservar el nuevo identificador devuelto. Este envío directo no pasa por `submit_experiments.sh`, por lo que no añade automáticamente una fila a `logs/submissions.tsv`. Si `runs/lstm-seed42-fd-fix` ya contiene un intento, revisarlo antes de enviar: el programa rechaza sobrescribirlo.

El archivo `cache/train.h5` sigue completo y sirve para DMEL y la ablación. Su descomposición no cambia: `decomposition.py` conserva SHA-256 `852b59039123cb3f59370f5c1183e78ce4826315d3c4ee81342ec42304605f86`. LSTM utiliza los datos originales y no necesita esa cache.

## 3. Confirmar la primera época

Sustituir `NUEVO_ID` por el identificador real:

```bash
squeue -j NUEVO_ID
tail -n 45 logs/lab2_lstm_fix_NUEVO_ID.out
cat runs/lstm-seed42-fd-fix/status.json
```

El primer objetivo es ver `epoch: 0` con su RMSE, los checkpoints `best.pt` y `last.pt`, y `epochs_completed: 1`. `complete: false` con `reason: epoch_cycle_limit` indica que la época se guardó y falta continuar. Tras el reencolado se espera `restart=1`, `resume=1`, `start_epoch=1`. La validación completa y el reencolado GPU aún requieren confirmación remota.
