# Estado de Khipu — actualizado el 03/10/2026

Evidencia remota: salidas copiadas por el usuario en la conversación. El agente no inspeccionó directamente esos archivos mediante SSH. Destino: `/home/max.ramirez/dmel`; Python 3.12 en `.venv`.

**Respaldo local incorporado el 03/10:** la descarga ejecutada por el usuario ya está en `migration/khipu/`. Se verificaron localmente los SHA-256 de sus 68 archivos inventariados, incluyendo datos, caché y ambas corridas completas; no se abrió una conexión SSH. Las métricas de esa copia se cotejaron con los JSON de resultados. Ver [registro de verificación](../outputs/verification/khipu-backup-20261003.json) y [guía de traslado](migracion-workstation.md). Las secciones fechadas que siguen conservan las limitaciones conocidas en cada momento.

| Job | Evidencia compartida |
|---|---|
| 53142 | HDF5 supervisado de 272142 filas; 46 pruebas aprobadas en 11.99 s. |
| 53145 | Piloto CEEMDAN completo, 320 ventanas, `COMPLETED 0:0`, 2m42s. Bucle de descomposición: 100.94 s con 16 CPU. |
| 53146 | Piloto GPU, RTX A6000/CUDA 12.6, `COMPLETED 0:0`, 4 min. Una época, 256 train/64 validation; RMSE 0.2728075841 mm/h, NSE -0.1791823650; diagnóstico, no resultado final. |
| 53150 | Solicitud de 36 h bloqueada por `QOSMaxWallDurationPerJobLimit`; sustituida en el flujo posterior. |
| 53248 | Batch anterior sin ciclos, n003. Interrumpido deliberadamente con SIGINT: `FAILED 0:2`, batch `CANCELLED 0:2`, 13m40s. |
| 53249 | **COMPLETED 0:0**, cache completa de 272142 ventanas. Último ciclo: 16277 ventanas/4859.775 s; `sacct` muestra 01:22:17 para la ejecución final. Inicio original 27/09/2026 08:23:05 -05, 16 CPU. |
| 53814 | **FAILED 1:0**, 00:11:13. Inició el 30/09 a las 18:27:04 -05 en g002, RTX A6000 con 4 shards. Smoke CUDA correcto; fallo `Too many open files` durante la validación de la época 0. Sustituido por la corrida corregida 53987. |
| 53987 | **COMPLETED 0:0**, confirmado el 03/10. `complete=true`, 20 épocas, `early_stopping`; validación final RMSE 0.10228306200404642 mm/h, NSE global 0.6446957192955849. Mejor época: `epoch=9`. `Elapsed=00:14:01` corresponde al último registro de ejecución, no al total de ciclos. |
| 54181 | **COMPLETED 0:0**, DMEL, 20 épocas y `early_stopping`. Mejor `epoch=9`. Validación RMSE 0.1056661151 mm/h, NSE global 0.6208033608. `Elapsed=00:13:01` corresponde al último ciclo, no al entrenamiento completo. |

SHA-256 remoto de train:
`38a1e2dee5f9aa61f5e1543dc6f7c22689f6ee3059511498022df31f01730078`.

Cache final remota: `cache/train.h5`, `complete=true`, `done=272142`, `remaining=0`.
SHA-256: `e8d304ca15b9f33eb5eb01bba82c32caffa004bc94bef2c183b038c57d7987e1`.
[Registro estructurado del resultado](../outputs/khipu-precompute-complete.json). No volver a ejecutar el precompute de train.

`myaccount` devuelve cuenta/default `postgrado`, QoS `a-postgrado`, `cpu=32,gres/gpu=1,gres/shard=+` y `08:00:00`. El `+` indica que el listado de recursos está truncado: esa salida no confirma cuántos shards ni cuánta RAM tiene la cuenta. El piloto sí obtuvo 8 shards RTX A6000. Los límites publicados generales de posgrado son 32 CPU/98 GB/8 shards/8 h; prevalece la configuración efectiva de la cuenta. Fuente: [tipos de cuenta](https://docs.khipu.utec.edu.pe/info/tipos-de-cuentas/).

## Adaptación para ciclos

Con el rendimiento observado, el precompute completo podría tardar unas 20–24 h con 16 CPU, sin garantizar el rendimiento de otras ventanas/nodos. Se mantiene esa asignación validada y se solicitan ciclos de 8 h. `MAX_SECONDS=25200` detiene nuevos envíos a las 7 h, drena un máximo de 2×workers ventanas en vuelo y cierra la cache. Una pausa devuelve código 75 y conserva `.partial`; el batch solicita otro ciclo con `scontrol requeue`, hasta 4 reinicios. Un error real se propaga y no se reencola. El nombre final `.h5` se crea únicamente al completar todas las filas.

La reserva de una hora es margen operativo, no garantía frente a fallos del nodo/SIGKILL o tareas bloqueadas. No enviar jobs simultáneos sobre el mismo archivo. No modificar código/datos/configuración entre ciclos. La guía de Khipu permite [continuar trabajos en varios ciclos](https://docs.khipu.utec.edu.pe/tutoriales/reinicio-job-pytorch/).

Verificación local del precompute: **58 pruebas aprobadas** en 15.40 s, [JUnit](../outputs/tests-khipu-cycles.xml). Incluye equivalencia exacta pausa/reanudación vs ejecución continua, drenaje de un pool real, contrato CLI 75/0 y batch con scheduler simulado (éxito, pausa, error, límite de reinicios y rechazo de requeue). Los logs posteriores muestran distintos `pending_at_start` bajo el mismo job y finalización completa, consistentes con los ciclos configurados; no se recibió la lista íntegra de sus cabeceras/reinicios.

## Incidencia de sincronización resuelta y avance confirmado

El job 53248 ejecutaba el batch anterior aunque recibió las variables nuevas mediante `--export`. No tenía pausa por presupuesto ni reencolado automático. El `--time=08:00:00` de sbatch prevalecía sobre las 6 h impresas en el archivo; ese encabezado no era la causa del problema.

El usuario confirmó la terminación de 53248, las opciones nuevas en el batch remoto y `--max-seconds` en la ayuda de Python. El job 53249 aceptó la cache parcial y empezó con 269681 ventanas pendientes de 272142: **2461 ventanas recuperadas**. Un ciclo posterior comenzó con 185846 pendientes; el último comenzó con 16277 y las completó todas. `done_now` cuenta solo el ciclo actual; el resultado final indica 272142 ventanas completas.

Una prueba local adicional con SIGINT a un grupo de procesos real cerró una cache parcial legible con 18 filas confirmadas; se verificó reconstrucción de las entradas. La recuperación de las 2461 ventanas remotas se confirmó después con el log de 53249; no se realizó una auditoría independiente de cada componente remoto.

## Siguiente fase: entrenamiento GPU por épocas

La actualización local del 29/09 añade `--epochs-per-cycle` y reencolado a `train.sbatch`, manteniendo hiperparámetros/dataset. El helper selecciona un modelo con `--model`, comprueba la presencia de la opción en batch/CLI y usa 8 horas, 8 shards y una época por ciclo (hasta 99 reinicios para el máximo de 100 épocas). Guarda modelo, optimizador y RNG; respeta early stopping. Errores reales no se reencolan deliberadamente. Cada época aún debe caber en 8 h; falta medir la primera con todos los datos en GPU.

**70 pruebas locales aprobadas en 19.43 s**, [JUnit](../outputs/tests-training-cycles.xml). Se verificó equivalencia exacta de entrenamiento continuo vs tres ciclos con LSTM/DMEL y dropout; early stopping entre ciclos; CLI 75 para pausa y 0 al completar; batch simulado con reanudación automática/error/límite/rechazo de requeue. Sintaxis shell y dry-runs del helper comprobados. El hash de `decomposition.py` sigue coincidiendo con la cache remota: no requiere recalcularla.

## Corrección de la comprobación previa al envío

El usuario ejecutó `bash slurm/submit_experiments.sh --model lstm --submit` y pulsó Ctrl+C durante la carga de `torch.nn.modules.normalization`. El traceback muestra `cli.py -> data.py -> torch`: la comprobación de `train --help` importaba dependencias científicas antes de analizar los argumentos. Esa invocación se interrumpió antes de llamar a sbatch; no acredita un fallo de entrenamiento ni permite medir cuánto habría tardado el import completo en Khipu.

Corrección local: importar módulos científicos solo dentro del subcomando, después de `parse_args()`. La comprobación usa Python con `-S`, captura toda la ayuda sin cerrar el pipe prematuramente y muestra mensajes de preparación/envío. Se comprobó que todas las ayudas funcionan sin site-packages y que una CLI antigua impide enviar jobs durante una sincronización incompleta. **81 pruebas aprobadas en 18.29 s**, [JUnit](../outputs/tests-cli-startup.xml). La sintaxis del helper y su dry-run LSTM también pasaron. La cache conserva la misma firma CEEMDAN.

El usuario confirmó la comprobación ligera y el envío de LSTM como **53814**. El último estado compartido es `PD (QOSGrpGRES)`: Slurm informa que se alcanzó el límite agregado de recursos genéricos de la QoS. [Definición oficial](https://slurm.schedmd.com/job_reason_codes.html).

La salida posterior de `scontrol show job 53814` confirma `Account=postgrado`, `QOS=a-postgrado`, `ReqTRES=cpu=8,mem=32G,node=1,billing=8,gres/shard=8` y `TresPerNode=gres/shard:rtxa6000:8`. `sacctmgr -P show qos format=Name,GrpTRES` muestra para `a-postgrado`: `cpu=240,gres/shard=8,mem=180G`. Es un límite agregado compartido por todos los trabajos que usan esa QoS, distinto de los límites individuales: nuestra solicitud de 8 shards necesita todo su cupo de shards disponible. No se conoce qué trabajos lo ocupan ni su tiempo restante. [Semántica de GrpTRES de QoS](https://slurm.schedmd.com/sacctmgr.html#SECTION_SPECIFICATIONS-FOR-QOS).

El usuario aplicó `scontrol update JobId=53814 Gres=shard:rtxa6000:4`. La lectura posterior confirma `ReqTRES=cpu=8,mem=32G,node=1,billing=8,gres/shard=4` y `TresPerNode=gres/shard:rtxa6000:4`; el mismo job continúa `PENDING Reason=QOSGrpGRES`. No se ha medido el rendimiento del entrenamiento completo con esa asignación. El helper sigue solicitando 8 shards por defecto; para futuros envíos permite `TRAIN_GRES=shard:rtxa6000:4`.

El usuario ejecutó `squeue --qos=a-postgrado --states=R,CG -O JobID:10,UserName:18,TimeLeft:14,tres-alloc:100`. La salida muestra un trabajo activo, **53466**, con `cpu=4,mem=16G,node=1,billing=4,gres/shard=8` y `TIME_LEFT=6:09:32`. Ese trabajo ocupa todo el cupo agregado de 8 shards de `a-postgrado` en esa instantánea; reducir más la solicitud de 53814 no resolvería la falta de cupo mientras siga así. `TimeLeft` refleja el tiempo hasta el límite de esa ejecución, no una garantía del inicio de 53814: puede terminar antes y la asignación posterior depende del scheduler. Mantener 53814 en cola con 4 shards; todavía no hay inicio de entrenamiento confirmado.

Los párrafos anteriores recogen las instantáneas históricas de la cola. La salida del 30/09 confirma que 53814 finalmente obtuvo GPU y falló como se describe a continuación.

## Fallo durante la validación y corrección local

El traceback de 53814 termina en `infer()` al recibir lotes del DataLoader, con `RuntimeError: Too many open files`. El evento `training_start` registra 254000 muestras de entrenamiento y 18142 de validación, `start_epoch=0`, `resume=false`. La GPU y el smoke CUDA funcionaron. La última línea de progreso impresa registra 7500/7938 lotes de entrenamiento en 313.035 s; alcanzar la llamada a `infer()` implica que el bucle de entrenamiento terminó. El fallo ocurrió antes de guardar el primer checkpoint. No se inspeccionó directamente el directorio remoto ni se conoce el límite de descriptores del job.

Se encontró que los arrays NumPy acumulados para IDs, cuencas y etiquetas retenían el almacenamiento de los tensores recibidos de los workers. Se añadieron copias independientes en esas tres rutas. [PyTorch 2.7 documenta el uso de descriptores por almacenamiento compartido](https://docs.pytorch.org/docs/2.7/multiprocessing.html#file-descriptor-file-descriptor). No se cambian modelo, datos, hiperparámetros ni descomposición. El hash de `decomposition.py` sigue siendo el de la cache final.

Pruebas: 4 regresiones fallaron antes de la corrección y pasaron después. La integración de 18142 muestras sintéticas/567 lotes con dos workers fue bloqueada inicialmente por el sandbox. En una repetición autorizada sus aserciones pasaron (igualdad exacta con serial), pero Pytest quedó detenido al salir y se cerró con SIGINT, código 130. No se cuenta como ejecución limpia completa. Esa integración queda reservada a plataformas con `file_descriptor` (Linux) y pendiente en Khipu. El Mac usa `file_system`; se mantienen allí las 4 regresiones portables y las 81 pruebas previas. [Reporte de suite](../outputs/tests-inference-resources.xml), [reporte del ensayo de workers](../outputs/tests-inference-workers.xml). Falta confirmar la ejecución corregida bajo Linux/GPU en Khipu.

[Evidencia estructurada](../outputs/khipu-lstm-failed-53814.json). [Pasos de recuperación](recuperacion-lstm-53814.md): subir la corrección desde el Mac y empezar una corrida con 4 shards en `runs/lstm-seed42-fd-fix`, conservando el intento fallido. Falta verificar esa primera validación en Khipu y luego el reencolado. También faltan DMEL, ablación, persistencia, comparación final, análisis de errores y predicciones de test.

La suite local final terminó con salida 0: **85 pruebas aprobadas y 1 omitida en 20.44 s**. La omitida corresponde a la integración específica de memoria compartida `file_descriptor` de Linux.

## Primera época confirmada de la corrida corregida 53987

El log del 01/10 empieza a las 19:24:16 -05 en g002, con 4 shards, `restart=0`, `resume=0`, `epochs_per_cycle=1`. Completó el bucle de entrenamiento y la validación de la época 0 sin el error de archivos abiertos. Resultado: MSE de entrenamiento normalizado 0.45434031145421305, RMSE de validación 0.10770595145036646 mm/h, 525.9833223489113 s de entrenamiento y validación, pico CUDA asignado por PyTorch 235.9443359375 MiB. El campo `seconds` se mide antes de guardar historial/checkpoints; también excluye la inicialización y la espera de Slurm.

El estado impreso es `complete=false`, `reason=epoch_cycle_limit`, `epochs_completed=1`, `epochs_target=100`. El batch confirma `Época guardada en runs/lstm-seed42-fd-fix/last.pt` y solicita el siguiente ciclo (1/99 reinicios). A las 19:38:25 -05 Slurm registra `CANCELLED ... DUE TO JOB REQUEUE`. En este contexto corresponde al cierre solicitado para reencolar, no a un nuevo traceback. Las consultas posteriores muestran el mismo job `PD (BeginTime)`.

[Slurm define BeginTime](https://slurm.schedmd.com/job_reason_codes.html) como la espera hasta la hora mínima de inicio elegible. Aún no se conoce su `EligibleTime`; esa hora no garantiza asignación inmediata de recursos. No se confirma todavía que haya cargado el checkpoint en otro ciclo. La siguiente evidencia esperada es `restart=1`, `resume=1` y `start_epoch=1`.

[Registro de esta época](../outputs/khipu-lstm-53987-epoch0.json). Es un resultado intermedio LSTM sobre toda la validación, no una comparación final ni directamente comparable al piloto DMEL de 64 muestras. Mantener el mismo job; el proceso está preparado para continuar hasta 100 épocas o early stopping. No modificar código/datos/configuración mientras continúe esta corrida.

## Reanudación confirmada y ralentización en la cuarta época

El usuario compartió `scontrol`: `RUNNING`, `Restarts=1`, `EligibleTime=2026-10-01T19:40:26`, `StartTime=2026-10-01T19:40:43`. El log posterior confirma `restart=1 resume=1` y `training_start` con `resume=true,start_epoch=1`. La carga del checkpoint y el inicio de la segunda época quedaron confirmados. La hora de fin del ciclo era su límite de ejecución, no una estimación de finalización del entrenamiento.

El fragmento compartido el 02/10 alcanza `start_epoch=3`, con 254000 train/18142 validation y progreso hasta 6500/7938 lotes en 7797.362 s (2 h 9 min 57 s). Esto confirma la cuarta época en curso y la carga de un checkpoint que contiene tres épocas terminadas. No se recibieron aún sus RMSE ni el `status.json` actualizado. El nodo de este ciclo no aparece en el fragmento; no se presupone que continúe en g002 después de cada requeue.

La cuarta época lleva mucho más tiempo que los 525.983 s de la primera época con validación. Los últimos 1500 lotes tardaron 1275.089 s; extrapolar ese tramo para los 1438 lotes de entrenamiento pendientes da unos 1222.385 s (20.4 min), sin validación, y no garantiza el tiempo real. Los intervalos varían fuertemente. [Registro de progreso](../outputs/khipu-lstm-53987-epoch3-progress.json).

La causa sigue sin confirmarse. [Khipu documenta GPU compartida mediante sharding](https://docs.khipu.utec.edu.pe/guia-de-usuario/enviar-jobs/particiones/), pero el log no distingue competencia por GPU de CPU o lectura de datos. Siguiente diagnóstico: estado/tiempo restante de 53987, estadísticas CPU/RAM de `53987.batch` y una consulta `nvidia-smi` dentro de la asignación existente con `srun --jobid=53987 --overlap`. Conservar código, configuración y checkpoints mientras se observa el ciclo.

## Recursos y diagnóstico recibidos después

El usuario confirmó Slurm 25.11.5 y asignación `cpu=8,mem=32G,gres/shard=4` a 53987. La QoS `a-postgrado` tiene `GrpTRES=cpu=240,gres/shard=8,mem=180G` y `MaxTRESPU=cpu=32,gres/gpu=1,gres/shard=8,mem=98G`. El primer límite es agregado para la QoS; el segundo es por usuario. La consulta de esa QoS mostraba 53987 con 4 shards y otro trabajo con una GPU completa. La consulta posterior de trabajos `R,CG` en g002 solo mostraba 53987; esto no prueba ausencia de procesos externos ni permite atribuir la lentitud a competencia por GPU.

Dos intentos de `srun --jobid=53987 --overlap ... nvidia-smi`, uno heredando GRES y otro especificando `--gres=shard:rtxa6000:4`, fallaron al crear el paso con `Invalid generic resource (gres) specification`. No se obtuvo una medición de GPU y el error no acredita un fallo del entrenamiento. No repetir esas variantes como solución verificada.

`sstat` mostró `AveCPU=00:07:11,AveRSS=1520060K,MaxRSS=4819280K`, equivalente a unos 4.60 GiB de pico RSS registrado, y `fs/disk=17712844334`. Son métricas del paso según la contabilidad de Slurm: no medir utilización instantánea de CPU a partir de AveCPU ni deducir latencia del almacenamiento a partir del volumen de lectura. No aparecían métricas GPU. El ciclo `restart=4,resume=1,start_epoch=4` confirmó cuatro épocas guardadas y una quinta en curso, a 4500/7938 lotes en 663.367 s.

## Historial de 20 épocas recibido el 03/10/2026

[Datos copiados del usuario](../outputs/khipu-lstm-53987-history-20261003.json), [cálculos](../outputs/khipu-lstm-53987-analysis-20261003.json), [gráfica PNG](../outputs/khipu-lstm-53987-learning-20261003.png), [gráfica PDF](../outputs/khipu-lstm-53987-learning-20261003.pdf). La figura se reproduce con `python scripts/plot_lstm_history.py` y requiere Matplotlib. Estos son artefactos locales de evidencia recibida, no una descarga SSH verificada.

El historial abarca `epoch=0..19`. Su mejor RMSE es **0.10228306200404642 mm/h** en `epoch=9` (décima época), una reducción del **5.0349 %** respecto de la primera. Las diez épocas posteriores no mejoran el mínimo; `configs/lstm.json` tiene `patience=10` y `engine.train` solo reinicia la paciencia cuando `rmse < best`. Por tanto corresponde detener el entrenamiento después de la época 20, aunque `epochs=100` sea el máximo. El código debe evaluar desde `best.pt` y escribir `complete=true,stop_reason=early_stopping,epochs_completed=20` al terminar. Falta verificar ese estado remoto y `COMPLETED 0:0`: `history.json` se escribe antes del checkpoint, por lo que el historial solo no confirma el guardado y cierre final.

La última validación tiene RMSE **0.1066446566583386 mm/h**, un **4.2642 %** peor que el mínimo, mientras el MSE normalizado de entrenamiento sigue bajando. Este patrón es compatible con sobreajuste; usar el mejor checkpoint esperado, no seleccionar el último por ser más reciente. Es validación usada para seleccionar el modelo, no un resultado de test ni una comparación con DMEL o persistencia.

La suma de `seconds` es **42263.38058044482 s (11 h 44 min 23 s)**. Excluye inicialización, escrituras de historial/checkpoints, espera/reencolado y evaluación final. Las últimas 13 épocas (`epoch=7..19`) tardaron **6.81 min de media**, con rango **5.67–9.56 min**; las épocas anteriores muy lentas no justifican extrapolar ahora 2–3 h por época. El pico de tensores CUDA de PyTorch es 235.9443359375 MiB en todas las épocas; no incluye necesariamente toda la memoria del proceso GPU y no mide porcentaje de cómputo ni garantiza VRAM por shard. No se confirmó la causa de las variaciones de tiempo.

Verificación remota pendiente, desde Khipu:

```bash
sacct -j 53987 --format=JobID,State,ExitCode,Elapsed
cat /home/max.ramirez/dmel/runs/lstm-seed42-fd-fix/status.json
cat /home/max.ramirez/dmel/runs/lstm-seed42-fd-fix/validation/metrics.json
```

Sigue pendiente la ejecución completa de DMEL, la ablación, persistencia y comparación, además de predicciones finales y actualización de resultados en la presentación.

## Cierre de LSTM y validación final confirmados

El archivo de texto adjuntado después por el usuario contiene `sacct` con **53987 y 53987.batch COMPLETED 0:0, Elapsed 00:14:01**, `status.json` con `complete=true,epochs_completed=20,epochs_target=100,stop_reason=early_stopping` y `validation/metrics.json` íntegro. Esta evidencia resuelve la verificación pendiente de la sección anterior. El RMSE final coincide exactamente con el mínimo de `epoch=9`, coherente con la evaluación de `best.pt` implementada en el motor.

[Estado recibido](../outputs/khipu-lstm-53987-completed/status.json), [métricas recibidas](../outputs/khipu-lstm-53987-completed/metrics.json), [análisis](../outputs/khipu-lstm-53987-completed/analysis.json), [procedencia y hash del adjunto](../outputs/khipu-lstm-53987-completed/provenance.json), [registro Slurm](../outputs/khipu-lstm-53987-completed/sacct.txt). Se extrajeron los JSON del texto sin ejecutar sus comandos; no hubo SSH directo. No se han descargado los checkpoints ni las predicciones y no se conocen sus hashes finales.

Validación completa, sin clipping: **18142 ventanas × 48 horizontes = 870816 pares**, 508 cuencas. RMSE global **0.1022830620 mm/h**, MAE **0.0212754267 mm/h**, sesgo **−0.0002784329 mm/h**, NSE **0.6446957193**, KGE2009 **0.7220572584**, correlación **0.8029318694**. Hay **28350 predicciones negativas (3.2556 %)**. Estas métricas son de la misma validación usada para elegir el checkpoint, no de test ni de una serie cronológica reconstruida.

Por horizonte: RMSE/NSE a 1 h **0.0327054/0.968539**; a 24 h **0.1096932/0.566035**; a 48 h **0.1196458/0.479802**. La calidad se degrada al extender el horizonte. Por cuenca: **70/508 (13.7795 %) con NSE negativo** y **62 con KGE negativo**. NSE macro medio **−49.2525115**, mediana **0.5939118**; KGE macro medio **−0.8037602**, mediana **0.6496218**. Se conservarán las medias, los extremos y las medianas; no seleccionar solo las agregaciones favorables.

La cuenca 377 tiene NSE **−11659.519** con RMSE **0.0131392 mm/h**. Aplicando la identidad del cálculo implementado, `std(y)=RMSE/sqrt(1−NSE)`, su desviación observada es aproximadamente **0.00012168 mm/h** (inferida de métricas, no recalculada desde predicciones). La baja variabilidad explica la gran sensibilidad de NSE en ese caso, sin invalidar su mal resultado relativo. La media macro negativa no implica un fallo del job ni debe borrarse como si fuera un error numérico. Pendiente revisar predicciones por cuenca y comparar persistencia, LSTM y DMEL con el mismo criterio.

Comprobaciones locales sobre la evidencia: `status.global == metrics.global`; 48 horizontes con 18142 pares cada uno; 508 cuencas cuya suma de pares es 870816; consistencia de RMSE por suma de errores cuadrados, MAE, sesgo y fracción negativa entre agregaciones; medias macro recalculadas; mejor RMSE del historial idéntico al final. No equivale a recalcular desde `predictions.npz`, que todavía no está disponible localmente.

Siguiente envío preparado y comprobado con `--dry-run` (no enviado por el agente):

```bash
cd /home/max.ramirez/dmel
CACHE=cache/train.h5 TRAIN_GRES=shard:rtxa6000:4 bash slurm/submit_experiments.sh --model dmel --submit
squeue --me
```

Usa `configs/dmel.json`, `runs/dmel-seed42`, la cache completa, 8 CPU/32 GB de RAM/4 shards RTX A6000 y hasta 8 h por ciclo. Mantiene 100 épocas máximas, paciencia 10, semilla 42, una época por ciclo y reanudación automática. No se vuelve a ejecutar LSTM ni CEEMDAN. La duración de DMEL debe medirse con su propia primera época completa.

## DMEL 54181 en cola

El usuario confirmó después el envío mediante `squeue --me`: `54181 gpu lab2_dme max.rami PD 0:00 1 (Resources)`. [Slurm define Resources](https://slurm.schedmd.com/job_reason_codes.html) como falta de disponibilidad de los recursos solicitados; esa salida no identifica qué recurso falta ni permite estimar cuándo iniciará. Mantener el envío existente. Cuando cambie a `R`, consultar el log previsto por el helper: `/home/max.ramirez/dmel/logs/lab2_dmel_54181.out`; el camino efectivo se puede verificar en `scontrol show job 54181` con `StdOut`. Mientras esté pendiente puede no existir todavía el log.


## Cierre de DMEL y comparación completa recibida

El log adjuntado por el usuario confirma **54181 y 54181.batch COMPLETED 0:0**, con 20 épocas consecutivas, `complete=true` y `stop_reason=early_stopping`. El mejor checkpoint corresponde a `epoch=9`, igual que en LSTM. Después se recibió `validation/metrics.json` íntegro mediante respuesta textual: 48 horizontes y 508 cuencas. Esto resuelve los estados pendientes históricos de las secciones anteriores. No volver a enviar estas corridas completadas.

[Estado](../outputs/khipu-dmel-54181-completed/status.json), [historial](../outputs/khipu-dmel-54181-completed/history.json), [métricas completas](../outputs/khipu-dmel-54181-completed/metrics.json), [análisis y comparación pareada](../outputs/khipu-dmel-54181-completed/analysis.json), [procedencia](../outputs/khipu-dmel-54181-completed/provenance.json) y [log recibido](../outputs/khipu-dmel-54181-completed/terminal-evidence.txt). Se copiaron y comprobaron los datos del usuario sin ejecutar comandos remotos. Los checkpoints y `predictions.npz` todavía no están disponibles localmente.

| Métrica de validación | LSTM | DMEL |
|---|---:|---:|
| RMSE global, mm/h | 0.1022830620 | 0.1056661151 |
| MAE global, mm/h | 0.0212754267 | 0.0222680635 |
| Sesgo global, mm/h | −0.0002784329 | 0.0007215198 |
| NSE global | 0.6446957193 | 0.6208033608 |
| KGE 2009 global | 0.7220572584 | 0.7188083987 |
| Predicciones negativas | 28350 (3.2556 %) | 9439 (1.0839 %) |
| Horizontes con menor RMSE | 48 de 48 | 0 de 48 |
| Cuencas con menor RMSE | 256 de 508 | 252 de 508 |
| Cuencas con NSE negativo | 70 | 80 |
| NSE macro medio | −49.2525115 | −83.5097907 |
| NSE mediano por cuenca | 0.5939118 | 0.5623444 |
| KGE macro medio | −0.8037602 | −1.3648343 |
| KGE mediano por cuenca | 0.6496218 | 0.6238706 |

El RMSE global de DMEL es **3.3075 % mayor**, y su MAE **4.6656 % mayor**. DMEL reduce la fracción de salidas negativas en **2.1716 puntos porcentuales**, pero tiene mayor sesgo absoluto. LSTM también tiene menor MAE y mayor NSE en los 48 horizontes; DMEL mejora KGE en 14, LSTM en 34. Por cuenca, DMEL tiene menor MAE en 175 y mayor KGE en 208. No hay empates en estas comparaciones. El recuento de cuencas ganadas no pondera la magnitud del error ni el número de pares y no sustituye las métricas globales.

Las medias macro negativas deben mostrarse junto a las medianas. Ejemplo DMEL, cuenca 369: RMSE **0.0031239491 mm/h**, NSE **−20042.7835**. La identidad `std(y)=RMSE/sqrt(1−NSE)` implica desviación observada **0.0000220655 mm/h**. La baja variabilidad explica la sensibilidad del denominador, sin eliminar el mal desempeño relativo. Los valores no se recortaron ni se descartaron.

Controles locales: ambos reportes contienen exactamente los mismos 48 horizontes, 508 IDs de cuenca y conteos por grupo. La suma es **870816 pares = 18142 × 48**. Los agregados ponderados de RMSE, MAE, sesgo y fracción negativa coinciden con el global. La varianza observada inferida desde RMSE/NSE coincide entre modelos en todos los grupos. `status.global` coincide con `metrics.global`, y el RMSE final con el mínimo del historial. Esto comprueba consistencia de los reportes, no sustituye una auditoría de predicciones individuales.

Tiempo DMEL: suma de entrenamiento y validación por época **14189.7566 s (3 h 56 min 30 s)**, mediana **527.05 s**, mínimo **468.77 s** y máximo **3372.91 s**. Excluye inicialización, checkpoint, cola y evaluación final. `sacct` muestra **00:13:01 para el último ciclo**. El pico de tensores CUDA de PyTorch es **746.854 MiB**. Los tiempos de LSTM y DMEL provienen de asignaciones distintas y variaciones de rendimiento no diagnosticadas; no permiten afirmar que una arquitectura sea más rápida por su total acumulado.

El log comunica SHA-256 de `best.pt`: `b384943057e1434bd15c2a7aeb8018b165375b99f6eb03b82a4ab826cbe60569`, y de `config.json`: `1c72aa7a0f8cdb62f56e33f3355b00b41e585bda5c0b2bf91e6e0998311a13e3`. Son hashes reportados remotamente, pendientes de contraste con copias descargadas.

La [PPT comparativa](../outputs/presentacion/DMEL-UTEC-comparacion-LSTM-DMEL.pptx) y el [guion](../outputs/presentacion/guion-dmel-comparacion.txt) recogen estos resultados. La conclusión se limita a esta adaptación, una semilla y validación utilizada para selección. Pendientes: persistencia, ablación con dos Informer, inspección de predicciones, test y material final de entrega. No se ha ejecutado SSH ni publicado en GitHub.
