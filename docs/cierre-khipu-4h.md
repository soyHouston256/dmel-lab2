# Plan reducido para cuatro horas · Khipu

## Resultado del plan, verificado el 08/10/2026

Se completaron los dos requisitos y se descargaron sus artefactos al Mac. La [auditoría local](../outputs/verification/cierre4h-local-20261008/README.md) verifica ambos tests de 27 983 muestras y recalcula la ablación: control DMEL RMSE 0.110024; dos Informer 0.111464 mm/h (+1.31 %). Son tres épocas y una semilla; no se infiere convergencia.

La ejecución manual del chat separó cuatro jobs: 54931 test LSTM (32 s), 54932 caché test (2 h 11 min 57 s), 54933 control y ablación (65.17 min sumados en historiales, sin sacct final recibido) y 54935 test DMEL (43 s). El job 54933 terminó sus dos modelos con 8 CPU, 32 GB y 2 shards, reducidos de la solicitud inicial. Sus carpetas reales son `dmel-3epochs/` y `mrs_informer-3epochs/`, bajo `runs/cierre-4h-seed42-20261008/`. Este nombre con guion bajo difiere del wrapper preparado.

**No repetir los envíos de abajo.** Se conservan como planificación previa, no como historial literal de comandos ejecutados. La siguiente etapa es terminar el video y realizar la entrega.

## Plan previo

El usuario fijó un plazo de cuatro horas. Este plan sustituye el envío largo anterior; comprobar `squeue --me` antes de enviar para evitar duplicados. El usuario ejecuta SSH, transferencias y sbatch. No se abrió una conexión remota desde el agente.

## Experimento que sí podemos defender

Comparar **DMEL y MRS–Informer, ambos entrenados desde cero durante 3 épocas**, sobre train y validation completos. Misma semilla, batch, optimizador, normalización y datos; cambia la rama baja. Evaluar el mejor checkpoint dentro de cada presupuesto de 3 épocas. No comparar una ablación de 3 épocas como si fuera equivalente al DMEL original de 20 épocas. Los checkpoints originales se conservan para las predicciones de test.

El informe identifica el presupuesto de 3 épocas y advierte que no demuestra convergencia. Es una ablación de presupuesto limitado; una semilla y distinta cantidad de parámetros siguen siendo limitaciones.

## Presupuesto operativo

| Trabajo | Duración estimada sin cola | Límite de job |
|---|---:|---:|
| LSTM: test completo + control DMEL 3 épocas + ablación 3 épocas + comparación | 1,5–2,5 h en total | 3 h |
| Caché de test, 16 CPU | Referencia histórica 2,3–2,5 h; puede excederla | 3 h |
| DMEL original: test completo, después de caché | 5–20 min | 30 min |

El primer job exporta LSTM antes de entrenar. La caché avanza en paralelo. No reencolar entre épocas ni tras agotar el presupuesto. El límite de job se cuenta desde que empieza a ejecutarse: **no limita la espera en cola ni garantiza el plazo de cuatro horas**. Si la caché no termina, conservar `.partial`; el test DMEL queda pendiente y solo puede acreditarse el CSV LSTM que se haya verificado. Si el entrenamiento agota el tiempo antes de terminar ambos modelos, no declarar la ablación completa.

La guía pide implementar el método y generar todas las predicciones de test; para evitar depender de que se acepte una entrega solo del baseline, intentar también la exportación DMEL. No presentar un CSV LSTM como si proviniera de DMEL.

## Comandos preparados

Primero, en Khipu, consultar `squeue --me`. Si hay trabajos del plan largo anterior, revisar sus IDs/estado antes de decidir cuáles cancelar o reutilizar. No cancelar otros trabajos por nombre parcial.

En el Mac:

```bash
cd /Users/maxhoustonramirezmartel/code/utec/5/deep/dmel
scp scripts/lab2_closure.py max.ramirez@khipu.utec.edu.pe:/home/max.ramirez/dmel/scripts/
scp configs/dmel-3epochs.json configs/mrs_informer-3epochs.json max.ramirez@khipu.utec.edu.pe:/home/max.ramirez/dmel/configs/
scp slurm/cierre_4h.sbatch slurm/cierre_test_dmel_4h.sbatch max.ramirez@khipu.utec.edu.pe:/home/max.ramirez/dmel/slurm/
```

En una misma sesión de Khipu, después de confirmar que no hay envíos duplicados:

```bash
cd /home/max.ramirez/dmel
mkdir -p logs outputs/verification
JCIERRE=$(sbatch --parsable slurm/cierre_4h.sbatch)
echo "$JCIERRE"
```

Antes de enviar precompute, revisar si ya hay una caché de test completa o en ejecución; reutilizar una compatible cuando proceda. La siguiente ruta nueva es para un inicio sin otro job trabajando en ella:

```bash
JCACHE=$(sbatch --parsable --job-name=lab2_cache4h --time=03:00:00 --no-requeue \
  --export=ALL,DATA=dataset/test.h5,CONFIG=configs/decomposition.json,OUT=cache/test-cierre-4h-20261008.h5,SPLIT=test,LIMIT=0,SELECTION_CONFIG=,MAX_SECONDS=9600,AUTO_REQUEUE=0,MAX_RESTARTS=0 \
  slurm/precompute.sbatch)
echo "$JCACHE"
JTEST=$(sbatch --parsable --dependency="afterok:${JCACHE%%;*}" slurm/cierre_test_dmel_4h.sbatch)
echo "$JTEST"
printf 'CIERRE=%s\nCACHE=%s\nTEST_DMEL=%s\n' "$JCIERRE" "$JCACHE" "$JTEST" > logs/cierre-4h-jobids.txt
squeue --me
```

`MAX_SECONDS=9600` detiene nuevas ventanas a las 2 h 40 min y drena las tareas en curso. Es un margen, no una garantía contra timeout. Si sale con código 75, la caché es parcial y el job dependiente no arranca: revisar al acercarse al plazo. Máxima solicitud si coinciden los tres: 32 CPU, 80 GB RAM y 6 shards. La disponibilidad efectiva la decide Slurm.

Consultar el job principal:

```bash
tail -n 20 "logs/lab2_cierre4h_${JCIERRE%%;*}.out"
```

Resultados: `runs/cierre-4h-seed42-20261008/` y `outputs/verification/cierre-4h-seed42-20261008/`. Los JSON de verificación solo se generan tras completar sus etapas. Descargar estas dos carpetas para incorporar resultados a PPT/informe, reservando 30–45 min para revisión y cierre.
