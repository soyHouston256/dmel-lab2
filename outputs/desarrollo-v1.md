# Desarrollo V1 y evidencia de verificación

Nota de actualización: este documento conserva la evidencia de la primera versión. Posteriormente el usuario confirmó datos y piloto GPU en Khipu; el estado vigente y la adaptación al límite de 8 h están en [estado-khipu.md](../docs/estado-khipu.md).

Fecha: 26/09/2026. **Implementación funcional de la adaptación; entrenamiento hidrológico pendiente.** Entorno elegido para entrenar: Khipu. Se utilizaron pruebas locales CPU para verificar el código, sin consumir GPU remota.

## Lo implementado

- `dmel/data.py`: esquema HDF5, selección de flags oficiales, normalización solo con train y carga por proceso.
- `dmel/decomposition.py`: EMD/CEEMDAN propio, SampEn, agrupación explícita y reconstrucción con residuo.
- `dmel/models.py`: ProbSparse, encoder con reducción temporal, decoder causal, LSTM y ensamble; baseline y ablaciones.
- `dmel/precompute.py`: caché con checksum, firma de implementación, cola acotada y reanudación de parciales compatibles.
- `dmel/engine.py` y `cli.py`: entrenamiento, early stopping, Adam, best/last, RNG, evaluación y CSV de predicciones.
- `dmel/metrics.py`: RMSE/MAE/sesgo/NSE/KGE2009 globales, por horizonte y cuenca, con valores indefinidos explícitos.
- `scripts/` y `slurm/`: prueba sintética, medición de descomposición, transferencia y jobs de Khipu.

Las decisiones científicas y divergencias del artículo están en [implementation.md](../docs/implementation.md). La regla de dos grupos simplifica MRS; la pérdida conjunta no identifica contribuciones físicas por rama. No se presentan resultados como réplica exacta.

## Verificaciones realizadas

Comando: `/opt/anaconda3/bin/python -m pytest tests -q --junitxml=outputs/tests-v1.xml`.

**46 pruebas aprobadas, 0 fallos, 6.95 s.** [Registro JUnit](tests-v1.xml).

Cobertura sustantiva: reconstrucción CEEMDAN con residuo y truncamiento; entropía contra conteo independiente; constantes/monótonas; determinismo sin alterar RNG global; cambios fuera del historial no afectan componentes; cambios en etiquetas no alteran preprocesamiento; normalización independiente de validation; suma correcta de escalas; gradientes en ambas ramas; invariancia de evaluación al tamaño del batch; límite de atención completa; métricas perfectas/degeneradas; entrenamiento, recarga y exportación; reanudación tras interrupción simulada igual a ejecución continua; precompute paralelo igual a serial; selección smoke que cubre exactamente train/validation; rechazo de clipping inválido.

La sintaxis de **cada** script shell y sbatch se comprobó con `bash -n`. Se inspeccionaron dry-runs de transferencia y de la cadena Slurm, incluida ruta de caché personalizada. No se enviaron jobs durante esos dry-runs. CLI `inspect` comprobó el archivo real de test; [registro](inspection-implementation.json).

## Prueba de extremo a extremo

Comando: `/opt/anaconda3/bin/python -m scripts.smoke --out runs/synthetic-smoke-v2`.

Generó 16 muestras sintéticas de entrenamiento, 8 de validación y 5 de test; descompuso con configuración pequeña, entrenó dos épocas en CPU, guardó/recargó el checkpoint y exportó `5×48` predicciones. El resumen local está en `runs/synthetic-smoke-v2/smoke-summary.json` y las predicciones sintéticas en `runs/synthetic-smoke-v2/model/test/predictions.csv`. Esos archivos de ejecución no se incluyen en Git; se pueden regenerar con el comando anterior, usando una salida nueva.

La pérdida de train normalizada pasó de 0.10794 a 0.07401 y RMSE sintético de validation de 0.08154 a 0.06886. Solo demuestra funcionamiento del pipeline. Los objetivos sintéticos fueron persistentes por construcción; no prueba ventaja frente a persistencia ni habilidad hidrológica.

Entorno comprobado: Python 3.12.12, torch 2.7.1, NumPy 1.26.4, SciPy 1.16.3, h5py 3.15.1 y pytest 8.4.2. CPU macOS, sin CUDA/MPS disponibles en esta sesión. No se instaló ni modificó el entorno global. `requirements.txt` registra esas versiones; `requirements-khipu.txt` propone el build CUDA ya usado en Sonidos, pendiente de nueva validación remota.

CEEMDAN final, con 100 realizaciones, se comprobó en cuatro historias reales sin leer objetivos: mediana 0.886 s/ventana, reconstrucción máxima ~1.20e-7 mm/h. Los detalles y límites de extrapolación están en [plan-khipu.md](plan-khipu.md).

## Datos y acceso pendientes

`train.h5` sigue ausente localmente, pero **se localizó en la carpeta oficial del laboratorio** mediante Google Drive. [Archivo original](https://drive.google.com/file/d/1Ou6LEqX0eflNhf0AA_qp-lXd0SPlm6GG/view?usp=drivesdk): 5,076,559,230 bytes. La descarga por el conector devolvió HTTP 413: supera su límite de 268,435,456 bytes. No se descargó un archivo parcial. Conviene transferirlo directamente a Khipu después de autenticar acceso; se registró el enlace en [dataset-source.json](dataset-source.json).

El listado oficial también contiene `test.h5`, `test_targets.csv`, `metadata.json` y `leer_datos.py`, con tamaños compatibles con las copias locales. Esto confirma su presencia, pero no aclara el uso autorizado del CSV ni verifica checksums de los archivos remotos.

SSH alcanzó Khipu y terminó con `Permission denied`; no se verificaron cuotas, paquetes o GPU actuales. No se copiaron datos ni se ejecutó entrenamiento remoto. [Instrucciones operativas](../slurm/README.md).

Siguiente fase: disponer de train y autenticar Khipu, auditar datos, medir piloto con dimensiones finales y ejecutar comparación. Después faltan gráficos/errores, predicción final, informe y video. El método de agrupación deberá caracterizarse en train antes de extraer conclusiones; el test no se usa para ajustarlo.
