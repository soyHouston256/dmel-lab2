# Reutilización de Sonidos para DMEL

Revisión 26/09/2026. Inspeccionados en `../sonidos`: guía Slurm, scripts de train/precompute/envío, sincronización, dependencias, configuraciones y logs 50993/51001. No se modificó Sonidos.

Los logs muestran Python 3.12.13, torch 2.7.1+cu126, RTX A6000 y 8 shards en entrenamiento; precompute 32 CPU/16GB, NumPy 1.26.4/SciPy 1.16.3. Las cuatro corridas finales y recuperación están documentadas como completas. Es evidencia histórica, no verificación de disponibilidad actual.

Esta revisión alcanzó el servidor por SSH pero falló autenticación: `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)`. No se consultaron recursos actuales ni se enviaron jobs.

| Flujo Sonidos | Adaptación implementada |
|---|---|
| rsync código/datos/resultados | `scripts/sync_khipu.sh`, directorio independiente, sin borrado |
| Features audio CPU | CEEMDAN/SampEn por ventana, cache HDF5, cola acotada |
| GPU mediante sbatch | Scripts Slurm con operación CUDA y versiones |
| Piloto y full | Config smoke separada; medir dimensiones finales antes de escalar |
| Comparación y ablación | Persistencia, LSTM, DMEL adaptado, MRS-Informer |
| Dependencias/reanudación | afterok, best/last, hashes de datos/código/cache y RNG |
| Recuperación | best.pt, predicciones, métricas y logs; caches remotas |

No se reutilizan pesos, datos o métricas de audio. Tampoco su lr 1e-3: la configuración final mantiene 1e-4 de DMEL. La memoria medida en audio no certifica la de DMEL.

CEEMDAN con 100 realizaciones se midió en historias reales de 336 h, Ids 0, 1, 100, 500, sin objetivos: **0.895,0.749,1.025,0.878s**; mediana 0.886 s. Error de reconstrucción máximo 1.20e-7 mm/h. Había otras comprobaciones locales en curso: son observaciones aproximadas de costo. [Registro completo](decomposition-benchmark.json).

Extrapolar aritméticamente la mediana a 272,142 ventanas da aproximadamente 67 horas de un worker, o 4.2 h ideales con 16. **No es tiempo validado de Khipu**: faltan I/O, colas, sobrecostes y representatividad de cuencas. El piloto remoto decidirá recursos. La GPU se reserva para las redes y se reutiliza una única cache entre corridas comparables.

En tres de cuatro historias todos los IMFs fueron a alta y el residuo a baja; en la cuarta tres IMFs también fueron a baja. Se deberá caracterizar el enrutamiento sobre entrenamiento. No se cambió el método para ajustarlo al test.

Según metadatos, X/y/y_aux train+val ocupan ~4.67GiB, sin overhead. Cache de dos canales añade ~0.68GiB antes de diagnósticos. Se usa carga por lotes.

Faltan `train.h5`, autenticación, verificación GPU y entrenamiento real. [Guía operativa](../slurm/README.md) y [decisiones científicas](../docs/implementation.md).
