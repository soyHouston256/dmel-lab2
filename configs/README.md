# Configuraciones

| Archivo | Uso | Estado |
|---|---|---|
| [lstm.json](lstm.json) | LSTM sobre las 12 variables originales | Corrida 53987 completa |
| [dmel.json](dmel.json) | Informer para grupo alto + LSTM para grupo bajo | Corrida 54181 completa |
| [mrs_informer.json](mrs_informer.json) | Informer en ambas ramas | Ablación pendiente |
| [decomposition.json](decomposition.json) | CEEMDAN y SampEn para datos reales | Caché train completa |
| [smoke.json](smoke.json) | Red pequeña, 16 train/8 validation, dos épocas | Diagnóstico sintético |
| [decomposition-smoke.json](decomposition-smoke.json) | Descomposición reducida | Solo diagnóstico |
| [pilot-full.json](pilot-full.json) | Arquitectura final, 256 train/64 validation, una época | Dimensionamiento |

Los tres experimentos finales comparten batch 32, Adam con tasa `1e-4`, semilla 42, máximo 100 épocas, paciencia 10, dos workers y ocho hilos. Los límites de muestras son cero (todos los datos del split). Ver [implementación](../docs/implementation.md) para capacidad y diferencias entre modelos.

Para un experimento nuevo, copiar el JSON con otro nombre y elegir una carpeta de salida nueva. La configuración efectiva se guarda en cada corrida. Cambiar código, datos, caché, configuración o la ruta absoluta del dataset invalida la reanudación estricta; no editar los hashes de una corrida para forzarla.

`--epochs-per-cycle` controla una pausa operativa. El valor `0`, predeterminado en la CLI, permite entrenar sin reencolar; Slurm utiliza normalmente una época por ciclo. Las configuraciones de diagnóstico no son checkpoints reutilizables para los modelos finales.
