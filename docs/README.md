# Guías del proyecto

## Uso actual

| Guía | Contenido |
|---|---|
| [Instalación](instalacion.md) | Python 3.12, CPU, CUDA, gráficos y pruebas |
| [Dataset](../dataset/README.md) | Archivos, dimensiones, particiones y procedencia |
| [Workstation Linux](migracion-workstation.md) | Trasladar el respaldo verificado y usar los pesos existentes |
| [GitHub](github.md) | Contenido versionable y publicación manual |
| [Implementación](implementation.md) | Modelos, normalización, causalidad y evaluación |
| [Descomposición](decomposition.md) | CEEMDAN propio, entropía y agrupación |
| [Configuraciones](../configs/README.md) | Experimentos finales y diagnósticos |
| [Scripts](../scripts/README.md) | Función y efectos de cada utilidad |
| [Slurm](../slurm/README.md) | Jobs y recursos en Khipu |
| [Resultados](../outputs/README.md) | Métricas finales, presentación y evidencia histórica |
| [Cierre del laboratorio](cierre-lab2.md) | Ablación, test y video terminados; auditoría y entrega |
| [Video final](../outputs/video-dmel/README.md) | Duración, bloques, respaldo de Resolve y evidencia |

## Registro de Khipu

[Estado de Khipu](estado-khipu.md) conserva la cronología y las salidas recibidas. LSTM 53987 y DMEL 54181 terminaron correctamente; el respaldo local contiene ambas corridas.

Las guías [paso a paso](khipu-paso-a-paso.md), [entrenamiento](entrenamiento-khipu.md) y [recuperación del fallo 53814](recuperacion-lstm-53814.md) documentan el procedimiento empleado. Sus instrucciones de envío son históricas: revisar el estado actual antes de iniciar una nueva corrida. No repetir el precompute de train ni los entrenamientos completados para trasladar el proyecto.
