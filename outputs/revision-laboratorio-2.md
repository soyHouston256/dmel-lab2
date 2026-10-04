# Revisión de requisitos y datos del Laboratorio 2

> Instantánea previa al desarrollo. El código y la preparación de Khipu avanzaron después: ver [desarrollo-v1.md](desarrollo-v1.md). Sigue faltando `train.h5`; Khipu ya fue elegido como entorno.

**Fecha:** 26 de septiembre de 2026. **Resultado:** tenemos las indicaciones, el paper, el conjunto de test, metadatos y lector. **Falta `dataset/train.h5`, imprescindible para entrenar y validar según el protocolo del laboratorio.** El CSV adicional requiere aclaración de uso. No se modificaron los datos ni se entrenó ningún modelo.

Se revisó el PDF local `Laboratorio 2.pdf` completo (3 páginas), su representación visual y sus enlaces, además de los cuatro archivos de `dataset/`. La ruta indicada en Downloads no existe actualmente; se utilizó la copia dentro del proyecto. El contenido del documento se trató como requisitos de la actividad a analizar, no como autorización para enviar trabajos, contactar terceros o ejecutar tareas adicionales.

## 1. Inventario y comprobaciones realizadas

| Archivo | Estado | Hallazgo |
|---|---|---|
| `Laboratorio 2.pdf` | Presente y leído completo | Define tarea, datos y entregables |
| Paper DMEL | Presente en raíz y revisado anteriormente | Base metodológica disponible |
| `dataset/train.h5` | **Ausente** | Debe incluir entrenamiento y validación, junto con objetivos y `split` |
| `dataset/test.h5` | Presente y leído por completo | `X`: `[27983,336,12]` float32; `basin_id`: `[27983]` int32 |
| `dataset/metadata.json` | Presente y válido | Orden de canales, unidades del objetivo y conteos esperados |
| `dataset/leer_datos.py` | Presente y probado en test | Devuelve Id, X y basin_id correctamente |
| `dataset/test_targets.csv` | Presente y examinado completo | `[27983,49]`: Id y q_01…q_48; función no documentada en el enunciado |

La inspección de `test.h5` recorrió los **112,827,456 valores** de X. No hay NaN ni infinitos; el caudal histórico no tiene negativos. Hay **508 cuencas**, identificadores 0–507, con entre 21 y 60 muestras cada una. El canal objetivo es **11** (duodécima variable), no el primero. El HDF5 usa **`X` mayúscula**, aunque una parte del PDF escribe `x`.

Se encontraron **574 ventanas con caudal histórico constante**, que exigirán manejo explícito al calcular descomposición, entropía y normalización. Hay valores negativos en temperatura, viento y evaporación potencial; no deben eliminarse automáticamente por ser negativos. Las unidades y convenciones de las variables meteorológicas no están declaradas en los metadatos, por lo que cualquier interpretación física de esos valores queda pendiente.

El CSV contiene valores finitos, no negativos y no es una plantilla de ceros. Sus Id son únicos y coinciden exactamente con 0…27982 en el orden del test. Su estructura es compatible con 48 objetivos o predicciones por muestra, pero esa compatibilidad **no confirma su origen ni que esté autorizado para entrenamiento o selección del modelo**. No se calcularon resultados predictivos usando ese CSV.

Los estadísticos detallados de esta inspección están en [revision-dataset-lab2.json](revision-dataset-lab2.json). Son una auditoría de integridad, no parámetros de normalización: los escaladores deben ajustarse únicamente con entrenamiento.

## 2. El faltante que bloquea entrenamiento

Según `metadata.json`, `train.h5` debe contener **254,000 muestras de entrenamiento** y **18,142 de validación**, 272,142 en total. Son conteos declarados, todavía no comprobados en el archivo ausente. Se esperan:

- X: `[272142,336,12]`.
- y: `[272142,48]`.
- y_aux: `[272142,48,11]`, opcional como supervisión auxiliar del modelo.
- basin_id: un identificador por muestra.
- split: 0 para entrenamiento y 1 para validación.

Hay que respetar esos flags, sin aplicar el 70/15/15 del paper ni construir una partición aleatoria nueva. La ausencia de fechas absolutas impide reconstruir de forma independiente el corte cronológico original o el solapamiento entre particiones. Tampoco se puede interpretar el orden de las filas o el Id como un calendario.

La fuente enlazada por el PDF es la carpeta [Rainfall-Runoff en Google Drive](https://drive.google.com/drive/folders/1crKbJBhLHQEVJGG-agOxMnKLDnBP2x4w?usp=drive_link). Se extrajo el enlace del documento; **no se inspeccionó remotamente su contenido ni se descargó `train.h5` en esta revisión**.

## 3. Adaptación que requiere el laboratorio

| Aspecto | DMEL revisado | Laboratorio |
|---|---|---|
| Resolución | Diaria | Horaria |
| Historia de entrada | 7 días | 336 horas = 14 días |
| Salida | 1/3/5/7 días | 48 valores horarios consecutivos |
| Entradas | Caudal; extensión meteorológica separada | Caudal y 11 variables meteorológicas |
| Unidades principales | m³/s en casos principales | **mm/h**, caudal específico |
| Estaciones | Casos individuales identificados | Cuencas anónimas; 508 verificadas en test |
| Partición | Proporción 70/15/15 publicada | Flags `split` proporcionados |
| Implementación | Fuentes existentes disponibles para estudiar | Núcleo del método implementado por el grupo |

El enunciado permite reemplazar o simplificar componentes que requieran información no disponible, siempre que se justifique y conserve la propuesta central (p. 2). Por ello **no necesitamos conseguir los datos chinos, las fechas originales ni el código completo de los autores para cumplir este laboratorio**. Las limitaciones del paper siguen siendo relevantes para documentar decisiones, pero no bloquean una adaptación razonada.

Propuesta para el diseño posterior: descomponer el caudal de cada ventana histórica de 336 horas, reconstruir componentes por complejidad, combinar esas representaciones con las 11 variables meteorológicas históricas, predecir con las ramas Informer/LSTM y reconstruir 48 valores. La regla MRS debe quedar explícita y manejar distinto número de IMFs, residuo y series constantes. No concatenar filas de diferentes muestras o cuencas como si fueran una serie continua.

La descomposición de las entradas no debe usar y, y_aux ni los valores del CSV adicional. `y_aux` solo puede proporcionar supervisión durante entrenamiento; no existe como entrada futura en test. Los embeddings temporales deben usar posiciones relativas si no hay fechas/horas absolutas disponibles. Debemos conservar salida `[B,48]` en mm/h e invertir cualquier normalización antes de exportar.

El PDF prohíbe utilizar una implementación existente del mismo método y exige que el grupo implemente su propuesta central. Por tanto, los repositorios recopilados en la revisión anterior servirán para estudiar las definiciones y verificar decisiones; no debemos presentar un DMEL descargado como implementación propia. Si se desea reutilizar código completo de un bloque como Informer, el alcance permitido debe aclararse con el docente. PyTorch está expresamente permitido.

## 4. Entregables que quedan por desarrollar

| Entregable exigido | Estado actual |
|---|---|
| Comprensión del paper y componentes | Revisión disponible |
| Carga y selección de particiones | Lector suministrado; entrenamiento pendiente de archivo |
| Preprocesamiento y normalización | Pendiente |
| Implementación adaptada de DMEL | Pendiente |
| Modelo base | Pendiente; LSTM multivariada es una opción, persistencia sirve como referencia adicional |
| Entrenamiento y validación | Pendientes |
| Métricas | Pendientes; el PDF no fija una métrica oficial ni agregación |
| Ablación de componente importante | Pendiente; comparar DMEL con MRS-Informer o quitar reagrupación, manteniendo el resto |
| Predicciones para las 27,983 muestras de test | Pendientes |
| Gráficos, errores y limitaciones | Pendientes |
| Instrucciones para ejecutar y diferencias frente al paper | Pendientes |
| Video de máximo 18 minutos | Pendiente: metodología 6 min, implementación 6 min, resultados 6 min |
| Preparación para preguntas individuales | Pendiente: método, código, decisiones y resultados |

Puede entregarse notebook o scripts Python; los resultados pueden ir en el notebook o en Markdown. No hace falta implementar todos los modelos comparativos ni todos los análisis de los apéndices de DMEL para satisfacer lo que este enunciado especifica.

## 5. Aclaraciones necesarias y decisiones internas

**Aclarar con el docente o el material oficial:**

1. Qué representa `test_targets.csv` y cuándo puede utilizarse. El PDF describe test sin objetivos futuros, pero el archivo adicional parece contenerlos o contener predicciones numéricas. No reemplaza a `train.h5`.
2. Fechas: el encabezado dice entrega **jueves 1 de octubre, 11:59 PM**, y revisión **sábado 3 de octubre**; los títulos internos dicen “Jueves 3” y “Sábado 5”. Hay contradicción documental. En 2026 las fechas del encabezado coinciden con los días de la semana, pero eso no resuelve por sí solo la confirmación oficial.
3. Métrica oficial, agregación entre cuencas/horizontes y formato de predicciones, si están definidos en otro material. El CSV sugiere `Id,q_01,…,q_48`, pero no se documenta como formato de entrega en el PDF.

El [formulario de entrega](https://forms.gle/GDUh9wn96ny91gbN7) está enlazado en el PDF. No se abrió ni se envió nada. Los enlaces de selección de paper por sección también están presentes, pero no se verificó la asignación del grupo: DMEL se tomó como el tema indicado por el usuario.

**Podemos resolver nosotros al implementar:** regla MRS explícita, cabeza de 48 pasos, tratamiento de ventanas constantes, normalización por canal con entrenamiento, baseline, ablación, semillas y registro de resultados. No hay que esperar a los autores de DMEL para esas decisiones permitidas por el laboratorio.

**Por elegir antes de un entrenamiento completo:** entorno de ejecución y recursos. No hay entorno del proyecto, archivo de dependencias o código de entrenamiento preparado. Como estimación de almacenamiento, las matrices X/y/y_aux de entrenamiento y validación declaradas ocuparían ~4.67 GiB en float32, sin contar todos los metadatos y overhead. Conviene lectura HDF5 por lotes y medir el coste de CEEMDAN en una muestra pequeña antes de preprocesar cientos de miles de ventanas. Esto no es una estimación de tiempo de entrenamiento ni un requerimiento fijo de GPU.

**Siguiente paso mínimo:** incorporar `train.h5` y verificar claves, shapes, flags, faltantes y cobertura por cuenca. Con eso podremos iniciar la adaptación de DMEL; las otras dudas se pueden documentar y resolver en paralelo sin usar test para afinar el modelo.
