# Especificación propuesta de implementación DMEL

> Documento previo al desarrollo, orientado al experimento diario original. El contrato horario y la implementación del laboratorio se describen en [docs/implementation.md](../docs/implementation.md).

Estado: diseño previo al código. Valores etiquetados **publicado** provienen del PDF; **propuesto** son decisiones para una implementación verificable; **pendiente** impide afirmar equivalencia exacta con los autores. No se instalaron dependencias ni se ejecutó entrenamiento.

## 1. Contrato de datos

Una fila por estación y día: `station_id`, `date`, `discharge`, `discharge_unit`, `quality_flag`. Para multivariado, columnas meteorológicas con unidades y fuente propias. El cargador debe:

- Conservar el archivo original y registrar URL, fecha de descarga y SHA-256.
- Validar fechas ordenadas, duplicados, días faltantes, flags y conversiones de unidades.
- Distinguir cero caudal de falta de observación; nunca convertir automáticamente faltantes a cero.
- Usar particiones por fechas guardadas en un manifiesto, antes de ajustar normalizadores, grupos o modelos.
- Ajustar estadísticas únicamente con entrenamiento; seleccionar opciones con validación y abrir prueba al terminar las decisiones.

**Publicado:** 70/15/15. **Propuesto:** cortes cronológicos deterministas por fecha. **Pendiente:** extremos exactos y redondeo del paper. Si la serie estuviera completa del 2002-01-01 al 2005-12-31, tendría 1461 días, pero no se debe fijar ese supuesto en el cargador.

Se permite que las entradas de una predicción de prueba incluyan observaciones anteriores ya disponibles, incluso del final de validación. Lo que no se permite es entrenar con objetivos de validación/prueba ni cruzar sus límites con las etiquetas de una ventana de entrenamiento. Guardar origen y fecha objetivo evita errores de desplazamiento.

## 2. Dos modos de investigación claramente identificados

### Reconstrucción del artículo

Requiere datos originales, versión CEEMDAN, grupos y protocolo de ventanas de los autores. Se pueden conservar los grupos de la tabla 3 para esos datos, pero solo después de confirmar que los IMFs tienen el mismo significado y número. Si se recibe un procedimiento que transforma toda la serie, documentar su carácter retrospectivo y no reportarlo como habilidad operativa.

### Evaluación causal propuesta

Para cada origen `t`, la entrada debe ser función únicamente de `q[:t+1]` y de información ya disponible. CEEMDAN no ofrece un `fit/transform` convencional que permita transformar el futuro como lo haría un escalador. Se necesita un procedimiento por orígenes:

1. Tomar un prefijo o una ventana histórica larga terminada en `t`.
2. Descomponer exclusivamente ese historial.
3. Aplicar una política de agrupación y alineación aprendida/definida con entrenamiento.
4. Extraer los **últimos siete pasos** de esos componentes como entrada neuronal.
5. Emitir H pasos sin caudales futuros en el encoder, el decoder o la descomposición.

**No descomponer solamente siete puntos esperando recuperar los nueve IMFs del paper.** El historial para descomposición y la ventana que recibe la red son parámetros distintos. Proponemos probar longitud de descomposición fija frente a prefijo creciente, seleccionando exclusivamente en validación y midiendo bordes y coste.

**Dificultad pendiente:** los componentes pueden cambiar al ampliar el historial; IMF3 en un origen no garantiza identidad estable con IMF3 en otro. El adaptador deberá alinear por escala/regularidad con reglas verificables o usar grupos agregados de dimensión fija. Una agrupación binaria global de todos los componentes altos/bajos es una variante nueva; no reproduce automáticamente la tabla 3.

Para entrenamiento supervisado por componente, las etiquetas pueden construirse con información del período objetivo de entrenamiento, pero **las entradas deben conservar la descomposición calculada hasta t**. Nunca derivar X y Y de una única descomposición que incluya t+H. Hay que documentar la alineación entre componentes de entrada y etiqueta. Una alternativa es entrenar contra caudal total con pérdidas auxiliares; eso también es una modificación y pierde identificabilidad de las contribuciones sin restricciones adicionales.

## 3. Interfaz de descomposición y MRS

```text
decompose(history, backend, noise_definition, trials, seed)
    -> components[K,T], residual[T], metadata

sample_entropy(series, m, tolerance, distance, count_convention)
    -> entropy, diagnostics

group_components(components, residual, train_policy)
    -> rimfs[G,T], group_members, entropies, high_low_routes
```

**Publicado:** 100 realizaciones, ruido 0.2; m=2, tolerancia 0.2×SD de cada IMF. **Propuesto:** semilla independiente de entrenamiento, ejecución determinista de referencia y guardar resultados/cache por checksum del historial y configuración. **Pendiente:** backend, interpolación EMD, criterios de parada, residuo, normalización del ruido, `ddof` y desigualdad de coincidencia SampEn.

La reconstrucción debe cumplir `q ≈ sum(IMFs)+residuo ≈ sum(RIMFs)`. Ningún componente se elimina ni duplica silenciosamente. La entropía de cada RIMF se recalcula después de sumar. Comparar con la entropía de la señal original es el criterio descrito para enrutar, pero falta la regla general para formar grupos.

Definir dos políticas separadas: `paper_explicit_groups`, restringida a los casos y descomposición verificados; y una política general propuesta, con algoritmo y tolerancias versionados. No introducir K-means o HDBSCAN y etiquetarlos como si fueran el MRS original.

## 4. Tensores y arquitectura

Usaremos `B`=batch, `L`=7, `H`∈{1,3,5,7}, `G_H/G_L`=número de componentes de cada rama.

| Etapa | Forma propuesta |
|---|---|
| Entrada alta | `[B,L,G_H]` |
| Entrada baja | `[B,L,G_L]` |
| Salidas por grupo | `[B,H,G_H]` y `[B,H,G_L]` |
| Caudal final | `[B,H]`, suma de salidas desnormalizadas |

Hay dos interpretaciones que el paper no permite resolver: un predictor independiente por RIMF o dos redes multisalida, una por rama. La figura 3 y el apéndice D sugieren dos ramas; los experimentos por componente y los tiempos de la tabla 7 no aclaran si comparten pesos. **Mantener esta elección pendiente en la configuración de réplica.** Para el primer prototipo, dos redes multisalida es una opción propuesta que deberá compararse con redes independientes, reportando parámetros y coste total.

**Informer publicado:** 256 dimensiones, 8 cabezas, 2 capas encoder, 1 decoder, feedforward 1024, GELU. Recuperar el modelo oficial y adaptar el cargador. Debe declararse `label_len` (≤7 si solo se admiten siete días históricos), `factor`, dropout, embeddings, frecuencia diaria y máscara. El decoder recibe historia conocida y ceros/placeholder para valores desconocidos; marcas de calendario futuras sí son conocidas. Variables meteorológicas futuras observadas no lo son.

**LSTM publicada:** dos capas, hidden=256. **Propuesta inicial:** estado final → capa lineal → `H×G_L`, sin iteración autoregresiva. La cabeza no está especificada en el paper. No restringir cada IMF a valores positivos: los componentes pueden ser negativos aunque el caudal no lo sea.

**Entrenamiento:** Adam, lr inicial 1e-4, batch32, 100 épocas y MSE están publicados. Proponemos un modelo por horizonte para ajustarnos a longitudes de salida distintas; registrar checkpoint por validación y semillas predefinidas. No heredar silenciosamente early stopping o scheduler del repositorio Informer. Si se usan, son decisiones adicionales.

Si hay normalización independiente por RIMF, invertirla individualmente antes de sumar. No sumar cantidades estandarizadas y aplicar el escalador del caudal: en general eso no reconstruye la misma magnitud física.

## 5. Evaluación y baselines

RMSE y NSE deben calcularse en unidades originales sobre exactamente las mismas fechas para todos los modelos de cada comparación. KGE inicial será la versión de 2009, no KGE modificado:

```text
RMSE = sqrt(mean((prediction - observation)^2))
NSE  = 1 - sum((prediction-observation)^2) / sum((observation-mean(observation))^2)
r    = Pearson(prediction, observation)
a    = std(prediction) / std(observation)
b    = mean(prediction) / mean(observation)
KGE  = 1 - sqrt((r-1)^2 + (a-1)^2 + (b-1)^2)
```

Casos constantes, medias nulas o correlación indefinida deben producir un estado explícito, no una sustitución numérica oculta. Guardar también r, a y b para depurar KGE.

El paper no explica si “Step-7” evalúa solo el séptimo día o agrega los siete. **Propuesta:** registrar cada horizonte `h=1..H`, el último horizonte como resultado principal y una agregación adicional etiquetada. No aplanar ventanas solapadas y llamar al resultado “día 7”. Guardar predicciones por `station, seed, model, origin, target_date, h`.

| Experimento | Qué responde |
|---|---|
| Persistencia `qhat(t+h)=q(t)` y referencia estacional definida con entrenamiento | ¿El modelo gana frente a pronósticos simples? |
| LSTM e Informer sobre señal original | ¿Mejora la descomposición? |
| CEEMDAN-LSTM y CEEMDAN-Informer sin reagrupación | ¿Mejora MRS? |
| MRS-LSTM y MRS-Informer | ¿Mejora mezclar modelos? |
| DMEL y asignación inversa | ¿Importa el enrutamiento? |
| Transformer, RNN, TimesNet, PatchTST | Comparación completa del paper |
| Procedimiento retrospectivo frente a causal, si se dispone del primero | ¿Cuánto depende la ventaja de información no disponible al pronosticar? |

Diez ejecuciones por condición final, con semillas guardadas. Comparaciones pareadas y corrección Holm donde proceda; intervalos temporales por bloques y múltiples períodos para variabilidad hidrológica. No confundir diez semillas con evidencia de generalización a diez cuencas. Registrar presupuesto y afinar baselines con la misma información y reglas.

En TimesNet probar explícitamente todos los horizontes con L=7 y top_k=5; en PatchTST usar patch_len=3 publicado y definir stride/padding faltantes. Fallos de compatibilidad requieren decisión documentada, no descartarlos o dejar configuraciones inoperantes.

EP en los máximos mensuales reproduce una parte del paper; añadir error de tiempo del pico, sesgo de volúmenes, caudales bajos y rendimiento por estación. No reutilizar el conjunto de prueba para elegir eventos que hagan ver mejor al modelo.

## 6. Módulos y orden de trabajo

Estructura propuesta, aún no implementada:

```text
configs/                 parámetros publicados, decisiones y variantes
data/raw/                archivos originales y manifiestos
src/dmel/data.py          validación, particiones, unidades y ventanas
src/dmel/decomposition.py CEEMDAN y política por origen
src/dmel/entropy.py       entropía y diagnósticos
src/dmel/mrs.py           grupos, alineación y rutas
src/dmel/models/          Informer, LSTM, baselines y ensamble
src/dmel/train.py         semillas, pérdidas y checkpoints
src/dmel/evaluate.py      métricas y exportación por origen/horizonte
tests/                   invariantes científicos y numéricos
artifacts/               componentes, predicciones y manifiestos de corridas
```

1. Ingesta de una estación y baselines simples; validar índices y métricas.
2. CEEMDAN/SampEn con señales sintéticas y reconstrucción.
3. MRS explícito y modelo de baja frecuencia.
4. Informer y suma del ensamble, prueba pequeña de entrenamiento.
5. Evaluación causal, ablaciones y diez semillas de las condiciones finales.
6. Extensiones multivariadas, otras cuencas y explicación de resultados.

Pruebas mínimas sustantivas: reconstrucción de señal; métrica exacta en predicción perfecta y casos degenerados; ninguna etiqueta de entrenamiento posterior al corte; cambiar valores después de t no cambia X_t ni la predicción a origen fijo; cero caudal futuro en decoder; suma tras invertir escala; correspondencia de fechas y horizontes. No hace falta entrenar todo el benchmark para comprobar estas invariantes.

## 7. Qué solicitar a los autores

Solicitar en un único paquete: series originales y códigos de estaciones; suplemento; fuentes/unidades meteorológicas; código CEEMDAN y SampEn; reglas MRS; cortes y ventanas; normalización; arquitectura por rama y cabeza LSTM; `label_len`, factor, dropout y scheduler; semillas/checkpoints; definición y código de métricas; predicciones por fecha de cada repetición; explicación de discrepancias de tablas/figuras; hardware y alcance de costes. La solicitud aún no fue enviada.

Antes de entrenar a escala, fijar entorno y dependencias con commits/lockfile, comprobar una corrida pequeña y medir el coste real de CEEMDAN por origen. Esta revisión no necesita elegir ni pagar infraestructura de cómputo.
