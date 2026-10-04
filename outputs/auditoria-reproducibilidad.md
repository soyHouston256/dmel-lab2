# Auditoría científica y de reproducibilidad de DMEL

**Fuente auditada:** Wang et al., *DMEL: A novel dual-modal ensemble learning architecture for multi-step runoff prediction*, Applied Soft Computing 192 (2026), 114792, DOI [10.1016/j.asoc.2026.114792](https://doi.org/10.1016/j.asoc.2026.114792). PDF aportado por el usuario, 30 páginas. Las referencias a páginas corresponden a su numeración impresa. Revisión del texto completo, incluidas las referencias y los apéndices A–E; inspección visual específica de las ecuaciones de p. 5, las figuras de dispersión de pp. 12–13, los diagramas estadísticos de p. 14 y la tabla rotada de p. 20. Este documento distingue **evidencia documental**, **deducción matemática**, **riesgo condicionado** y **decisión propuesta**. No constituye una reproducción experimental ni demuestra que haya ocurrido fuga de información.

## Dictamen

La idea central es implementable: descomponer caudal con CEEMDAN, reagrupar componentes por entropía muestral, asignar las componentes complejas a Informer y las suaves a LSTM, y sumar sus predicciones. El artículo proporciona un punto de partida útil, pero **no una especificación ejecutable suficiente para reproducir sus cifras**. Los bloqueos principales son el protocolo temporal de descomposición, la regla de agrupamiento MRS, el acceso a los datos exactos y la definición/agregación de métricas. Además, una comprobación algebraica encuentra una incompatibilidad importante entre ciertos valores de RMSE/NSE y la dispersión que muestra la figura de repeticiones, bajo el protocolo que el texto declara.

La primera implementación debe etiquetarse como **implementación basada en el artículo con decisiones documentadas**, hasta obtener código, datos, predicciones y aclaraciones de los autores. Debe privilegiar una evaluación causal verificable; los valores publicados sirven como referencia documental, no como una tolerancia de aceptación garantizada.

## Hallazgos prioritarios

### P0 — No se puede verificar que las entradas respeten el origen del pronóstico

**Evidencia:** §2.3, p. 5, describe primero la reconstrucción de las series históricas; §3.1 informa solo la proporción 70/15/15. §4.1 y Figs. 7–9, pp. 7–10, presentan una descomposición por estación y sus entropías. No se especifica si CEEMDAN, entropías, agrupamiento y normalización se calculan antes o después de particionar, ni si se actualizan usando exclusivamente observaciones disponibles en cada fecha.

**Riesgo condicionado:** si se descompone la serie completa antes de dividirla, las componentes usadas como entradas de entrenamiento o prueba pueden depender de observaciones futuras. CEEMDAN utiliza extremos y envolventes de la señal; recortar una descomposición calculada sobre toda la serie no equivale a descomponer solamente el prefijo disponible. Incluso descomponer por separado todo el bloque de prueba incorpora días posteriores al origen dentro del mismo bloque. No basta con ordenar cronológicamente las ventanas después.

**Decisión propuesta:** declarar las fechas de corte antes de cualquier ajuste; hacer que cada entrada al origen `t` dependa solo de datos `<= t`; congelar selección de parámetros/umbrales usando entrenamiento o validación. Si se utiliza CEEMDAN en ventanas o prefijos, explicitar la longitud, el tratamiento del borde y la correspondencia de modos entre fechas. Conservar una variante retrospectiva únicamente como diagnóstico claramente separado. Una prueba de causalidad debe cambiar los datos posteriores a `t` y verificar que las entradas/predicciones emitidas en `t` permanezcan iguales.

### P0 — MRS no está definido como un algoritmo general

**Evidencia:** §2.2.1, p. 5, pide agrupar IMF de entropía “similar”, pero no da distancia, tolerancia, algoritmo de agrupación, regla de adyacencia, número de grupos ni desempates. La tabla 3, p. 10, da las agrupaciones resultantes de dos estaciones:

| Estación | Grupos altos | Grupos bajos |
|---|---|---|
| Shuangpai | IMF1, IMF2, IMF3, IMF4, IMF5, cada una separada | IMF6+IMF7+IMF8+IMF9 |
| Fenghuang | IMF1, IMF2, IMF3+IMF4+IMF5 | IMF6+IMF7+IMF8+IMF9 |

El umbral alto/bajo sí tiene una descripción: comparar con la SE de la serie original (§2.3, p. 5). En Fenghuang, la tabla 2, p. 9, da SE original=0.4523 y SE de IMF3/4/5=0.3247/0.2056/0.1522, todas inferiores; el texto de p. 7 informa que su suma tiene SE=0.6825 y por ello se asigna al grupo alto.

**Implicación:** hay que reagrupar, recalcular la SE de cada suma y luego clasificar. Clasificar cada IMF respecto al umbral y sumar después no reproduce la tabla 3. La entropía de la suma no es la suma ni el promedio de entropías. Tampoco es una magnitud directamente equivalente a frecuencia espectral: “alto/bajo” es una interpretación del artículo, no una identidad matemática demostrada.

**Pendiente:** pedir la regla exacta; mientras tanto, ofrecer una configuración explícita de agrupaciones para estas estaciones y una política general alternativa identificada como elección propia. No presentar los índices `[1..5]`/`[6..9]` como algoritmo adaptativo universal.

### P0 — RMSE, NSE y la figura de repeticiones requieren reconciliación

**Evidencia:** tabla 4, p. 10, reporta promedios de diez corridas sobre el conjunto de prueba; Fig. 14, p. 14, muestra sus distribuciones. Para Shuangpai, DMEL tiene RMSE=126.9731 y NSE=0.9307; TimesNet, RMSE=231.0061 y NSE=0.7201; Informer, RMSE=198.5082 y NSE=0.8096.

**Deducción matemática, con supuestos explícitos:** si todas las corridas usan el mismo vector observado `y`, sin ponderación, y las ecuaciones (15)–(16), entonces para cada corrida:

`RMSE_s² = V_y (1 − NSE_s)`, donde `V_y = mean((y − mean(y))²)`.

Para promedios aritméticos de las mismas diez corridas:

`V_y = (mean(RMSE)² + Var_population(RMSE)) / (1 − mean(NSE))`.

Por ello **no** es correcto exigir igualdad exacta de `mean(RMSE)²/(1−mean(NSE))`: falta la varianza entre corridas. Sin embargo, DMEL impone `V_y >= 232643.1187`. Bajo ese mismo `V_y`, TimesNet necesita una desviación estándar poblacional de RMSE **al menos 108.4112** (CV=46.9%), e Informer al menos **69.9267** (CV=35.2%). La Fig. 14 presenta resultados mucho más concentrados; todos los RMSE del panel Step-1 se ven aproximadamente dentro de 120–260. Incluso ese intervalo global completo permitiría como máximo una desviación estándar de 70; la exigida para TimesNet supera ese límite.

**Conclusión acotada:** tabla, figura y agregación declarada no parecen compatibles simultáneamente. Esto puede deberse a una errata, conjuntos/máscaras de evaluación diferentes, agregación no descrita u otra implementación. No identifica por sí mismo la causa ni demuestra manipulación. Hay que pedir `y_true`, `y_pred`, máscara, fechas, métricas por semilla y código de evaluación. La implementación propia debe comprobar esta identidad por corrida antes de promediar.

### P1 — KGE está especificado de forma ambigua o incorrecta

**Evidencia visual:** debajo de la ecuación (17), p. 5, el PDF imprime `α = σ(q_i / q̂_i)` y `β = μ(q_i / q̂_i)`, y después atribuye a α y β las funciones de desviación estándar y media. La forma de la ecuación exterior corresponde a KGE, pero estas definiciones no fijan una implementación inequívoca. La ambigüedad está en el PDF, no solo en la extracción de texto.

**Decisión propuesta:** elegir y citar una variante precisa de KGE y registrar por separado correlación, razón de desviaciones y razón de medias. En la variante KGE original habitual, usar `α=std(pred)/std(obs)` y `β=mean(pred)/mean(obs)`, con casos degenerados definidos. No sustituirlos silenciosamente por la media/desviación de cocientes punto a punto ni invertirlos. Hasta recibir el código, no se puede asumir que las cifras publicadas usan exactamente esa variante.

### P1 — Figuras de dos estaciones distintas prácticamente duplicadas

**Evidencia visual y digital:** Fig. 11 (Shuangpai, p. 12) y Fig. 12 (Fenghuang, p. 13) presentan las mismas siete nubes de dispersión, rectas, ecuaciones, R² y escalas. Por ejemplo, M1 es `y=0.963x+0.3`, `R²=0.971`; M2 es `y=0.869x+2.0`, `R²=0.850`, en ambas figuras. El eje de caudal observado abarca aproximadamente 0–30 m³/s en ambas, pese a las escalas muy distintas de las estaciones.

Se extrajeron sin modificar los JPEG embebidos con `pdfimages`: ambos tienen 1800×1845 píxeles; los objetos PDF son 278 y 282. Los hashes de archivo y de píxeles son distintos, así que **no son archivos idénticos byte por byte**. No obstante, la diferencia absoluta media entre píxeles RGB es solo **0.2767 niveles sobre 255**, con RMSE ≈0.880 niveles; ello confirma que son visualmente prácticamente el mismo gráfico, con pequeñas diferencias compatibles con recompresión.

**Conclusión acotada:** probable duplicación editorial o sustitución incorrecta de una figura. Sin datos y figuras corregidas, estos gráficos no proporcionan evidencia independiente para ambas estaciones. Si existió una transformación de unidades o una selección de rango, tampoco está explicada en los ejes. Este hallazgo no establece la causa del error ni invalida por sí solo todos los experimentos.

### P1 — La arquitectura y el significado de “multi-step” están incompletos

**Evidencia:** ecuación (14), p. 5, muestra dos predictores; tabla 7, p. 18, cuenta seis/cuatro subsecuencias; la conclusión, p. 26, dice que los modelos se establecen por separado para las componentes. Tabla D1, p. 29, informa aproximadamente 2.377 M parámetros en ambas estaciones, pese al diferente número de RIMF. No aclara si son dos modelos multivariantes, un modelo por RIMF, parámetros compartidos, o conteos parciales.

La entrada tiene longitud 7 y las salidas longitudes 1/3/5/7 (§3.2, p. 5). El texto de p. 3 describe el decodificador generativo de Informer, pero no define la cabeza multi-salida de LSTM ni si cada longitud se entrena de forma independiente. Las tablas 4/6 presentan un número por horizonte sin decir si evalúan el último lead, todos los leads aplanados o un promedio por lead. Para St. Joe se habla de fusionar resultados mediante `cat`, p. 25, en lugar de la superposición de la ecuación (14); falta la operación que convierte la concatenación en caudal.

**Decisión propuesta:** definir un contrato tensorial y una tabla explícita de modelos, entradas y salidas. Registrar predicciones con claves `station, seed, origin_date, target_date, lead`; reportar métricas por lead y agregado definido. Para un diseño aditivo, exigir que el número final sea la suma de contribuciones de caudal después de invertir la normalización de cada componente. La elección de dos redes o redes por componente debe figurar como decisión pendiente, no inferirse del nombre “dual”.

### P1 — Faltan datos y decisiones de preparación esenciales

**Evidencia:** §3.1, p. 5, da las estaciones chinas y los años 2002–2005, pero no un archivo, versión, identificador de estación o fechas exactas de corte. La disponibilidad de p. 29 remite al autor por solicitud razonable. St. Joe (§5.6, p. 25) añade siete covariables, sin detallar la fuente de cada una, estación ni tratamiento de sus tiempos. Jemez (apéndice C, p. 28) ofrece un enlace general a USGS y 1980-01-01–2010-12-31, pero no el identificador concreto de estación ni la unidad descargada.

La estandarización Z-score descrita en p. 19 pertenece al análisis K-means; no demuestra qué normalización usan las redes. La normalización mencionada en apéndice B tampoco especifica su ajuste para el experimento principal. No hay protocolo completo de datos faltantes, duplicados, ceros, escalado por RIMF ni conversión de unidades. La ecuación (1), p. 3, incluye un residuo, pero las tablas hablan de nueve IMF sin aclarar si la novena lo incluye.

**Decisión propuesta:** conservar datos originales, unidades, zona/convención de fecha, banderas de calidad, versión de descarga y hashes. Verificar reconstrucción `q ≈ sum(componentes)+residuo`; nunca descartar el residuo por asumir que toda salida de una librería es IMF. La proporción 70/15/15 no permite reconstruir por sí sola redondeo y corte de ventanas.

### P1 — Las grandes ganancias de portada no aíslan la contribución del predictor dual

**Evidencia:** tablas 4/6 comparan DMEL con Informer/LSTM/etc.; tabla 8 añade MRS y CEEMDAN a cada modelo. Para Shuangpai, 1 día, MRS-Informer obtiene RMSE=130.5517, NSE=0.9106, KGE=0.9521, frente a DMEL=126.9731/0.9307/0.9706. La ganancia de DMEL en RMSE respecto a este comparador es **2.74%**, y en KGE **1.94%**; no el 36.04% y 22.60% respecto al Informer de tabla 4. A 7 días, frente a MRS-Informer, son **7.21%** en RMSE y **1.24%** en KGE. La tabla 10, p. 20, sí incluye la asignación invertida LSTM-alta/Informer-baja.

**Interpretación:** la descomposición/reagrupación explica gran parte de la diferencia mostrada frente a redes sobre series sin esa transformación. El texto de §5.1, p. 11, que dice que todos los comparadores usan el mismo MRS, debe entenderse con cuidado en el contexto de sus ablaciones: los valores de los modelos con MRS están claramente separados en tabla 8.

**Decisión propuesta:** comparar persistencia, promedio estacional y un modelo lineal; cada red con/sin MRS; CEEMDAN sin agrupación; DMEL; asignación invertida. Usar los mismos orígenes y una selección de hiperparámetros transparente. Longitud de entrada 7 para todos no demuestra que cada familia esté cerca de su mejor configuración; es especialmente corta para afirmaciones sobre dependencias largas.

## Otras limitaciones que afectan la interpretación

### Pruebas estadísticas y generalización

- **Evidencia:** diez ejecuciones, Wilcoxon al 5%, asteriscos agregados y boxplots, pp. 8–10 y tablas 4/6/10/14. **Falta:** semillas, valores individuales, pareamiento, p-valores exactos, tamaño de efecto, intervalos y corrección por comparaciones múltiples. Diez semillas sobre el mismo período cuantifican principalmente variabilidad de optimización; no equivalen a diez cuencas o períodos hidrológicos independientes.
- **Evidencia:** el experimento principal usa cuatro años y dos estaciones; St. Joe/Jemez aportan casos adicionales, pp. 25–28. **Límite:** el artículo describe entrenamientos/evaluaciones en otras cuencas, no una transferencia documentada de un único modelo entrenado en China a una cuenca nunca vista. No afirmar generalización cero-shot.
- **Decisión propuesta:** repetición por semillas más cortes temporales por años/eventos; intervalos por bloques temporales; reportar los tamaños de test y el universo de comparaciones. Mantener un test final intacto para seleccionar parámetros.

### El horizonte “3 a 7 días sin sacrificar precisión” necesita un criterio

Las tablas prueban que el DMEL publicado retiene mayor puntuación que esos comparadores en los horizontes evaluados. No dan un umbral operativo de pronóstico útil, una tolerancia de degradación predefinida ni evaluación diaria de todos los leads 1–7. En Shuangpai su RMSE aumenta **44.76%** de 1 a 7 días, el NSE baja 0.1052 y KGE baja 0.0947. Por tanto, la frase de los highlights no significa que la precisión de DMEL a 7 días sea igual a su precisión a 1 día. Puede interpretarse como extensión respecto a otros modelos a menor lead; esa comparación debe definirse formalmente.

### Eventos extremos: resultados útiles, alcance limitado

El criterio es máximo mensual (p. 21), y tablas 12/13, pp. 23–24, evalúan siete fechas por estación. Esto da cobertura estacional, pero un máximo de mes seco no tiene necesariamente severidad de inundación. El EP absoluto no informa signo del sesgo, desfase del pico, volumen ni falsas alarmas. Contando los mínimos tabulados, DMEL gana **16/28** casos en Shuangpai y **19/28** en Fenghuang: mayoría, con excepciones relevantes. Su EP a 7 días en Shuangpai llega a 57.62% en 2005-07-02 y 38.49% en 2005-06-21. No basta para asegurar desempeño operativo de alerta de crecidas.

El análisis LIME selecciona explícitamente picos donde DMEL tuvo precisión excepcional (p. 20); es una selección ilustrativa, no una muestra imparcial. Añadir todos los eventos definidos de antemano, sesgo de magnitud, error de fecha, volumen y métricas de excedencia.

### SHAP/LIME no establecen causalidad física ni interacción aprendida

El artículo usa SHAP/LIME para atribuir contribuciones de componentes, pp. 20–23. No informa explanador, background, perturbaciones, agregación temporal ni fidelidad del sustituto local. Estas atribuciones describen el modelo bajo sus supuestos; no demuestran procesos hidrológicos causales. Si la arquitectura final es una suma de predictores independientes, la complementariedad puede ser aditiva por construcción; afirmar interacción cruzada aprendida exige especificar dónde se mezclan las ramas. Los propios autores reconocen falta de interpretación física en p. 26.

### Sensibilidad y robustez: objetivos y causalidad de las perturbaciones

Apéndice A, pp. 26–27, aplica Sobol a la **entropía media de las IMF**, no a NSE/RMSE de la predicción. Por tanto, baja sensibilidad de esa estadística a `noise_std` no prueba baja sensibilidad del pronóstico. Declara 32 conjuntos y muestreo Saltelli sin precisar si 32 es tamaño base o total, cómo discretiza `m∈[2,5]`, ni intervalos/semillas.

Apéndice B, pp. 27–28, inyecta ruido y huecos 5/10%, interpola linealmente y repite diez veces. Falta aclarar si perturba entrenamiento, prueba, objetivos o solo entradas, y si el porcentaje de desviación estándar se aplica antes/después de normalizar. Interpolar un dato ausente en tiempo real con observaciones posteriores al origen produciría fuga; la interpolación retrospectiva debe distinguirse del escenario operativo. No se reporta una ablación con enrutamiento fijo para aislar si la mejora proviene de la adaptación MRS.

### Eficiencia: no es todavía una estimación de despliegue reproducible

Tabla D1, p. 29, publica aproximadamente 2.377 M parámetros, 0.0172 G FLOPs, 0.60–0.95 s/epoch y 2.50–2.95 ms/inferencia. No especifica hardware, librerías, precisión numérica, batch de inferencia, sincronización, calentamiento ni si el tiempo incluye CEEMDAN, entropía, agrupación, normalización y todos los submodelos. No inferir capacidad de dispositivos de borde a partir de la latencia sin esos datos. La eficiencia de eliminar componentes no prueba por sí sola reducción del error: los errores de las componentes también pueden compensarse y su covarianza importa.

## Erratas e inconsistencias documentales menores verificables

| Localización | Hallazgo | Tratamiento |
|---|---|---|
| p. 7 vs tabla 2, p. 9 | Se describe descenso gradual de SE al bajar frecuencia, pero Shuangpai pasa 0.1885→0.2226→0.1911→0.2593→0.2167; no hay monotonicidad en las primeras IMF. | Tratar la relación como tendencia general, no regla de ordenamiento. |
| p. 9 vs tabla 4, p. 10 | La mejora NSE frente a PatchTST se da como 42.30%; `(0.9307/0.7240−1)×100 = 28.55%`. El 42.30% sí corresponde aproximadamente a KGE. | Usar valores tabulados y recalcular porcentajes. |
| p. 12 vs tabla 8, p. 18 | El texto da NSE=0.7102 para CEEMDAN-Informer, Shuangpai, 7 pasos; tabla 8 da 0.7402. | Conservar ambas evidencias y preguntar cuál es correcta. |
| p. 23 vs tabla 13, p. 24 | El texto cita RNN como mejor en Fenghuang 2005-07-07 a 5 pasos; tabla 13 muestra DMEL=0.26%, menor que RNN=7.63%. | Preferir tabla para recuentos provisionales. |
| tabla C1, p. 28 | Asteriscos solo en Step-1, tanto para DMEL como Informer; otros horizontes no llevan asterisco. | No afirmar que todos los leads de Jemez tienen significancia documentada. |
| referencias [55]–[56], p. 30 | La justificación de 70/15/15 apunta a trabajos de salud/imágenes, no define validación temporal hidrológica. | Justificar el diseño temporal por disponibilidad y objetivo, no por llamarlo proporción estándar. |

La tabla 10 **sí existe** en p. 20, girada 90°; puede desaparecer en extracción `pdftotext -layout`. No debe diagnosticarse como tabla ausente.

## Información que deben resolver autores o código original

1. Archivos exactos, permisos de uso, identificadores de las cuatro estaciones, unidades, calendario, flags de calidad y fechas de partición.
2. ¿Cuándo se descompone respecto a las fechas de entrenamiento/validación/prueba y a cada origen de pronóstico? ¿Cómo se trata el borde y la actualización de modos?
3. Implementación/versión CEEMDAN, significado de ruido=0.2, 100 realizaciones, semillas, interpolación, límites de sift, criterio de parada, máximo de modos y residuo.
4. Implementación de SampEn: norma, exclusión de auto-coincidencias, tolerancia absoluta, tratamiento de ceros, longitud usada y regla exacta de agrupación. ¿Se recalculan las entropías después de cada suma?
5. ¿Dos redes, una por componente o pesos compartidos? ¿Cómo explicar el conteo de parámetros? ¿Qué representa `cat` en St. Joe?
6. `seq_len`, `label_len`, `pred_len`, `factor`, distil, embedding/timestamps, dropout, padding, configuración LSTM, cabeza de salida y normalizadores por variable/componente.
7. ¿Predicción directa multi-salida, directa por lead o recursiva? ¿Las métricas por Step-H corresponden al lead H o a todos los leads 1..H?
8. Código de KGE y métricas, observaciones/predicciones por semilla, orden de promediado, máscaras y corrección de las discrepancias RMSE/NSE.
9. Selección de hiperparámetros/checkpoint, early stopping, scheduler, diez semillas, p-valores exactos, pareamiento y corrección por multiplicidad.
10. Fuentes y disponibilidad temporal de covariables St. Joe; ruido/huecos aplicados a qué partición/variable; background de SHAP/LIME; hardware y alcance de mediciones de eficiencia.

## Condiciones de aceptación de nuestra implementación

- Reconstrucción aditiva verificada y residuo contabilizado.
- Esquema de ventanas, partición y agregación de métricas documentado, con fechas de objetivos que no crucen particiones.
- Prueba de independencia respecto a observaciones futuras para todo el preprocesamiento de entrada.
- Parámetros aprendidos/seleccionados sin tocar prueba y decisiones no publicadas señaladas en la configuración.
- Identidades de métricas verificadas por corrida; exportación de predicciones y métricas por origen/lead/semilla.
- Comparadores simples y ablaciones que aíslen CEEMDAN, reagrupación y asignación de redes.
- Resultados propios separados de las cifras del paper; ningún indicador simulado presentado como reproducción.

**Acción recomendada:** construir primero la especificación y una prueba mínima verificable del pipeline, solicitar los materiales faltantes y después ejecutar experimentos en el entorno de cómputo elegido. La revisión documental no autoriza a atribuir los resultados del artículo a una implementación todavía no ejecutada.
