# Fuentes verificadas para implementar DMEL

Fecha de revisión: 2026-09-26. Alcance: materiales originales del artículo, implementaciones de sus componentes y obstáculos para una reproducción numérica. Las recomendaciones de adaptación que siguen son decisiones de ingeniería propuestas, no decisiones documentadas de los autores de DMEL.

## Disponibilidad de DMEL y de los datos

El artículo corresponde a Wen-chuan Wang, Miao Gu, Zong Li, Yi-yang Wang y Yan-wei Zhao, *Applied Soft Computing* 192 (2026), 114792. Se verificó la coincidencia del título y DOI con la [ficha editorial](https://www.sciencedirect.com/science/article/pii/S1568494626002401). El PDF aportado es la fuente del texto completo revisado; la ficha web accesible por búsqueda no sustituye al PDF.

El experimento utiliza caudal diario de **Shuangpai y Fenghuang, 2002–2005**, con división 70 % / 15 % / 15 % (PDF, §3.1, p. 5). La declaración de datos, p. 29, indica disponibilidad mediante solicitud razonable al autor de correspondencia. **No se ha verificado una descarga pública de esas dos series.** Tampoco se localizó un repositorio público atribuible a DMEL en búsquedas por título, DOI, autores y combinaciones con GitHub, Mendeley Data y Zenodo. Esto describe el resultado de la búsqueda; no demuestra que el código no exista.

El apéndice E, p. 29, anuncia información suplementaria en el [DOI del artículo](https://doi.org/10.1016/j.asoc.2026.114792). El contenido de ese suplemento queda **pendiente de inspección**: la herramienta accede a resultados indexados del editor, pero la apertura directa de la página falló; la comprobación complementaria del equipo encontró 403 y CAPTCHA en el navegador. No se intentó resolver el CAPTCHA. No debe afirmarse que el suplemento contiene código o datos sin inspeccionarlo. El trabajo previo del mismo grupo [SMGformer, 2024](https://www.nature.com/articles/s41598-024-74329-0) también remite los datos al autor bajo solicitud y no proporciona una vía pública alternativa confirmada.

Para una réplica numérica faltan las series originales, las fechas/índices exactos de cada partición, tratamiento de faltantes, configuraciones finales por estación y horizonte, semillas, código de descomposición y reglas completas de agrupamiento. Se puede construir y verificar el software antes de disponer de todos estos elementos, pero no presentar sus resultados como reproducción de las tablas del artículo.

## Mapa de componentes y código reutilizable

| Componente | Fuente primaria o del mantenedor verificada | Estado y uso | Licencia observada |
|---|---|---|---|
| Informer | [Artículo AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17325), [repositorio de los autores](https://github.com/zhouhaoyi/Informer2020) | Implementación original PyTorch; base para el predictor de alta frecuencia | [LICENSE Apache-2.0](https://github.com/zhouhaoyi/Informer2020/blob/main/LICENSE); el README conserva un badge CC BY-NC-SA discordante |
| LSTM | [Artículo de Hochreiter y Schmidhuber](https://www.bioinf.jku.at/publications/older/2604.pdf) | Base metodológica; la cabeza de salida multihorizonte y su entrenamiento deben especificarse en DMEL | La disponibilidad del PDF no equivale a licencia de código |
| CEEMDAN original | [Artículo de Torres et al., 2011, alojado por coautor](https://perso.ens-lyon.fr/patrick.flandrin/0004144.pdf), [página de software de Flandrin](https://perso.ens-lyon.fr/patrick.flandrin/emd.html) | La página anuncia versión MATLAB de noviembre de 2012; el destino de descarga devolvió 502 | Licencia del archivo original no comprobada |
| CEEMDAN mejorado | [Repositorio de Marcelo Colominas](https://github.com/macolominas/CEEMDAN) | MATLAB, versión mejorada de 2014; archivos `ceemdan.m` y `emd.m` | [MIT](https://github.com/macolominas/CEEMDAN/blob/main/LICENSE); revisar también cabeceras de componentes incorporados al fijar versión |
| CEEMDAN Python | [PyEMD del mantenedor](https://github.com/laszukdawid/PyEMD), [documentación CEEMDAN](https://pyemd.readthedocs.io/en/latest/ceemdan.html) | Implementación mantenida; incluye mejoras de Colominas 2014; no es código oficial de DMEL | [Apache-2.0](https://github.com/laszukdawid/PyEMD/blob/master/LICENSE.md) |
| Sample Entropy | [Richman y Moorman, 2000](https://journals.physiology.org/doi/10.1152/ajpheart.2000.278.6.H2039), [software PhysioNet](https://physionet.org/content/sampen/1.0.0/) | Referencia conceptual y código C/MATLAB para contraste numérico | La ficha PhysioNet declara Open Data Commons Attribution v1.0 para sus archivos |
| Sample Entropy Python | [AntroPy](https://github.com/raphaelvallat/antropy) | Implementación de mantenedor independiente, útil para producción y validación | BSD-3-Clause declarada en el repositorio |
| TimesNet | [Artículo de los autores](https://arxiv.org/abs/2210.02186), [repositorio oficial](https://github.com/thuml/TimesNet) | El propio repositorio remite el código completo a [Time-Series-Library](https://github.com/thuml/Time-Series-Library) | [MIT en TimesNet](https://github.com/thuml/TimesNet/blob/main/LICENSE); TSLib también declara MIT |
| PatchTST | [Artículo de los autores](https://arxiv.org/abs/2211.14730), [repositorio oficial](https://github.com/yuqinie98/PatchTST) | Usar variante supervisada para este benchmark; fijar configuración de parches | [Apache-2.0](https://github.com/yuqinie98/PatchTST/blob/main/LICENSE) |

Las licencias indicadas son las observadas en los enlaces visitados, no una auditoría de todas las dependencias transitivas. Antes de incorporar código, fijar un commit y conservar la licencia y avisos correspondientes. No se descargaron ni ejecutaron estos repositorios durante esta investigación.

## Adaptaciones que deben quedar explícitas

### Informer: reutilizar el modelo y adaptar el experimento

La [clase `Informer`](https://github.com/zhouhaoyi/Informer2020/blob/main/models/model.py) expone `seq_len`, `label_len`, `out_len`, `enc_in`, `dec_in`, `c_out`, `d_model`, capas, atención y frecuencia. Su `forward` requiere valores y marcas temporales para encoder y decoder, y devuelve los últimos pasos de predicción. Para DMEL deben fijarse los valores publicados y declarar las opciones omitidas; `label_len` no se puede adivinar como un hecho del artículo.

El [`Dataset_Custom` original](https://github.com/zhouhaoyi/Informer2020/blob/main/data/data_loader.py) divide 70 % entrenamiento y 20 % prueba, dejando el resto para validación, y usa `StandardScaler`. **No produce por defecto el 70/15/15 de DMEL.** El artículo no especifica inequívocamente el escalador del entrenamiento principal: el z-score de §5.3 pertenece al análisis K-means. Propuesta: escribir nuestro cargador con índices cronológicos guardados, documentar el escalado como elección pendiente de confirmación y ajustar el escalador únicamente en entrenamiento. Evaluar el caudal en unidades originales después de invertir la normalización correspondiente.

### CEEMDAN: no intercambiar nombres de bibliotecas con algoritmos

El repositorio de Colominas especifica que contiene la mejora de 2014 y requiere la EMD de Rilling/Flandrin; por tanto, es referencia de autor pero no identifica por sí solo la versión empleada por DMEL. [README del autor](https://github.com/macolominas/CEEMDAN).

En Python se instala **`EMD-signal`** y se importa `PyEMD`; `pyemd` en minúsculas puede referirse a otro paquete para Earth Mover's Distance. [Instalación del mantenedor](https://github.com/laszukdawid/PyEMD).

La documentación de PyEMD define `epsilon` como factor multiplicado por la desviación estándar, `trials` como número de realizaciones y declara las mejoras de 2014. Para reproducibilidad recomienda semilla mediante `noise_seed` y `parallel=False`. **No asumir que el ruido “0.2” de DMEL equivale sin más a cualquier parámetro llamado `noise_scale`.** La correspondencia debe contrastarse contra la implementación MATLAB elegida y registrarse. [Documentación CEEMDAN](https://pyemd.readthedocs.io/en/latest/ceemdan.html).

Propuesta: separar una interfaz de descomposición de su backend, guardar la variante, semilla, criterios de parada e IMFs, verificar que la suma de componentes y residuo reconstruye la entrada y probar señales sintéticas. Para un pronóstico operativo, ninguna descomposición de entrada puede utilizar observaciones posteriores al origen del pronóstico. Cambiar esa política altera el experimento y se debe informar.

### Entropía muestral: fijar convenciones numéricas

[PhysioNet](https://physionet.org/content/sampen/1.0.0/) proporciona programas C/MATLAB con dimensión y tolerancia configurables; su opción de normalización relaciona la tolerancia con la desviación estándar. Sirve como referencia independiente para cotejar resultados.

En [AntroPy, `sample_entropy`](https://github.com/raphaelvallat/antropy/blob/master/src/antropy/entropy.py), el valor por defecto es dimensión 2, métrica Chebyshev y tolerancia `0.2 * std(x, ddof=0)`. Una `tolerance` explícita se interpreta como distancia absoluta. Para DMEL, registrar si la SD se calcula sobre cada IMF, señal original o serie normalizada; distinguir el factor 0.2 de la tolerancia absoluta. También fijar manejo de series constantes, coincidencias nulas y desigualdad estricta/inclusiva. Estas decisiones pueden cambiar la agrupación MRS y el reparto alta/baja frecuencia.

### Baselines modernos con ventana de siete días

[PatchTST supervisado](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/run_longExp.py) trae `patch_len=16` y `stride=8`. **DMEL sí especifica `Patch_len=3` en Tabla 1, p. 8**, de modo que debe sobrescribirse el default 16. Su [backbone](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/PatchTST_backbone.py) construye parches con `unfold`. **Inferencia de ingeniería:** cambiar sólo `seq_len` a 7 manteniendo ambos defaults falla incluso con padding final de 8, porque 7+8<16. `stride` y padding efectivos de DMEL quedan por especificar y afectan cuántos parches procesa el baseline.

[TimesNet en TSLib](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py) elige frecuencias con FFT y `topk`, calcula períodos a partir de sus índices y proyecta la longitud temporal antes de los bloques de predicción. **Inferencia de ingeniería concreta:** con `seq_len=7`, `pred_len=1` y `top_k=5` de DMEL, la longitud proyectada es 8 y `rfft` produce 5 bins. El código pone amplitud de DC a cero, pero seleccionar los 5 bins la incluye igualmente; el cálculo `8 // 0` produce un período inválido. Se requiere una adaptación explícita que excluya DC y acote `top_k`, o el código usado por los autores. No se ejecutó este caso; la conclusión proviene de inspección estática. La adaptación cambia el baseline y debe documentarse.

## Lecturas complementarias prioritarias

1. **Li et al., 2025, CEEMDAN–Informer–LSTM.** Ya combina CEEMDAN con Informer para alta frecuencia y LSTM para baja frecuencia. DMEL lo cita como ref. 67. Esto es antecedente directo para situar la contribución de MRS y la evaluación hidrológica; no sustenta una acusación de duplicación. [Editor y DOI 10.1016/j.asoc.2025.113241](https://www.sciencedirect.com/science/article/abs/pii/S1568494625005526).
2. **SMGformer, 2024, del grupo de DMEL.** Útil para comparar decisiones de preprocesamiento y evaluación multihorizonte; no resuelve por sí mismo la falta de los datos originales. [Artículo completo](https://www.nature.com/articles/s41598-024-74329-0).
3. **CAMELS, Addor et al., 2017.** Ofrece una alternativa pública de caudal diario y atributos de 671 cuencas para validar la implementación en otras cuencas. Sería una evaluación externa, no una reproducción de Shuangpai/Fenghuang. No se descargó el dataset ni se verificaron sus archivos individuales. [Artículo y enlaces de datos](https://hess.copernicus.org/articles/21/5293/2017/index.html).

## Fuentes candidatas para las extensiones estadounidenses

El artículo también evalúa St. Joe en 2002–2005 (§5.6) y Jemez en 1980–2010 (§5.7 y Apéndice C). Se verificaron estos posibles puntos de partida, **sin identificar como hecho que sean las estaciones/archivos usados por DMEL**:

| Cuenca | Registro oficial candidato | Evidencia y limitación |
|---|---|---|
| Jemez | [USGS 08324000, Jemez River Near Jemez, NM](https://waterdata.usgs.gov/monitoring-location/USGS-08324000/) | Registro de caudal y metadatos; área oficial actual 470 mi² ≈ 1217.29 km². No coincide exactamente con 1208 km² del paper. Su [ficha histórica HCDN](https://pubs.usgs.gov/wri/wri934076/stations/08324000.html) también indica 470 mi². Coincidencia de río y escala hace plausible la candidatura, no confirma identidad |
| St. Joe | [USGS 12414500, St. Joe River at Calder, ID](https://waterdata.usgs.gov/monitoring-location/USGS-12414500/) | Identidad y caudal USGS verificables; área actual 1025 mi². El paper no aporta un código de estación que permita confirmar la correspondencia |

La [página oficial de descarga CAMELS de NCAR](https://ral.ucar.edu/dataset/camels-catchment-attributes-and-meteorology-large-sample-studies-dataset-downloads) ofrece la versión 2.0 con cobertura indicada 1980–2014, enlaza datos y condiciones de uso. Todavía hace falta inspeccionar el inventario de estaciones y los archivos originales para comprobar la inclusión de los dos candidatos, intervalos exactos y faltantes. La documentación primaria de un [estudio LSTM con CAMELS](https://hess.copernicus.org/articles/22/6005/2018/) confirma las siete variables meteorológicas y las fuentes Daymet/Maurer/NLDAS; no demuestra qué fuente usó DMEL.

Los nombres de las siete covariables de DMEL coinciden con [Daymet](https://daymet.ornl.gov/overview): `dayl` (s/día), `prcp` (mm/día), `srad` (W/m²), `swe` (kg/m²), `tmax`, `tmin` (°C), `vp` (Pa). Esto es una pista de procedencia, no prueba. `dayl` representa duración astronómica del día, no horas medidas de sol sin nubes. La versión Daymet directa usa 365 días/año: conserva el 29 de febrero y descarta el 31 de diciembre en años bisiestos. Antes de unir con caudales diarios hay que inspeccionar el calendario de la versión CAMELS/Daymet seleccionada, sin presuponer cómo se procesó. No confundir un valor de caudal en ft³/s con m³/s ni profundidad de escorrentía; el RMSE cambia con la unidad. No se descargaron series en esta revisión.

## Material mínimo antes de afirmar una réplica

- Datos originales con fechas, unidades, faltantes, identificación de estaciones y condiciones de uso.
- Suplemento del artículo inspeccionado y configuraciones completas por estación/horizonte.
- Identidad y versión de CEEMDAN/EMD, semilla, ruido, tratamiento de bordes y residuo.
- Convenciones SampEn, regla de similitud para fusionar IMFs y umbral alta/baja frecuencia.
- `label_len`, embeddings temporales, máscaras, predictor por RIMF o salidas conjuntas y política multihorizonte.
- Preprocesamiento temporal causal, escalado, selección de checkpoint y evaluación definidos y guardados.
- Commits, dependencias y licencias fijados; predicciones y métricas exportadas por fecha/horizonte y repetición.

Esta lista permite separar especificaciones recuperadas del artículo, convenciones elegidas para implementarlo y cuestiones que necesitan confirmación de sus autores. No se enviaron mensajes a los autores.
