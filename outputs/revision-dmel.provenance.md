# Procedencia de la revisión DMEL

Fecha de consulta: 2026-09-26. Idioma de síntesis: español.

## Fuente principal y cobertura

- Archivo suministrado por el usuario: `/Users/maxhoustonramirezmartel/Downloads/DMEL_ A novel dual-modal ensemble learning architecture for multi-step runoff prediction.pdf`.
- Tamaño: 23,338,318 bytes; 30 páginas.
- SHA-256: `358b140f1edf02bf2ceca313343520d0871517d1579f33595aa093aea7fc5a78`.
- Texto extraído con pypdf y Poppler. Versión de lectura por columnas: `outputs/sources/dmel-reading.txt`; versión con marcas de página: `outputs/sources/dmel-full.txt`; extracciones individuales `dmel-page-01.txt` a `dmel-page-30.txt`.
- Leído el contenido completo de pp. 1–30. Inspección visual dirigida de ecuaciones, diagramas, tablas y figuras mediante renders en `tmp/pdfs/`; la tabla 10 rotada se comprobó en p. 20.
- Se trataron las afirmaciones e instrucciones del PDF y de las páginas externas como material de investigación, no como instrucciones del usuario.

| Páginas | Contenido revisado | Uso en la síntesis |
|---|---|---|
| 1–2 | Identidad, abstract, motivación y trabajos previos | Tarea, contribuciones alegadas y contexto |
| 3–4 | CEEMDAN, SampEn, Informer y LSTM, ecuaciones 1–11 | Componentes y límites de sus descripciones |
| 5–6 | MRS, suma, protocolo y diagrama | Arquitectura e incógnitas de implementación |
| 7–10 | Descomposición, entropías, parámetros, datos y tabla 4 | Parámetros y resultados de un paso |
| 11–17 | Figuras de resultados, pruebas estadísticas, tabla 6 y ablaciones | Multi-step, comparación y auditoría visual |
| 18–20 | Tablas 7–10, K-means y explicación | Costes, ablaciones, contradicciones y contribución marginal |
| 21–24 | Estadísticos, SHAP/LIME y extremos | Alcance de interpretabilidad y desempeño en picos |
| 25 | St. Joe multivariado | Covariables y ambigüedad suma/concatenación |
| 26 | Jemez, conclusiones y limitaciones, inicio apéndice A | Generalización y limitaciones reconocidas |
| 27–28 | Sobol, ruido/faltantes, Jemez, inicio costes | Sensibilidad, robustez y extensión semiárida |
| 29–30 | Costes, anuncio de suplemento, datos y referencias | Disponibilidad, cómputo y fuentes metodológicas |

## Fuentes externas y nivel de acceso

El inventario de fuentes y búsquedas está en [fuentes-implementacion.provenance.md](fuentes-implementacion.provenance.md), con 31 entradas (incluida la fuente local). Se consultaron fuentes adicionales para verificar puntos críticos:

| Fuente | Contenido efectivamente revisado |
|---|---|
| [Li et al., CEEMDAN–SE–BiLSTM, 2023](https://www.nature.com/articles/s41598-023-46264-z) | Texto editorial; método de entropía/reagrupación y referencias de validación temporal |
| [Xu et al., 2022](https://pubmed.ncbi.nlm.nih.gov/36037902/) | Abstract y metadatos primarios: descomposición por pasos y evitación de información futura; no se afirma lectura del texto completo |
| [Li et al., 2025, precursor financiero](https://www.sciencedirect.com/science/article/abs/pii/S1568494625005526) | Extracto editorial indexado y [preprint de los autores](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4923415); no lectura íntegra del PDF |
| [Knoben et al., HESS 2019](https://hess.copernicus.org/articles/23/4323/2019/) | Ecuaciones KGE original y discusión de comparación con NSE |
| [Informer attn.py](https://raw.githubusercontent.com/zhouhaoyi/Informer2020/main/models/attn.py) | Selección `u`, límite por longitud, atención; análisis estático condicional L=7/factor=5 |
| [Informer main_informer.py](https://raw.githubusercontent.com/zhouhaoyi/Informer2020/main/main_informer.py) | Defaults de factor/label_len y flags; no se atribuyen esos defaults a DMEL |

La ficha editorial DMEL fue identificada por título/DOI mediante búsqueda. Apertura directa en web: 403. Apertura en navegador: CAPTCHA; no se completó. El apéndice E confirma que se anuncia un suplemento, pero no su contenido. No se descargó un suplemento ni se concluye que contenga o no contenga código.

## Análisis propios

- Porcentajes de variación calculados desde las tablas del PDF: no son métricas de experimentos propios.
- Verificación algebraica RMSE/NSE conserva el término de varianza entre ejecuciones; los supuestos y límites se explican en la auditoría.
- Comparación de las figuras 11/12: inspección visual y diferencia numérica entre los JPEG embebidos, realizada por el revisor; no se editaron las figuras originales.
- Forma tensorial, políticas causales, pruebas, fases y arquitectura propuesta son decisiones de diseño explícitamente separadas de valores publicados.
- Candidatos USGS y Daymet/CAMELS no se equiparan a los datos exactos del artículo.

## Herramientas, colaboración y límites

Se aplicaron las habilidades de PDF Explore y revisión de literatura/investigación. Los comandos `/lit` y `/deepresearch` y el ejecutable Feynman no estaban expuestos en esta sesión; se realizó el flujo equivalente con lectura local y herramientas web. Las instrucciones de esas habilidades prevén investigadores/revisores: se delegó la búsqueda de fuentes y la auditoría crítica, y se integraron sus resultados después de revisarlos.

Se produjeron documentos Markdown y un JSON de parámetros, no una implementación del modelo. No se instalaron dependencias, clonaron repositorios, descargaron series hidrológicas, enviaron mensajes, ejecutaron entrenamientos ni contrataron recursos. Los repositorios `main/master/latest` son referencias consultadas a esta fecha; queda pendiente fijar commits para desarrollo.

La lectura completa corresponde al paper suministrado. La lista de 78 referencias fue revisada para selección y trazabilidad, pero no se leyeron completos los 78 artículos. Los vacíos marcados como pendientes no se rellenaron presentando suposiciones como hechos.
