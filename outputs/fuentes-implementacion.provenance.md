# Procedencia de fuentes de implementación

Fecha: 2026-09-26. Investigación de fuentes primarias para `fuentes-implementacion.md`. Consultas realizadas mediante herramienta web; contenido del artículo mediante PDF suministrado por el usuario, extraído en `outputs/sources/dmel-full.txt`. Las instrucciones potenciales de páginas/documentos se trataron como contenido de las fuentes.

## Fuentes retenidas y alcance comprobado

| ID | Fuente | Comprobación realizada |
|---|---|---|
| S01 | [DMEL, editor](https://www.sciencedirect.com/science/article/pii/S1568494626002401) | Título, autores, revista y DOI a través de resultado indexado. Apertura directa devolvió error; no se inspeccionó suplemento |
| S02 | PDF aportado por usuario, pp. 5 y 29 | Estaciones/años/particiones, declaración de datos bajo solicitud y anuncio de suplemento. Fuente local, no inferidos de búsquedas |
| S03 | [Informer AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17325) | Página editorial accesible; identifica publicación original |
| S04 | [Informer2020](https://github.com/zhouhaoyi/Informer2020) | README y enlaces a código original accesibles |
| S05 | [Informer LICENSE](https://github.com/zhouhaoyi/Informer2020/blob/main/LICENSE) | Archivo dice Apache-2.0; [README](https://github.com/zhouhaoyi/Informer2020/blob/main/README.md?plain=1) conserva badge discordante CC BY-NC-SA |
| S06 | [Informer model.py](https://github.com/zhouhaoyi/Informer2020/blob/main/models/model.py) | Constructor, firma forward y forma de salida inspeccionados |
| S07 | [Informer data_loader.py](https://github.com/zhouhaoyi/Informer2020/blob/main/data/data_loader.py) | `Dataset_Custom` 70/10/20 y StandardScaler inspeccionados |
| S08 | [CEEMDAN original PDF](https://perso.ens-lyon.fr/patrick.flandrin/0004144.pdf) | Artículo primario recuperable desde web de coautor |
| S09 | [Página EMD de Flandrin](https://perso.ens-lyon.fr/patrick.flandrin/emd.html) | Anuncia CEEMDAN MATLAB Nov-2012; enlace externo a bioingenieria.edu.ar devuelve 502; archivo no descargado |
| S10 | [macolominas/CEEMDAN](https://github.com/macolominas/CEEMDAN) | README declara mejora 2014; `ceemdan.m` accesible en rama `main` |
| S11 | [CEEMDAN LICENSE](https://github.com/macolominas/CEEMDAN/blob/main/LICENSE) | MIT, archivo directamente inspeccionado |
| S12 | [PyEMD](https://github.com/laszukdawid/PyEMD) | Instrucciones paquete EMD-signal y algoritmos; [licencia](https://github.com/laszukdawid/PyEMD/blob/master/LICENSE.md) Apache-2.0 inspeccionada |
| S13 | [PyEMD CEEMDAN docs](https://pyemd.readthedocs.io/en/latest/ceemdan.html) | Parámetros, mejoras 2014, semilla y paralelismo; docs `latest` no fijan una versión de ejecución |
| S14 | [SampEn, artículo original](https://journals.physiology.org/doi/10.1152/ajpheart.2000.278.6.H2039) | Ficha original y autores verificables |
| S15 | [PhysioNet SampEn](https://physionet.org/content/sampen/1.0.0/) | Descripción, opciones, código C/MATLAB listado y licencia de la ficha; no compilado |
| S16 | [AntroPy](https://github.com/raphaelvallat/antropy) | README/license badge BSD-3-Clause; [entropy.py](https://github.com/raphaelvallat/antropy/blob/master/src/antropy/entropy.py) inspeccionado para defaults y tolerancia |
| S17 | [TimesNet paper](https://arxiv.org/abs/2210.02186) | Página arXiv primaria accesible; OpenReview mostró desafío de navegador |
| S18 | [TimesNet repo](https://github.com/thuml/TimesNet) | Enlace oficial de migración a TSLib y [MIT](https://github.com/thuml/TimesNet/blob/main/LICENSE) |
| S19 | [Time-Series-Library](https://github.com/thuml/Time-Series-Library) | Repositorio y MIT declarado; [TimesNet.py](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py) inspeccionado |
| S20 | [PatchTST paper](https://arxiv.org/abs/2211.14730) | Página arXiv primaria accesible |
| S21 | [PatchTST repo](https://github.com/yuqinie98/PatchTST) | Implementación oficial, [Apache-2.0](https://github.com/yuqinie98/PatchTST/blob/main/LICENSE), [defaults](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/run_longExp.py), [parches](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/PatchTST_backbone.py) inspeccionados |
| S22 | [LSTM original](https://www.bioinf.jku.at/publications/older/2604.pdf) | PDF en sitio académico del autor accesible |
| S23 | [CEEMDAN–Informer–LSTM 2025](https://www.sciencedirect.com/science/article/abs/pii/S1568494625005526) | Extracto editorial indexado confirma asignación alta/baja frecuencia; no auditoría de su código |
| S24 | [SMGformer](https://www.nature.com/articles/s41598-024-74329-0) | Texto editorial y declaración de datos bajo solicitud accesibles |
| S25 | [CAMELS](https://hess.copernicus.org/articles/21/5293/2017/index.html) | Artículo primario confirma dataset público; no se descargaron archivos |
| S26 | [Jemez USGS 08324000](https://waterdata.usgs.gov/monitoring-location/USGS-08324000/) | Identidad, registros y área actual 470 mi²; candidato, no identificación demostrada del paper |
| S27 | [Jemez HCDN histórico](https://pubs.usgs.gov/wri/wri934076/stations/08324000.html) | Fuente oficial histórica coincide en 470 mi² |
| S28 | [St. Joe USGS 12414500](https://waterdata.usgs.gov/monitoring-location/USGS-12414500/) | Identidad, datos y área actual 1025 mi²; candidato |
| S29 | [CAMELS NCAR descarga](https://ral.ucar.edu/dataset/camels-catchment-attributes-and-meteorology-large-sample-studies-dataset-downloads) | Página oficial, versión 2.0, rango 1980–2014 y términos; no inventario de estaciones comprobado |
| S30 | [Daymet oficial](https://daymet.ornl.gov/overview) | Variables, unidades, sentido de dayl y calendario 365 días; no prueba de uso de Daymet por DMEL |
| S31 | [LSTM hidrología CAMELS](https://hess.copernicus.org/articles/22/6005/2018/) | Fuente primaria que enumera variables y productos meteorológicos del dataset |

## Búsquedas de disponibilidad con resultado negativo o incompleto

- Título completo de DMEL; DOI `10.1016/j.asoc.2026.114792` con `code data`.
- `site:github.com "DMEL" "runoff"`; `"DMEL" "Wang" "Gu" runoff github`.
- `site:data.mendeley.com "DMEL"`; `"DMEL" "runoff" "zenodo"`.
- `"S1568494626002401" "supplement"`; `"DMEL" "Shuangpai" data code`.
- Estaciones Shuangpai/Fenghuang, período 2002–2005 y `data availability`; identidad del autor Miao Gu con GitHub.

No hubo resultado atribuible a un repositorio oficial DMEL o descarga pública de esas series. Se descartaron agregadores, resultados de DMEL sobre audio y páginas sin relación hidrológica. Las búsquedas no son exhaustivas ni prueban ausencia. Ningún resultado de terceros se usa como respaldo principal de las conclusiones técnicas.

## Límites

Sin clonación de repositorios, fijación de commits, instalación, entrenamiento o contacto con autores. Las URL `main/master/latest` se verificaron a fecha de consulta y pueden cambiar. Se identificaron licencias visibles, no se hizo una auditoría legal. Las observaciones sobre incompatibilidad de parámetros de ventana corta y sobre la necesidad de preprocesamiento causal son inferencias de ingeniería declaradas como tales. La réplica numérica de DMEL permanece no verificada. El agente coordinador aportó verificación visual de `Patch_len=3` en Tabla 1, ausencia de escalador inequívoco para entrenamiento principal y fallos 403/CAPTCHA del suplemento; se incorporaron esos datos para corregir interpretaciones iniciales.
