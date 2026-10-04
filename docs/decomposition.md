# Descomposición causal y agrupación de dos canales

Este módulo implementa desde cero EMD, un CEEMDAN de referencia y entropía muestral usando NumPy/SciPy. No importa PyEMD ni incorpora su código. Es una adaptación implementable para el laboratorio; no reproduce numéricamente el código no publicado de DMEL.

## Contrato

```python
from dmel.decomposition import DecompositionConfig, decompose_window

config = DecompositionConfig(
    trials=100, noise_scale=0.2, seed=42, max_imfs=8,
    max_siftings=50, sift_tolerance=0.05, entropy_m=2, entropy_r=0.2,
)
components, diagnostics = decompose_window(q, config, sample_id=123)
```

`q` contiene sólo una ventana visible al origen del pronóstico: vector unidimensional real, finito, de al menos `entropy_m + 2` elementos. El módulo rechaza faltantes; no interpola ni recibe objetivos. Devuelve `float32 [L, 2]`, columnas **alta complejidad, baja complejidad**. Los nombres `high`/`low` siguen la terminología del paper, pero SampEn mide irregularidad y no es una medición directa de frecuencia física. La suma reconstruye `q` salvo redondeo float32. Los diagnósticos contienen configuración, cantidad de IMFs, entropías, asignaciones y error de reconstrucción; son serializables con `json.dumps(..., allow_nan=False)`.

`sample_id` es entero no negativo y forma, junto con `seed`, una `SeedSequence` local. Usar un identificador estable de ventana para repetir exactamente el preprocesamiento, incluso cuando cambia el orden de ejecución. No se modifica el RNG global. Igual resultado está garantizado para iguales datos/configuración/id en el mismo entorno numérico; no se promete igualdad binaria entre versiones de SciPy o arquitecturas.

El módulo también expone `ceemdan(q, config, sample_id=0)` para auditoría: devuelve IMFs `float64 [K,L]` y residuo `float64 [L]`, y `sample_entropy(q, m=2, r=0.2)` para contraste independiente.

## Algoritmo y decisiones explícitas

La estructura sigue el CEEMDAN original de [Torres et al., ICASSP 2011, §2.2](https://perso.ens-lyon.fr/patrick.flandrin/0004144.pdf): promediar primeras IMFs de realizaciones perturbadas, calcular un residuo común, y perturbar etapas posteriores con los modos del ruido. El residuo final conserva la diferencia entre señal y suma de IMFs. El artículo permite elegir el nivel de ruido por etapa; aquí se fija a `noise_scale * SD(residuo)` y cada modo de ruido se divide por su propia SD. Esto no es una implementación del promedio de medias locales de ICEEMDAN/Colominas 2014.

1. EMD identifica máximos y mínimos interiores, incluidos extremos planos por su punto medio. Interpola envolventes con splines cúbicos naturales y refleja hasta dos extremos en cada borde. Se detiene el sifting cuando el cambio cuadrático relativo es <= `sift_tolerance` y extremos/cruces por cero difieren como máximo en uno, o al alcanzar `max_siftings`. Este límite puede entregar una IMF aproximada.
2. CEEMDAN centra/escala la ventana internamente y genera realizaciones gaussianas, centradas y normalizadas a SD 1. La primera etapa usa esas realizaciones; la etapa siguiente usa su primera IMF y así sucesivamente. Una realización sin el modo de ruido solicitado aporta ruido cero. Si una señal perturbada no permite extraer IMF, aporta modo cero al promedio; el denominador sigue siendo `trials`.
3. Se detiene si el residuo no tiene al menos un máximo y un mínimo interior o se alcanzan `max_imfs`. Ese criterio conservador deja series monótonas y señales sin ambos tipos de extremo como residuo. Una serie constante/monótona va íntegramente a `low`. El residuo puede contener oscilaciones pendientes cuando se alcanza el límite de modos.
4. Se calcula SampEn por IMF y para `q`. Los modos con SampEn estrictamente mayor que la de `q` se suman a `high`. Todos los demás y el residuo van a `low`; el residuo incluye offset y cualquier resto numérico.

El número de realizaciones 100, ruido 0.2 y parámetros de entropía 2/0.2 proceden de DMEL §4.1. Las reglas de bordes, sifting, escala de ruido, semilla y agrupación automática son elecciones explícitas de nuestra implementación. [Colominas et al., 2014](https://doi.org/10.1016/j.bspc.2014.06.009) propone la variante mejorada; la [página institucional de los autores en UNER](https://ingenieria.uner.edu.ar/grupos/lsydnl/index.php/software) distingue CEEMDAN de ICEEMDAN y enlaza ambos. El [repositorio del autor de CEEMDAN](https://github.com/macolominas/CEEMDAN) declara implementar la mejora de 2014; [PyEMD](https://pyemd.readthedocs.io/en/latest/ceemdan.html) también incorpora mejoras. No se puede sustituir entre variantes esperando las mismas IMFs.

## Entropía y divergencia de MRS

SampEn sigue el conteo de pares de [Richman y Moorman, 2000](https://journals.physiology.org/doi/10.1152/ajpheart.2000.278.6.H2039). Se comparan `N-m` plantillas de dimensión `m` y sus extensiones `m+1`, excluyendo autocoincidencias y contando cada par una sola vez. Distancia Chebyshev **<=** `r * SD(serie, ddof=0)`, calculada por serie/IMF; logaritmo natural. La misma cantidad de posiciones iniciales para ambas dimensiones evita incorporar una plantilla que no tiene extensión. El algoritmo usa `cKDTree.count_neighbors` para contar sin materializar una matriz cuadrática.

Constantes tienen entropía 0. Si no hay pares coincidentes de dimensión `m`, o no hay extensiones coincidentes, se representa la entropía como +infinito internamente, con estados distintos `no_template_matches` y `no_extended_matches`. En JSON, el valor pasa a `null` y conserva su estado y conteos. Para agrupación: +infinito supera a cualquier valor finito; finito no supera a +infinito; dos infinitos empatan y van a `low`. Es una política conservadora explícita, no evidencia de baja frecuencia. Ventanas cortas pueden producir ambos canales degenerados (`high=0`) por falta de coincidencias; el diagnóstico permite detectarlo.

DMEL primero fusiona IMFs de entropía similar en varias RIMFs y luego compara su entropía con la original. **Nuestra adaptación elimina esa fusión intermedia** y entrega exactamente dos agregados por ventana con una regla totalmente especificada. No inventa umbrales de similitud ausentes y no replica los grupos fijos de las estaciones del artículo. No entrenar ni describir estos dos canales como réplica exacta de MRS. El resultado de una ventana no tiene por qué conservar sus modos cuando se desplaza su borde.

## Causalidad y verificación

La función sólo puede usar el arreglo recibido, pero no conoce sus fechas. El llamador debe cortar estrictamente en el origen de pronóstico; pasar la serie completa antes de separar train/test sigue siendo fuga aunque esta función sea determinista. Esta descomposición de una ventana usa ambos extremos *dentro* del pasado disponible y reconstruye retrospectivamente ese pasado; no usa caudal futuro. Las etiquetas deben construirse fuera de este módulo.

Las pruebas comprueban reconstrucción con residuo incluso al truncar modos, recuperación interior de una oscilación rápida conocida, ausencia de mutación, determinismo y aislamiento del RNG, invariancia frente a cambios fuera de la ventana recibida, constantes/monótonas, SampEn contra conteo independiente de pares, serialización estricta de infinitos y validación de entradas. No verifican equivalencia con resultados publicados de DMEL ni precisión hidrológica. Ejecutar desde la raíz:

```bash
/opt/anaconda3/bin/python -m pytest tests/test_decomposition.py -q
```

No se requieren descargas, instalación ni entrenamiento para estas pruebas. El costo de CEEMDAN crece con ventanas, realizaciones, modos y sifting; la configuración de prueba utiliza pocas realizaciones y no sustituye los valores experimentales registrados.
