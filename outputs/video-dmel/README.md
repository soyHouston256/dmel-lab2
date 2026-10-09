# Video grupal final

La edición está terminada en DaVinci Resolve, proyecto **DMEL Demo Houston**, línea de tiempo **DMEL Grupo — FINAL con cierre**. Incluye las cuatro intervenciones, la locución del cierre y las referencias; dura **17:49.84**, dentro del límite de 18 minutos.

La [PPT final](../presentacion/DMEL-UTEC-lab2.pptx) y el [manifiesto de verificación](../verification/video-final-20261008.json) se versionan en Git. Los videos, voces, fotografías, renders y proyectos de edición se conservan en disco y se entregan por separado.

## Archivos locales

Rutas relativas a la raíz del repositorio:

| Archivo | Uso |
|---|---|
| `outputs/video-dmel/grupo/DMEL-Grupo-final.mp4` | Video final para entregar, 1920×1080, 25 fps, H.264/AAC |
| `outputs/video-dmel/grupo/DMEL-Grupo-final.mov` | Exportación original de Resolve; el MP4 conserva sus flujos sin recomprimir |
| `outputs/video-dmel/grupo/DMEL-Grupo-final.drp` | Respaldo editable del proyecto |
| `outputs/video-dmel/grupo/DMEL-Grupo-final.xml` | Reconstrucción de la secuencia |
| `outputs/video-dmel/grupo/LEEME.txt` | Instrucciones locales y detalle de la edición |
| `outputs/video-dmel/grupo/revision/` | Cortes, transcripciones auxiliares y verificaciones |

El DRP **no incluye los medios**. Para reabrirlo en otro equipo, transferir también `outputs/video-dmel/grupo/media/` y `outputs/video-dmel/resolve-media/`, e indicar sus nuevas ubicaciones en Resolve. No mover los medios del equipo de edición mientras el proyecto los utilice.

## Distribución final

| Tramo | Integrante / contenido | Diapositivas |
|---|---|---|
| 00:00.00–05:35.80 | Edwin Tapia | 1–5 |
| 05:35.80–11:12.32 | Alberto Velásquez | 6–10 |
| 11:12.32–13:25.84 | José Nevado | 11–15 |
| 13:25.84–16:51.08 | Houston Ramírez | 16–19 |
| 16:51.08–17:42.84 | Houston: ablación, test y conclusiones | 20–22 |
| 17:42.84–17:49.84 | Referencias sin locución | 23–24 |

Los tiempos parten del inicio del video; Resolve inicia su contador en `01:00:00:00`. Las voces mantienen su velocidad original. La grabación de José no contiene webcam; su intervención utiliza un retrato en la plantilla. El cierre usa la voz y cámara reales de `Cierre.mp4` con cuentas de entrada e intentos repetidos recortados.

Se comprobaron 62 fragmentos visuales, 59 fragmentos de voz y cinco muestras de cámara del cierre. Son verificaciones técnicas y visuales, no una escucha humana completa. El manifiesto conserva los SHA-256 y resultados de estas comprobaciones.

Las revisiones anteriores permanecen locales. Los nombres que contienen `revision`, `pendiente` o `falta voz` no corresponden al archivo final para entregar. Un clon de GitHub contiene esta guía y la evidencia ligera, pero no descarga el material audiovisual.
