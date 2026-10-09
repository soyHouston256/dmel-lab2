# Preparar el siguiente push

El repositorio local ya está inicializado en la rama `main`. El remoto configurado es [soyHouston256/dmel-lab2](https://github.com/soyHouston256/dmel-lab2). No hace falta repetir `git init` ni agregar otro `origin`.

## Contenido del repositorio

| Se versiona | Se conserva solo localmente |
|---|---|
| Código, tests y configuraciones originales y de tres épocas | Entornos, paquetes y cachés |
| Guías de instalación, metodología, Slurm y cierre | HDF5, CSV del dataset y tensores |
| Métricas, historiales y auditorías seleccionadas | Checkpoints, predicciones completas, `runs/` y `migration/` |
| PPT final `DMEL-UTEC-lab2.pptx`, versiones anteriores y guiones | Grabaciones, voces, fotografías de montaje y renders de Resolve |
| Guía y manifiesto de verificación del video final | MP4/MOV, DRP/DRT, XML de edición y medios de `outputs/video-dmel/` |
| Workflow de pruebas CPU e informes propios | Notas de sesión, credenciales, temporales y PDFs originales |

Las reglas están en [`.gitignore`](../.gitignore). En `outputs/video-dmel/` solo se publica su [README](../outputs/video-dmel/README.md); la evidencia del video está en `outputs/verification/video-final-20261008.json`. Los medios permanecen en sus rutas para no romper el proyecto de Resolve.

No se necesita Git LFS para la selección actual. El paper se referencia por [DOI](https://doi.org/10.1016/j.asoc.2026.114792); la copia local `DMEL.pdf` y `Laboratorio 2.pdf` quedan fuera de Git. `notes/` conserva el historial de trabajo privado del equipo y no es documentación de entrega. Los informes científicos históricos se mantienen con sus fechas y procedencia.

La [verificación del siguiente push](../outputs/verification/repository-readiness-20261008.json) registra los archivos admitidos, enlaces, exclusiones y pruebas locales. El estado remoto no se modifica durante esta preparación.

## Revisar, crear el commit y subir

Desde la raíz del proyecto:

```bash
git status --short
git diff --check
git add --dry-run .
git add .
git diff --cached --stat
git diff --cached --check
git commit -m "Completa Lab 2: ablación, test, presentación y cierre"
git push origin main
```

Revisar la lista antes del commit: no debe incluir datos, pesos, videos, proyectos de Resolve, `migration/` ni `notes/`. `.gitignore` no excluye archivos ya versionados; la comprobación local también revisa ese caso. Si el push indica que existen cambios remotos nuevos, recuperarlos y revisar la integración antes de volver a subir; no usar `--force`.

El commit y el push los ejecuta el propietario. No se publica el video ni se cargan respaldos automáticamente: transferirlos por el canal de entrega elegido, siguiendo la [guía del video](../outputs/video-dmel/README.md) y la [guía de Linux](migracion-workstation.md).

## Comprobaciones locales

```bash
python -m pytest tests -q
python -S -m dmel.cli train --help
```

Las pruebas usan datos sintéticos, sin GPU, SSH, HDF5 reales ni checkpoints. El resultado local no acredita la ejecución de GitHub Actions; verificarla después del push. Si Git del sistema solicita la licencia de Xcode, puede usarse la instalación existente `/opt/homebrew/bin/git` para estas mismas órdenes.
