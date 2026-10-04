# Preparar la publicación en GitHub

Destino previsto: repositorio **privado** `dmel-lab2` en la cuenta del propietario. La creación, el commit y el push quedan a cargo del usuario.

## Contenido del repositorio

| Se versiona | Se mantiene fuera de Git |
|---|---|
| Código, tests y configuraciones | `.venv/`, paquetes y cachés Python |
| Guías y contratos del dataset | HDF5, CSV del dataset y tensores |
| Métricas JSON y evidencia seleccionada | Checkpoints `*.pt`, predicciones completas y `runs/` |
| Figuras, PowerPoint y guiones | `migration/`, `cache/`, `logs/` |
| Workflow de pruebas CPU | Credenciales, ajustes locales y temporales |
| Informes propios y referencias enlazadas | PDFs originales del artículo/laboratorio y texto extraído |

Las reglas están en [`.gitignore`](../.gitignore). Los binarios de entrenamiento quedan excluidos; no se necesita Git LFS para el contenido seleccionado. El respaldo se conserva en disco y se transfiere por separado siguiendo [la guía de Linux](migracion-workstation.md).

Los informes históricos conservan nombres, fechas y rutas originales como evidencia. Las entradas vigentes son el [README principal](../README.md), el [índice de documentación](README.md) y el [catálogo de resultados](../outputs/README.md). No se asignó una licencia nueva al código ni a los datos.

## Comandos para el propietario

Crear en GitHub el repositorio privado vacío `dmel-lab2`, sin README inicial ni otra plantilla. Desde esta carpeta, si todavía no existe un repositorio Git:

```bash
git init -b main
git status --short --ignored
git add --dry-run .
git add .
git diff --cached --stat
git diff --cached --check
```

Comprobar la lista preparada: no debe incluir HDF5, checkpoints, `migration/`, `.venv/` ni el CSV de objetivos. `.gitignore` no retira archivos que ya estuvieran versionados en un repositorio anterior.

Después de revisar, sustituir `TU_USUARIO` por la cuenta real y ejecutar:

```bash
git commit -m "Prepara DMEL Lab 2: fuentes, resultados y presentación"
git remote add origin git@github.com:TU_USUARIO/dmel-lab2.git
git push -u origin main
```

Si ya existe `origin`, consultar `git remote -v` antes de agregarlo. Estos comandos se entregan para ejecución manual; la preparación local no crea commits ni publica archivos.

En este Mac, si `git` invoca la instalación de Xcode y pide aceptar su licencia, existe una instalación funcional en `/opt/homebrew/bin/git`. Se puede usar ese ejecutable para las mismas órdenes.

## Comprobaciones del proyecto

```bash
python -m pytest tests -q
python -S -m dmel.cli train --help
```

La verificación de preparación se registra en [`outputs/verification/`](../outputs/verification/README.md). El workflow de GitHub queda preparado, pero su primer resultado remoto solo existirá después de publicar el repositorio.
