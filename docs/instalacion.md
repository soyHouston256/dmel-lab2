# Instalación y verificación

Usar Python **3.12** y ejecutar todos los comandos desde la raíz del repositorio. Crear un entorno propio por equipo; no copiar `.venv` entre macOS y Linux.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

## Elegir dependencias

**macOS o entorno general:**

```bash
python -m pip install -r requirements.txt
```

**Linux, solo CPU:** instalar primero la distribución CPU de PyTorch; la segunda orden conserva esa versión compatible con el requisito `torch==2.7.1`.

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

**Linux con NVIDIA/CUDA:** el archivo siguiente reproduce las dependencias directas confirmadas en Khipu (PyTorch 2.7.1, CUDA 12.6). La GPU y el controlador de la workstation todavía deben comprobarse allí.

```bash
nvidia-smi
python -m pip install -r requirements-khipu.txt
python -m dmel.cli environment
python -c 'import torch; assert torch.cuda.is_available(); x = torch.ones((64, 64), device="cuda"); print((x @ x).sum().item(), torch.cuda.get_device_name())'
```

Si falla la comprobación CUDA, conservar el error y revisar GPU/controlador antes de entrenar. Tener 48 GB de VRAM no identifica el modelo de GPU ni su compatibilidad. Los índices CPU y CUDA 12.6 corresponden a las [distribuciones oficiales de PyTorch 2.7.1](https://pytorch.org/get-started/previous-versions/#v271).

**Gráficos opcionales**, después de instalar el entorno elegido:

```bash
python -m pip install -r requirements-analysis.txt
```

Las versiones directas están fijadas; las dependencias transitivas no están bloqueadas por completo. Cada corrida registra sus versiones en `environment.json`.

## Comprobar el código sin datos privados

```bash
python -S -m dmel.cli --help
python -m pytest tests -q
python -m scripts.smoke --out runs/synthetic-smoke
```

El `smoke` genera un dataset pequeño, calcula una caché, entrena un DMEL reducido y exporta validación y predicciones sintéticas. Si ya existe la salida, elegir un nombre nuevo, por ejemplo `runs/synthetic-smoke-02`. No contiene resultados del laboratorio.

Las pruebas cubren normalización con train, causalidad, reconstrucción, modelos, métricas, checkpoints, reanudación e inferencia. Slurm se simula localmente. La integración de descriptores con dos workers se ejecuta en plataformas que admiten `file_descriptor`; se omite en macOS.

El [workflow CPU](../.github/workflows/tests.yml) ejecutará la suite en Ubuntu/Python 3.12 al subir cambios de código y en pull requests. También admite ejecución manual. No necesita secretos, GPU ni conexión a Khipu. Usa las acciones oficiales [checkout](https://github.com/actions/checkout) y [setup-python](https://github.com/actions/setup-python). Su ejecución en GitHub queda pendiente del primer push del propietario.
