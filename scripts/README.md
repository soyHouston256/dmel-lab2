# Utilidades

Ejecutar desde la raíz del proyecto. Las herramientas de transferencia las ejecuta el usuario en su terminal.

| Script | Función y efectos |
|---|---|
| `python -m scripts.smoke` | Genera datos sintéticos y comprueba el pipeline en una salida nueva |
| `python -m scripts.benchmark_decomposition` | Mide descomposición de historias; no evalúa objetivos de test |
| `python -m scripts.plot_lstm_history` | Genera PNG/PDF del historial recibido de LSTM; requiere `requirements-analysis.txt` |
| `bash scripts/descargar_khipu.sh` | Descarga datos, caché, fuentes y corridas completas a `migration/khipu/`; verifica SHA-256 |
| `python scripts/pull_khipu.py` | Implementación del respaldo, solo biblioteca estándar de Python |
| `bash scripts/sync_khipu.sh` | Utilidad anterior: `--code` y `--data` suben a Khipu; `--pull` recupera resultados parciales |
| `python -m scripts.lab2_closure` | Auditoría de datos/checkpoints, verificación de las 27 983 filas de test y comparación emparejada de ablación |
| `bash scripts/run_lab2_closure.sh ablation` | Entrena/reanuda MRS–Informer en el equipo donde se ejecuta; otras etapas: `test-lstm` y `test-dmel`. Ver [guía](../docs/cierre-lab2.md) |

## Respaldo

```bash
# Mostrar el plan sin conexión ni escritura:
bash scripts/descargar_khipu.sh --dry-run
# Descargar: abre SSH; lo ejecuta el usuario.
bash scripts/descargar_khipu.sh
# Volver a verificar la copia local, sin SSH:
bash scripts/descargar_khipu.sh --verify-only
```

`--remote`, `--remote-dir` y `--dest` permiten cambiar origen y destino. También se admiten `KHIPU_REMOTE`, `KHIPU_DIR` y `MIGRATION_PYTHON`. Los valores por defecto corresponden al proyecto de Max en Khipu. El verificador comprueba los hashes conocidos de train/caché de este laboratorio y exige la corrida LSTM completa; no es un descargador genérico.

El inventario incluye solo corridas completas; las omitidas quedan registradas. Los logs son diagnóstico mutable y no tienen la misma garantía de verificación. `sync_khipu.sh --pull` excluye `last.pt` y las cachés: para trasladar todo usar `descargar_khipu.sh`.

## Regenerar la figura de LSTM

```bash
python -m scripts.plot_lstm_history --status outputs/khipu-lstm-53987-completed/status.json --out runs/figures/lstm-learning
```

La entrada predeterminada es el historial recibido de Khipu, que tiene una clave `history`. La figura nueva se guarda fuera de los entregables originales. El benchmark también permite `--out`; usar una salida nueva para conservar la evidencia histórica.
