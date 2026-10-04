"""Ejecutar con python -m dmel.cli --help."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="DMEL adaptado al Laboratorio 2")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("inspect", help="Comprobar esquema y checksum HDF5")
    p.add_argument("--data", required=True)
    commands.add_parser("environment", help="Versiones y disponibilidad CUDA")
    p = commands.add_parser("precompute", help="CEEMDAN/MRS; guarda y reanuda cache parcial")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/decomposition.json")
    p.add_argument("--split", choices=["all", "train", "validation", "test"], default="all")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-seconds", type=float, default=0,
                   help="Presupuesto suave por ciclo; 0 sin límite. Salida 75 si queda cache parcial")
    p.add_argument("--selection-config", help="Cache solo las filas train/val del config de smoke; usar limit=0")
    p = commands.add_parser("train")
    p.add_argument("--data", default="dataset/train.h5")
    p.add_argument("--config", default="configs/dmel.json")
    p.add_argument("--out", required=True)
    p.add_argument("--cache")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--epochs-per-cycle", type=int, default=0,
                   help="Pausa tras N épocas completas; 0 sin pausa. Salida 75 si queda entrenamiento")
    for name in ("evaluate", "predict"):
        p = commands.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--run", required=True)
        p.add_argument("--cache")
        p.add_argument("--out")
        p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p = commands.add_parser("persistence")
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    # --help y los errores de argumentos deben funcionar sin cargar PyTorch,
    # HDF5 ni SciPy. Importar dependencias solo al ejecutar el subcomando.
    if args.command == "environment":
        from .engine import environment
        result = environment()
    elif args.command == "inspect":
        from .data import schema, sha256
        result = {**schema(args.data), "sha256": sha256(args.data)}
    elif args.command == "precompute":
        from .decomposition import DecompositionConfig
        from .precompute import precompute
        cfg = DecompositionConfig(**json.loads(Path(args.config).read_text()))
        result = precompute(args.data, args.out, cfg, args.split, args.limit, args.workers,
                            args.selection_config, args.max_seconds)
    elif args.command == "train":
        from .engine import train
        result = train(args.data, args.config, args.out, args.cache, args.device, args.resume,
                       args.epochs_per_cycle)
    elif args.command == "persistence":
        from .engine import evaluate
        result = evaluate(args.data, None, split=args.split, output=args.out, persistence=True)
    else:
        from .engine import evaluate
        result = evaluate(args.data, args.run, args.cache, args.device,
                          "test" if args.command == "predict" else "validation", args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if args.command in ("precompute", "train") and not result["complete"]:
        raise SystemExit(75)  # Pausa limpia; nunca confundir un resultado parcial con éxito.


if __name__ == "__main__":
    main()
