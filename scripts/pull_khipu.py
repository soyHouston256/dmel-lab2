"""Backup verificable, ejecutado por el usuario; solo biblioteca estándar.

El inventario remoto se ejecuta por stdin: no instala ni escribe scripts en Khipu.
Las corridas incompletas se omiten; los logs se copian como diagnóstico mutable.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
KNOWN_HASHES = {
    "dataset/train.h5": "38a1e2dee5f9aa61f5e1543dc6f7c22689f6ee3059511498022df31f01730078",
    "cache/train.h5": "e8d304ca15b9f33eb5eb01bba82c32caffa004bc94bef2c183b038c57d7987e1",
}

# Se ejecuta con el Python de la .venv de Khipu, sin importar torch/h5py.
INVENTORY_CODE = r'''
import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
os.chdir(root)
required = ["dataset/train.h5", "dataset/test.h5", "dataset/metadata.json",
            "dataset/leer_datos.py", "cache/train.h5"]
core = ["best.pt", "last.pt", "config.json", "normalizer.json", "environment.json",
        "decomposition.json", "split_ids.npz", "history.json", "status.json",
        "validation/metrics.json", "validation/predictions.npz", "validation/predictions.csv"]
files, logs, included, skipped = set(required), [], [], []
def walk(folder):
    for base, dirs, names in os.walk(folder, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".pytest_cache", ".git", ".venv")
                         and not Path(base, d).is_symlink())
        for name in sorted(names):
            p = Path(base, name)
            if p.is_symlink() or p.suffix in (".pyc", ".tmp"):
                continue
            if p.is_file():
                yield p.as_posix()
for folder in ("dmel", "configs", "scripts", "slurm", "tests", "docs"):
    files.update(walk(folder))
for name in ("README.md", "requirements.txt", "requirements-khipu.txt"):
    if Path(name).is_file(): files.add(name)
for run in sorted(Path("runs").glob("*")):
    if not run.is_dir() or run.is_symlink(): continue
    status_path = run / "status.json"
    try:
        status = json.loads(status_path.read_text())
    except (OSError, ValueError):
        skipped.append({"run": run.name, "reason": "sin status.json válido"})
        continue
    if status.get("complete") is not True:
        skipped.append({"run": run.name, "reason": "incompleta; puede seguir activa",
                        "epochs_completed": status.get("epochs_completed")})
        continue
    missing = [name for name in core if not (run / name).is_file()]
    if missing:
        skipped.append({"run": run.name, "reason": "faltan artefactos", "missing": missing})
        continue
    included.append(run.name)
    files.update(walk(run))
if "lstm-seed42-fd-fix" not in included:
    raise SystemExit("No se encontró el LSTM completo con best.pt, last.pt y validación. No se descargó nada.")
records = []
for name in sorted(files):
    p = Path(name)
    if not p.is_file() or p.is_symlink() or any(parent.is_symlink() for parent in p.parents):
        raise SystemExit("Falta archivo regular o hay symlink: " + name)
    before = p.stat()
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise SystemExit("Archivo cambió durante inventario: " + name)
    records.append({"path": name, "bytes": after.st_size, "sha256": digest.hexdigest()})
logs = list(walk("logs"))
queue = subprocess.run(["squeue", "--me", "--format=%.18i %.30j %.12T %.12M %R"],
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(json.dumps({"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "remote_dir": str(root), "files": records, "logs": logs,
                  "included_runs": included, "skipped_runs": skipped,
                  "squeue": queue.stdout, "squeue_exit_code": queue.returncode}, indent=2))
'''


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or ".." in p.parts or str(p) != name
            or any(ord(c) < 32 for c in name)):
        raise ValueError(f"Ruta no permitida en inventario: {name!r}")
    return p


def safe_target(destination, name):
    rel = safe_relative(name)
    target = destination.joinpath(*rel.parts)
    if target.is_symlink() or any(p.is_symlink() for p in target.parents):
        raise ValueError(f"Destino contiene un symlink: {target}")
    return target


def validate_inventory(inventory, destination):
    if inventory.get("schema") != 1:
        raise ValueError("Versión de inventario desconocida")
    indexed = {}
    for record in inventory["files"]:
        name = record["path"]
        safe_target(destination, name)
        if name in indexed or not isinstance(record["bytes"], int) or record["bytes"] < 0:
            raise ValueError("Inventario duplicado o tamaño inválido")
        if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError("SHA-256 inválido")
        indexed[name] = record
    for name, expected in KNOWN_HASHES.items():
        if indexed.get(name, {}).get("sha256") != expected:
            raise ValueError(f"{name}: no coincide con el SHA-256 confirmado en Khipu")
    for name in inventory["logs"]:
        if not name.startswith("logs/"):
            raise ValueError("Log fuera de logs/")
        safe_target(destination, name)
    return indexed


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def verify(destination, inventory):
    validate_inventory(inventory, destination)
    marker = destination / "VERIFIED.json"
    marker.unlink(missing_ok=True)
    errors = []
    for record in inventory["files"]:
        path = safe_target(destination, record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            errors.append(record["path"])
    if errors:
        raise ValueError("Verificación fallida: " + ", ".join(errors))
    write_json(marker, {"verified_utc": datetime.now(timezone.utc).isoformat(),
                        "manifest_sha256": sha256(destination / "manifest.json"),
                        "verified_files": len(inventory["files"]),
                        "included_runs": inventory["included_runs"],
                        "skipped_runs": inventory["skipped_runs"],
                        "scope": "Archivos enumerados; logs solo diagnósticos, corridas omitidas no respaldadas"})
    print(f"VERIFICADO: {len(inventory['files'])} archivos en {destination}", flush=True)


def pull(args, destination):
    for program in ("ssh", "rsync"):
        if not shutil.which(program):
            raise ValueError(f"Instalar {program} en el equipo desde el que se descarga")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "VERIFIED.json").unlink(missing_ok=True)
    # Una conexión autenticada por el usuario, compartida por inventario y rsync.
    # /tmp mantiene la ruta del socket por debajo del límite de macOS.
    with tempfile.TemporaryDirectory(prefix="dmel-ssh-", dir="/tmp") as temp:
        socket = str(Path(temp) / "control")
        ssh = ["ssh", "-o", "ControlMaster=auto", "-o", "ControlPersist=600",
               "-o", f"ControlPath={socket}", "-o", "ConnectTimeout=20"]
        try:
            command = shlex.join([args.remote_dir + "/.venv/bin/python", "-", args.remote_dir])
            print("Leyendo inventario y hashes en Khipu; puede tardar varios minutos.\n"
                  "SSH pedirá tu contraseña en esta terminal si hace falta.", flush=True)
            result = subprocess.run(ssh + [args.remote, command], input=INVENTORY_CODE,
                                    stdout=subprocess.PIPE, text=True, check=True)
            inventory = json.loads(result.stdout)
            inventory["remote"] = args.remote
            indexed = validate_inventory(inventory, destination)
            print(inventory["squeue"], flush=True)
            print("Corridas completas: " + ", ".join(inventory["included_runs"]), flush=True)
            for run in inventory["skipped_runs"]:
                print(f"OMITIDA {run['run']}: {run['reason']}", flush=True)
            total = sum(r["bytes"] for r in indexed.values())
            # rsync sustituye archivos completos temporalmente; no usa --inplace.
            additional = sum(r["bytes"] for name, r in indexed.items()
                             if not safe_target(destination, name).is_file())
            margin = max((r["bytes"] for r in indexed.values()), default=0) + 2 * 1024**3
            free = shutil.disk_usage(destination).free
            print(f"Inventario: {total / 1024**3:.2f} GiB; libres: {free / 1024**3:.2f} GiB.", flush=True)
            if free < additional + margin:
                raise ValueError("Espacio insuficiente con margen temporal; usa --dest en otro disco")
            write_json(destination / "manifest.json", inventory)
            for group, names in (("artefactos", list(indexed)), ("logs", inventory["logs"])):
                if not names:
                    continue
                file_list = Path(temp) / f"{group}.txt"
                file_list.write_text("\n".join(names) + "\n")
                print(f"Descargando {group}…", flush=True)
                subprocess.run(["rsync", "-avP", "--partial-dir=.rsync-partial", "--checksum",
                                "--files-from=" + str(file_list), "-e", shlex.join(ssh),
                                f"{args.remote}:{args.remote_dir}/", str(destination) + "/"], check=True)
            print("Comprobando SHA-256 locales…", flush=True)
            verify(destination, inventory)
        finally:
            # Cierra solo la conexión creada por este script, nunca otros SSH.
            if Path(socket).exists():
                subprocess.run(["ssh", "-S", socket, "-O", "exit", args.remote],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default=os.environ.get("KHIPU_REMOTE", "max.ramirez@khipu.utec.edu.pe"))
    parser.add_argument("--remote-dir", default=os.environ.get("KHIPU_DIR", "/home/max.ramirez/dmel"))
    parser.add_argument("--dest", type=Path, default=ROOT / "migration/khipu")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Solo muestra el plan; no abre SSH ni escribe")
    mode.add_argument("--verify-only", action="store_true", help="Verifica copia local, sin abrir SSH")
    args = parser.parse_args()
    if (not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", args.remote)
            or not re.fullmatch(r"/[A-Za-z0-9_./-]+", args.remote_dir)
            or ".." in PurePosixPath(args.remote_dir).parts
            or args.remote_dir in ("/", "/home/max.ramirez/deep")):
        parser.error("Host o ruta remota inválidos")
    destination = args.dest.expanduser().absolute()
    if destination == ROOT or ROOT.is_relative_to(destination):
        parser.error("Usa un directorio separado para el backup, nunca la raíz del proyecto")
    if destination.is_symlink() or any(p.is_symlink() for p in destination.parents):
        parser.error("El destino no puede atravesar symlinks")
    if args.dry_run:
        print(f"Origen: {args.remote}:{args.remote_dir}\nDestino: {destination}\n"
              "Datos train/test + cache/train.h5 + código remoto + corridas completas (best.pt y last.pt).\n"
              "Logs diagnósticos; corridas incompletas omitidas y registradas.\n"
              "Validación SHA-256. Sin cambios en Khipu, GitHub ni trabajos Slurm.\n"
              "Plan solamente: no se conectó ni se descargó nada.")
        return
    if args.verify_only:
        verify(destination, json.loads((destination / "manifest.json").read_text()))
    else:
        pull(args, destination)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Migración no verificada: {exc}\nCorrige el problema y repite el mismo comando; se conservan las descargas parciales.", file=sys.stderr)
        sys.exit(1)
