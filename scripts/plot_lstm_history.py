"""Gráfica de evidencia recibida de Khipu; no importa ni modifica dmel."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=root / "outputs/khipu-lstm-53987-history-20261003.json")
    parser.add_argument("--out", type=Path, default=root / "outputs/khipu-lstm-53987-learning-20261003")
    parser.add_argument("--status", type=Path, help="Estado final recibido, opcional")
    args = parser.parse_args()
    evidence = json.loads(args.history.read_text())
    rows = evidence["history"]
    assert [r["epoch"] for r in rows] == list(range(len(rows)))
    epoch = [r["epoch"] + 1 for r in rows]
    rmse = [r["val_rmse_mm_h"] for r in rows]
    minutes = [r["seconds"] / 60 for r in rows]
    best = min(range(len(rows)), key=lambda i: rmse[i])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#16324f", "text.color": "#16324f",
                         "axes.titleweight": "bold", "savefig.facecolor": "white"})
    fig, (ax, duration) = plt.subplots(1, 2, figsize=(12, 5.1))
    fig.subplots_adjust(left=.085, right=.98, top=.78, bottom=.29, wspace=.30)
    fig.suptitle("LSTM · validación y tiempos de 20 épocas", fontsize=18, y=.965)
    fig.text(.5, .89, "Job 53987 · entrenamiento 254 000 ventanas · validación 18 142",
             ha="center", fontsize=11)
    ax.axvspan(epoch[best]+.5, epoch[-1]+.5, color="#eff3f6", zorder=0)
    ax.plot(epoch, rmse, color="#006d8f", marker="o", markersize=4, linewidth=2)
    ax.scatter([epoch[best]], [rmse[best]], color="#152f4a", s=70, zorder=4)
    ax.annotate(f"Mínimo: {rmse[best]:.6f} mm/h",
                xy=(epoch[best], rmse[best]), xytext=(0, 60),
                textcoords="offset points", fontsize=10, ha="center",
                arrowprops={"arrowstyle": "-", "color": "#152f4a"})
    ax.set(title="Mejor resultado: época 10", xlabel="Época (numeración desde 1)",
           ylabel="RMSE de validación (mm/h)", xlim=(.5, 20.5))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(axis="y", alpha=.22)
    duration.bar(epoch, minutes, color="#007fa3", width=.78)
    duration.set(title="Duración por época", xlabel="Época (numeración desde 1)",
                 ylabel="Minutos: entrenamiento + validación", xlim=(.5, 20.5), ylim=(0, 180))
    duration.set_xticks([1, 5, 10, 15, 20])
    duration.yaxis.set_major_locator(MaxNLocator(5))
    duration.grid(axis="y", alpha=.22)
    duration.set_axisbelow(True)
    fig.text(.085, .125, "Zona sombreada: 10 épocas consecutivas sin mejorar el mínimo.",
             fontsize=10)
    fig.text(.085, .077, "Fuente: history.json compartido por el usuario el 03/10/2026. Una corrida, semilla 42.",
             fontsize=9, color="#52616b")
    completion = "Finalización de Slurm pendiente de confirmar."
    if args.status:
        status = json.loads(args.status.read_text())
        if status.get("complete") and status.get("epochs_completed") == len(rows):
            completion = "Estado final recibido: complete=true; parada temprana."
    fig.text(.085, .04, "Tiempos excluyen inicialización, colas y guardado. " + completion,
             fontsize=9, color="#52616b")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.out.with_suffix(suffix), dpi=180)
    plt.close(fig)
    print(args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
