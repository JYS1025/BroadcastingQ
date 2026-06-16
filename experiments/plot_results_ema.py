from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir()) / "broadcastingq"
mpl_config_dir = cache_root / "matplotlib"
cache_dir = cache_root / "cache"

mpl_config_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_value(value: str) -> float:
    if value == "":
        return float("nan")
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    return float(value)


def exponential_moving_average(values: list[float], span: int) -> list[float]:
    if span <= 1:
        return values
    alpha = 2.0 / (float(span) + 1.0)
    out: list[float] = []
    ema = float("nan")
    for value in values:
        if math.isnan(value):
            out.append(ema)
            continue
        if math.isnan(ema):
            ema = value
        else:
            ema = alpha * value + (1.0 - alpha) * ema
        out.append(ema)
    return out


def read_eval_metrics(path: Path, metric: str) -> tuple[list[int], list[float]]:
    steps = []
    values = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            values.append(parse_value(row[metric]))
    return steps, values


def read_train_metrics(path: Path, metric: str, x_axis: str) -> tuple[list[int], list[float]]:
    by_episode: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_episode[int(row["episode"])] = row

    xs = []
    values = []
    for episode, row in sorted(by_episode.items()):
        xs.append(int(row[x_axis]) if x_axis == "step" else episode)
        values.append(parse_value(row[metric]))
    return xs, values


def pretty_label(run_dir: Path) -> str:
    label = run_dir.name.replace("_", " ")
    label = label.replace("(ours)", " (ours)")
    return label


def setup_style() -> None:
    plt.rcParams.update(
        {
            "axes.facecolor": "#f8fafc",
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "axes.labelcolor": "#0f172a",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "font.size": 11,
            "grid.alpha": 0.35,
            "grid.color": "#94a3b8",
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#e2e8f0",
            "lines.linewidth": 2.5,
            "savefig.dpi": 180,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training or eval metrics with exponential moving averages.")
    parser.add_argument("runs", nargs="+", help="Run output directories.")
    parser.add_argument("--source", choices=["eval", "train"], default="eval", help="CSV source to plot.")
    parser.add_argument(
        "--metric",
        default="eval_return_mean",
        help="Metric column to plot. For train, common values are episode_return, success, episode_length, loss.",
    )
    parser.add_argument("--x-axis", choices=["step", "episode"], default="step", help="Training x-axis.")
    parser.add_argument("--window", type=int, default=5, help="EMA span. Use 1 to disable smoothing.")
    parser.add_argument("--out", default="outputs/eval_returns_ema.png", help="Output image path.")
    parser.add_argument("--show-raw", action="store_true", help="Also draw faint unsmoothed curves.")
    parser.add_argument("--raw-alpha", type=float, default=0.18, help="Opacity for --show-raw curves.")
    args = parser.parse_args()

    setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = plt.get_cmap("tab10").colors
    xlabel = "Environment steps" if args.source == "eval" or args.x_axis == "step" else "Episode"

    for idx, run in enumerate(args.runs):
        run_dir = Path(run)
        if args.source == "eval":
            xs, values = read_eval_metrics(run_dir / "eval_metrics.csv", args.metric)
        else:
            xs, values = read_train_metrics(run_dir / "metrics.csv", args.metric, args.x_axis)

        color = colors[idx % len(colors)]
        smoothed = exponential_moving_average(values, args.window)
        label = pretty_label(run_dir)
        if args.show_raw:
            ax.plot(xs, values, color=color, alpha=max(0.0, min(1.0, args.raw_alpha)), linewidth=1.1)
        ax.plot(xs, smoothed, color=color, label=label)

    title_metric = args.metric.replace("_", " ")
    ax.set_title(f"{title_metric} with EMA smoothing", color="#0f172a", pad=14, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(args.metric)
    ax.margins(x=0.015)
    ax.legend(loc="best")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
