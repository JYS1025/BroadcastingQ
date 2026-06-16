from __future__ import annotations

import argparse
import csv
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


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out = []
    running_sum = 0.0
    running_count = 0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        if value == value:
            running_sum += value
            running_count += 1
        if len(queue) > window:
            old = queue.pop(0)
            if old == old:
                running_sum -= old
                running_count -= 1
        out.append(running_sum / running_count if running_count else float("nan"))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training or eval metrics from one or more runs.")
    parser.add_argument("runs", nargs="+", help="Run output directories.")
    parser.add_argument("--source", choices=["eval", "train"], default="eval", help="CSV source to plot.")
    parser.add_argument(
        "--metric",
        default="eval_return_mean",
        help="Metric column to plot. For train, common values are episode_return, success, episode_length, loss.",
    )
    parser.add_argument("--x-axis", choices=["step", "episode"], default="step", help="Training x-axis.")
    parser.add_argument("--window", type=int, default=1, help="Rolling mean window for plotted values.")
    parser.add_argument("--out", default="outputs/eval_returns.png", help="Output image path.")
    args = parser.parse_args()

    for run in args.runs:
        run_dir = Path(run)
        if args.source == "eval":
            xs, values = read_eval_metrics(run_dir / "eval_metrics.csv", args.metric)
            xlabel = "Environment steps"
        else:
            xs, values = read_train_metrics(run_dir / "metrics.csv", args.metric, args.x_axis)
            xlabel = "Environment steps" if args.x_axis == "step" else "Episode"
        values = rolling_mean(values, args.window)
        plt.plot(xs, values, label=run_dir.name)
    plt.xlabel(xlabel)
    plt.ylabel(args.metric)
    plt.legend()
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)


if __name__ == "__main__":
    main()
