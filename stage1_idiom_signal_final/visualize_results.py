from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODELS = [
    "old_marked_input",
    "new_target_only",
    "hybrid_marked_mlp",
    "neural_m",
    "hybrid_llm_fusion",
]

DISPLAY_NAMES = {
    "old_marked_input": "Old marked input",
    "new_target_only": "Target only",
    "hybrid_marked_mlp": "Hybrid marked MLP",
    "neural_m": "Neural M",
    "hybrid_llm_fusion": "Hybrid LLM fusion",
}


pd = None
plt = None
sns = None


def ensure_report_dependencies() -> None:
    global pd, plt, sns
    if pd is not None and plt is not None and sns is not None:
        return

    try:
        import matplotlib.pyplot as matplotlib_pyplot
        import pandas as pandas_module
        import seaborn as seaborn_module
    except ImportError as exc:
        raise ImportError(
            "Missing report dependencies. Install them with: "
            "pip install pandas matplotlib seaborn"
        ) from exc

    pd = pandas_module
    plt = matplotlib_pyplot
    sns = seaborn_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tables and plots from Stage 1 comparison metrics.")
    parser.add_argument("--output_root", default="stage1_idiom_signal_final/outputs")
    parser.add_argument("--report_dir", default="stage1_idiom_signal_final/reports")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    return parser.parse_args()


def load_metrics(output_root: Path, model_name: str) -> dict[str, Any] | None:
    path = output_root / model_name / "metrics.json"
    if not path.exists():
        print(f"Skipping missing metrics: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        model_name = result["name"]
        best_metrics = result.get("best_metrics") or {}
        rows.append(
            {
                "model": model_name,
                "display_name": DISPLAY_NAMES.get(model_name, model_name),
                "best_epoch": result.get("best_epoch"),
                "best_val_f1": result.get("best_val_f1"),
                "best_val_accuracy": best_metrics.get("accuracy"),
                "best_val_auc": best_metrics.get("roc_auc"),
                "best_val_loss": best_metrics.get("loss"),
                "train_size": result.get("train_size"),
                "val_size": result.get("val_size"),
                "num_examples": result.get("num_examples"),
                "input_format": result.get("input_format"),
                "head_type": result.get("head_type"),
            }
        )
    return rows


def build_history_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        model_name = result["name"]
        display_name = DISPLAY_NAMES.get(model_name, model_name)
        for item in result.get("history", []):
            rows.append(
                {
                    "model": model_name,
                    "display_name": display_name,
                    "epoch": item.get("epoch"),
                    "train_loss": item.get("train_loss"),
                    "val_loss": item.get("val_loss"),
                    "val_accuracy": item.get("val_accuracy"),
                    "val_f1": item.get("val_f1"),
                    "val_auc": item.get("val_roc_auc"),
                }
            )
    return rows


def format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def save_markdown_table(summary_df: Any, path: Path) -> None:
    columns = [
        "display_name",
        "best_epoch",
        "best_val_f1",
        "best_val_accuracy",
        "best_val_auc",
        "best_val_loss",
    ]
    table = summary_df[columns].copy()
    table = table.rename(
        columns={
            "display_name": "Model",
            "best_epoch": "Best Epoch",
            "best_val_f1": "Val F1",
            "best_val_accuracy": "Val Acc",
            "best_val_auc": "Val AUC",
            "best_val_loss": "Val Loss",
        }
    )
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(format_markdown_value(row[column]) for column in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_best_metrics(summary_df: Any, path: Path) -> None:
    plot_df = summary_df.melt(
        id_vars=["display_name"],
        value_vars=["best_val_f1", "best_val_accuracy", "best_val_auc"],
        var_name="metric",
        value_name="score",
    )
    plot_df["metric"] = plot_df["metric"].map(
        {
            "best_val_f1": "F1",
            "best_val_accuracy": "Accuracy",
            "best_val_auc": "ROC-AUC",
        }
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="display_name", y="score", hue="metric")
    plt.ylim(0.0, 1.0)
    plt.xlabel("")
    plt.ylabel("Validation score")
    plt.title("Best Validation Metrics by Model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_history_metric(history_df: Any, metric: str, ylabel: str, title: str, path: Path) -> None:
    if history_df.empty or metric not in history_df.columns:
        return

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=history_df, x="epoch", y=metric, hue="display_name", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def create_report(output_root: str | Path, report_dir: str | Path, models: list[str] | None = None) -> dict[str, Path]:
    ensure_report_dependencies()

    output_root_path = Path(output_root)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    selected_models = models or DEFAULT_MODELS
    results = [result for name in selected_models if (result := load_metrics(output_root_path, name)) is not None]
    if not results:
        raise FileNotFoundError(f"No metrics files found under {output_root_path}")

    summary_df = pd.DataFrame(build_summary_rows(results))
    history_df = pd.DataFrame(build_history_rows(results))

    model_order = [DISPLAY_NAMES.get(name, name) for name in selected_models]
    summary_df["display_name"] = pd.Categorical(summary_df["display_name"], categories=model_order, ordered=True)
    summary_df = summary_df.sort_values("display_name")

    summary_csv = report_path / "summary.csv"
    summary_md = report_path / "summary.md"
    history_csv = report_path / "history.csv"
    best_metrics_png = report_path / "best_metrics.png"
    f1_curve_png = report_path / "val_f1_curve.png"
    loss_curve_png = report_path / "val_loss_curve.png"

    summary_df.to_csv(summary_csv, index=False)
    history_df.to_csv(history_csv, index=False)
    save_markdown_table(summary_df, summary_md)

    sns.set_theme(style="whitegrid")
    plot_best_metrics(summary_df, best_metrics_png)
    plot_history_metric(history_df, "val_f1", "Validation F1", "Validation F1 Across Epochs", f1_curve_png)
    plot_history_metric(history_df, "val_loss", "Validation loss", "Validation Loss Across Epochs", loss_curve_png)

    print("")
    print("Summary")
    print(summary_df[["display_name", "best_epoch", "best_val_f1", "best_val_accuracy", "best_val_auc"]].to_string(index=False))
    print("")
    print(f"Saved report files to: {report_path}")
    print(f"- {summary_csv}")
    print(f"- {summary_md}")
    print(f"- {history_csv}")
    print(f"- {best_metrics_png}")
    print(f"- {f1_curve_png}")
    print(f"- {loss_curve_png}")

    return {
        "summary_csv": summary_csv,
        "summary_md": summary_md,
        "history_csv": history_csv,
        "best_metrics_png": best_metrics_png,
        "f1_curve_png": f1_curve_png,
        "loss_curve_png": loss_curve_png,
    }


def main() -> None:
    args = parse_args()
    create_report(output_root=args.output_root, report_dir=args.report_dir, models=args.models)


if __name__ == "__main__":
    main()
