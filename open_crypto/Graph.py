#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Accuracy Overview Plotter
- sucht alle .xlsx Dateien im gleichen Ordner wie dieses Skript (rekursiv optional)
- liest aus jedem Excel das Sheet 'metrics'
- sucht dort die Zeile mit metric == 'accuracy'
- sammelt alle accuracy Werte
- erstellt Übersichtsgrafik + CSV Export

Erwartetes Format im Sheet 'metrics':
metric | value
accuracy | 0.51
...

Falls in manchen Dateien accuracy anders heißt (z.B. 'Accuracy'), wird case-insensitive gesucht.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def find_accuracy_in_excel(xlsx_path: Path) -> float | None:
    """Return accuracy value from sheet 'metrics', or None if not found."""
    try:
        metrics = pd.read_excel(xlsx_path, sheet_name="metrics", engine="openpyxl")
    except Exception:
        return None

    if metrics.empty:
        return None

    # Normalize columns
    cols = [c.lower().strip() for c in metrics.columns]
    metrics.columns = cols

    if "metric" not in metrics.columns or "value" not in metrics.columns:
        return None

    # Case-insensitive metric match
    m = metrics["metric"].astype(str).str.strip().str.lower()
    hit = metrics.loc[m == "accuracy", "value"]

    if hit.empty:
        return None

    val = hit.iloc[0]

    # Convert comma decimals "0,51" -> 0.51 if needed
    if isinstance(val, str):
        val = val.strip().replace(",", ".")
        try:
            val = float(val)
        except ValueError:
            return None

    try:
        return float(val)
    except Exception:
        return None


def main():
    base_dir = Path(__file__).resolve().parent

    # Alle Excel-Dateien im Ordner (nicht rekursiv)
    xlsx_files = sorted(base_dir.glob("*.xlsx"))

    # Wenn du auch Unterordner willst, nutze stattdessen:
    # xlsx_files = sorted(base_dir.rglob("*.xlsx"))

    if not xlsx_files:
        print("Keine .xlsx Dateien gefunden.")
        return

    rows = []
    for f in xlsx_files:
        acc = find_accuracy_in_excel(f)
        if acc is not None:
            rows.append({"file": f.name, "accuracy": acc})

    if not rows:
        print("Keine accuracy Werte gefunden (Sheet 'metrics' oder Zeile 'accuracy' fehlt).")
        return

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)

    # CSV export
    csv_path = base_dir / "accuracy_overview.csv"
    df.to_csv(csv_path, index=False)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(df["file"], df["accuracy"])
    plt.axhline(0.5, linestyle="--")  # Zufallsbaseline
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Übersicht: Accuracy aus allen Excel-Backtests")
    plt.tight_layout()

    out_png = base_dir / "accuracy_overview.png"
    plt.savefig(out_png, dpi=200)
    plt.show()

    print(f"✅ CSV gespeichert: {csv_path}")
    print(f"✅ Grafik gespeichert: {out_png}")
    print("\nGefundene Dateien & Accuracy:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()