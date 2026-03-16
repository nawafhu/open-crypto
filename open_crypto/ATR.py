#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ATR Backtest Tool

- lädt historic_rates_view aus SQLite
- Nutzer wählt ATR-Periode
- berechnet Average True Range (ATR)

Signal:
    pred_up = 1 wenn ATR steigt (Volatilitätsanstieg)

Wahrheit:
    direction = 1 wenn close[t+1] > close[t]

Export:
    Sheet: atr_backtest
    Sheet: metrics
"""

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# time -> datetime erkennen
# --------------------------------------------------
def time_to_datetime(time_series: pd.Series) -> pd.Series:
    s = pd.to_numeric(time_series, errors="coerce")
    mx = s.dropna().max()

    if mx > 1e11:
        dt = pd.to_datetime(s, unit="ms", utc=True)
    elif mx > 1e8:
        dt = pd.to_datetime(s, unit="s", utc=True)
    else:
        dt = pd.to_datetime("1970-01-01", utc=True) + pd.to_timedelta(s, unit="D")

    return dt.dt.tz_convert(None)


# --------------------------------------------------
# ATR berechnen
# --------------------------------------------------
def compute_atr(close: pd.Series, period: int = 14):

    high = close
    low = close
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()

    return atr


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
def confusion_counts(y_true, y_pred):
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    return int(tn), int(fp), int(fn), int(tp)


def safe_div(a, b):
    return a / b if b != 0 else np.nan


# --------------------------------------------------
# Hauptprogramm
# --------------------------------------------------
def main():

    print("=== ATR Backtest Tool ===")

    base_dir = Path(__file__).resolve().parent
    db_files = sorted(base_dir.glob("*.db"))

    if not db_files:
        print("Keine Datenbanken gefunden.")
        return

    print("\nGefundene Datenbanken:")
    for i, db in enumerate(db_files, 1):
        print(f"{i}: {db.name}")

    try:
        choice = int(input("\nNummer der Datenbank auswählen: "))
        if choice < 1 or choice > len(db_files):
            raise ValueError
    except ValueError:
        print("Ungültige Auswahl.")
        return

    try:
        atr_period = int(input("ATR-Periode (z.B. 14): "))
        if atr_period < 2:
            raise ValueError
    except ValueError:
        print("Ungültige ATR-Periode.")
        return

    db_path = db_files[choice - 1]

    print(f"\nVerbinde mit: {db_path.name}")

    # --------------------------------------------------
    # Daten laden
    # --------------------------------------------------
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(
        """
        SELECT time, close
        FROM historic_rates_view
        WHERE close IS NOT NULL
        ORDER BY time
        """,
        conn
    )

    conn.close()

    if len(df) < atr_period + 10:
        print("Nicht genügend Daten.")
        return

    # --------------------------------------------------
    # Datum & Sortierung
    # --------------------------------------------------
    df["date"] = time_to_datetime(df["time"])
    df["date_str"] = df["date"].dt.strftime("%d.%m.%Y")

    df = df.sort_values("time").reset_index(drop=True)

    # --------------------------------------------------
    # ATR berechnen
    # --------------------------------------------------
    atr_col = f"ATR_{atr_period}"

    df[atr_col] = compute_atr(df["close"], atr_period)

    # --------------------------------------------------
    # Aktueller ATR-Wert
    # --------------------------------------------------
    last = df.iloc[-1]

    current_close = float(last["close"])
    current_atr = float(last[atr_col])

    print("\n=== Aktueller ATR-Wert ===")
    print(f"Datum        : {last['date_str']}")
    print(f"Close        : {current_close:.6f}")
    print(f"{atr_col:<12}: {current_atr:.6f}")

    prev_atr = float(df.iloc[-2][atr_col])

    if current_atr > prev_atr:
        print("📈 Volatilität steigt → möglicher Trendbeginn")
    elif current_atr < prev_atr:
        print("📉 Volatilität sinkt → Markt beruhigt sich")
    else:
        print("🤝 Volatilität unverändert")

    # --------------------------------------------------
    # Ziel & Prognose
    # --------------------------------------------------
    df["direction"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df["pred_up"] = (df[atr_col].diff() > 0).astype(int)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    eval_df = df.dropna(subset=[atr_col, "direction"])

    y_true = eval_df["direction"].values
    y_pred = eval_df["pred_up"].values

    tn, fp, fn, tp = confusion_counts(y_true, y_pred)

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    specificity = safe_div(tn, tn + fp)
    signal_rate = safe_div(y_pred.sum(), len(y_pred))

    print("\n=== Backtest Ergebnisse ===")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print(f"Specificity   : {specificity:.4f}")

    # --------------------------------------------------
    # Export
    # --------------------------------------------------
    export_df = df[
        ["time", "date_str", "close", atr_col, "direction", "pred_up"]
    ].sort_values("time", ascending=False)

    metrics_df = pd.DataFrame({
        "metric": [
            "atr_period",
            "rows_evaluated",
            "signal_rate",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "TN", "FP", "FN", "TP"
        ],
        "value": [
            atr_period,
            len(eval_df),
            signal_rate,
            accuracy,
            precision,
            recall,
            f1,
            specificity,
            tn, fp, fn, tp
        ]
    })

    out_path = base_dir / f"{db_path.stem}_ATR{atr_period}_BACKTEST.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="atr_backtest", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)

    print("\n✅ Backtest abgeschlossen")
    print(f"📁 Datei erstellt: {out_path}")


if __name__ == "__main__":
    main()