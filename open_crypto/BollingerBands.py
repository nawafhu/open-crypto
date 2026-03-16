#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bollinger Bands Backtest Tool

- lädt historic_rates_view aus SQLite
- Nutzer wählt Periode und Standardabweichung
- Bollinger Bands werden berechnet

Signal:
    pred_up = 1 wenn close < lower_band

Wahrheit:
    direction = 1 wenn close[t+1] > close[t]

Export:
    Sheet: bollinger_backtest
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
# Bollinger Bands berechnen
# --------------------------------------------------
def compute_bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = sma + std_mult * std
    lower = sma - std_mult * std

    return sma, upper, lower


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

    print("=== Bollinger Bands Tool ===")

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
        period = int(input("Bollinger Periode (z.B. 20): "))
        std_mult = float(input("Standardabweichung (z.B. 2): "))
    except ValueError:
        print("Ungültige Parameter.")
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

    if len(df) < period + 10:
        print("Nicht genügend Daten.")
        return

    # --------------------------------------------------
    # Datum & Sortierung
    # --------------------------------------------------
    df["date"] = time_to_datetime(df["time"])
    df["date_str"] = df["date"].dt.strftime("%d.%m.%Y")

    df = df.sort_values("time").reset_index(drop=True)

    # --------------------------------------------------
    # Bollinger Bands berechnen
    # --------------------------------------------------
    sma, upper, lower = compute_bollinger(df["close"], period, std_mult)

    df["bb_mid"] = sma
    df["bb_upper"] = upper
    df["bb_lower"] = lower

    # --------------------------------------------------
    # Aktueller Wert + Empfehlung
    # --------------------------------------------------
    last = df.iloc[-1]

    close = float(last["close"])
    upper = float(last["bb_upper"])
    lower = float(last["bb_lower"])
    mid = float(last["bb_mid"])

    print("\n=== Aktuelle Bollinger Bands ===")
    print(f"Datum        : {last['date_str']}")
    print(f"Close        : {close:.6f}")
    print(f"BB Upper     : {upper:.6f}")
    print(f"BB Middle    : {mid:.6f}")
    print(f"BB Lower     : {lower:.6f}")

    if close < lower:
        print("📈 Empfehlung: KAUFEN (Preis unter unterem Band)")
    elif close > upper:
        print("📉 Empfehlung: VERKAUFEN (Preis über oberem Band)")
    else:
        print("🤝 Empfehlung: HALTEN (Preis innerhalb der Bänder)")

    # --------------------------------------------------
    # Ziel & Prognose
    # --------------------------------------------------
    df["direction"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df["pred_up"] = (df["close"] < df["bb_lower"]).astype(int)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    eval_df = df.dropna(subset=["bb_lower", "direction"])

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
        ["time", "date_str", "close", "bb_upper", "bb_mid", "bb_lower",
         "direction", "pred_up"]
    ].sort_values("time", ascending=False)

    metrics_df = pd.DataFrame({
        "metric": [
            "period",
            "std_multiplier",
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
            period,
            std_mult,
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

    out_path = base_dir / f"{db_path.stem}_BOLLINGER_BACKTEST.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="bollinger_backtest", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)

    print("\n✅ Backtest abgeschlossen")
    print(f"📁 Datei erstellt: {out_path}")


if __name__ == "__main__":
    main()