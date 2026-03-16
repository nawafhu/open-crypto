#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stochastic Oscillator Backtest Tool

- lädt historic_rates_view aus SQLite
- benötigt time, high, low, close
- Nutzer wählt %K-Periode und %D-Periode
- berechnet Stochastic Oscillator (%K und %D)

Signal:
    pred_up = 1 wenn %K > %D und %K < 20

Wahrheit:
    direction = 1 wenn close[t+1] > close[t]

Export:
    Sheet: stochastic_backtest
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
# Stochastic Oscillator berechnen
# --------------------------------------------------
def compute_stochastic(high: pd.Series,
                       low: pd.Series,
                       close: pd.Series,
                       k_period: int = 14,
                       d_period: int = 3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    denom = (highest_high - lowest_low).replace(0, np.nan)
    k_percent = 100 * (close - lowest_low) / denom
    d_percent = k_percent.rolling(window=d_period).mean()

    return k_percent, d_percent


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

    print("=== Stochastic Oscillator Backtest Tool ===")

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
        k_period = int(input("%K-Periode (z.B. 14) [Enter=14]: ").strip() or "14")
        d_period = int(input("%D-Periode (z.B. 3) [Enter=3]: ").strip() or "3")
        if k_period < 2 or d_period < 1:
            raise ValueError
    except ValueError:
        print("Ungültige Parameter.")
        return

    db_path = db_files[choice - 1]
    print(f"\nVerbinde mit: {db_path.name}")

    # --------------------------------------------------
    # Daten laden
    # --------------------------------------------------
    conn = sqlite3.connect(db_path)

    try:
        df = pd.read_sql_query(
            """
            SELECT time, high, low, close
            FROM historic_rates_view
            WHERE high IS NOT NULL
              AND low IS NOT NULL
              AND close IS NOT NULL
            ORDER BY time
            """,
            conn
        )
    except Exception:
        conn.close()
        print("Die Tabelle historic_rates_view benötigt die Spalten: time, high, low, close.")
        return

    conn.close()

    if len(df) < k_period + d_period + 10:
        print("Nicht genügend Daten.")
        return

    # --------------------------------------------------
    # Datum & Sortierung
    # --------------------------------------------------
    df["date"] = time_to_datetime(df["time"])
    df["date_str"] = df["date"].dt.strftime("%d.%m.%Y")
    df = df.sort_values("time").reset_index(drop=True)

    # --------------------------------------------------
    # Stochastic Oscillator berechnen
    # --------------------------------------------------
    df["stoch_k"], df["stoch_d"] = compute_stochastic(
        df["high"],
        df["low"],
        df["close"],
        k_period=k_period,
        d_period=d_period
    )

    # --------------------------------------------------
    # Aktuelle Werte + Empfehlung
    # --------------------------------------------------
    last = df.iloc[-1]

    current_close = float(last["close"])
    current_k = float(last["stoch_k"])
    current_d = float(last["stoch_d"])

    print("\n=== Aktueller Stochastic Oscillator ===")
    print(f"Datum        : {last['date_str']}")
    print(f"Close        : {current_close:.6f}")
    print(f"%K           : {current_k:.2f}")
    print(f"%D           : {current_d:.2f}")

    if current_k > current_d and current_k < 20:
        print("✅ Signal: BULLISH")
        print("Begründung   : %K liegt über %D und der Markt ist im überverkauften Bereich.")
        print("📈 Empfehlung: KAUFEN")
    elif current_k < current_d and current_k > 80:
        print("⚠️ Signal: BEARISH")
        print("Begründung   : %K liegt unter %D und der Markt ist im überkauften Bereich.")
        print("📉 Empfehlung: VERKAUFEN")
    else:
        print("➖ Signal: NEUTRAL")
        print("Begründung   : Kein klares Kauf- oder Verkaufssignal.")
        print("🤝 Empfehlung: HALTEN")

    print("\nLetzte 5 Werte:")
    print(df[["date_str", "close", "stoch_k", "stoch_d"]].tail(5))

    # --------------------------------------------------
    # Ziel & Prognose
    # --------------------------------------------------
    df["direction"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Kaufsignal:
    # %K kreuzt über %D und Bereich < 20
    df["pred_up"] = ((df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"] < 20)).astype(int)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    eval_df = df.dropna(subset=["stoch_k", "stoch_d", "direction"]).copy()

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
    print(f"Rows evaluated: {len(eval_df)}")
    print(f"Signal Rate   : {signal_rate:.4f}")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print(f"Specificity   : {specificity:.4f}")
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

    # --------------------------------------------------
    # Export
    # --------------------------------------------------
    export_df = df[
        ["time", "date_str", "high", "low", "close", "stoch_k", "stoch_d", "direction", "pred_up"]
    ].sort_values("time", ascending=False).copy()

    metrics_df = pd.DataFrame({
        "metric": [
            "k_period",
            "d_period",
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
            k_period,
            d_period,
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

    out_path = base_dir / f"{db_path.stem}_STOCHASTIC_{k_period}_{d_period}_BACKTEST.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="stochastic_backtest", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)

    print("\n✅ Backtest abgeschlossen")
    print(f"📁 Datei erstellt: {out_path}")


if __name__ == "__main__":
    main()