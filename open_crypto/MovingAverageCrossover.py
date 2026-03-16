#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moving Average Crossover Backtest Tool

- lädt historic_rates_view aus SQLite
- benötigt time, close
- berechnet zwei EMAs:
    short EMA
    long EMA

Signal:
    pred_up = 1 wenn EMA_short > EMA_long

Wahrheit:
    direction = 1 wenn close[t+1] > close[t]

Export:
    Sheet: ma_crossover_backtest
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
    s_non_na = s.dropna()
    if s_non_na.empty:
        raise ValueError("time-Spalte enthält keine numerischen Werte.")

    mx = float(s_non_na.max())

    if mx > 1e11:
        dt = pd.to_datetime(s, unit="ms", utc=True)
    elif mx > 1e8:
        dt = pd.to_datetime(s, unit="s", utc=True)
    else:
        dt = pd.to_datetime("1970-01-01", utc=True) + pd.to_timedelta(s, unit="D")

    return dt.dt.tz_convert(None)


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
    print("=== Moving Average Crossover Backtest Tool ===")

    base_dir = Path(__file__).resolve().parent
    db_files = sorted(base_dir.glob("*.db"))

    if not db_files:
        print("Keine Datenbanken gefunden.")
        return

    print("\nGefundene Datenbanken:")
    for i, db in enumerate(db_files, 1):
        print(f"{i}: {db.name}")

    try:
        choice = int(input("\nNummer der Datenbank auswählen: ").strip())
        if choice < 1 or choice > len(db_files):
            raise ValueError
    except ValueError:
        print("Ungültige Auswahl.")
        return

    try:
        short_period = int(input("Kurze EMA-Periode (z.B. 20) [Enter=20]: ").strip() or "20")
        long_period = int(input("Lange EMA-Periode (z.B. 50) [Enter=50]: ").strip() or "50")

        if short_period < 2 or long_period < 2 or short_period >= long_period:
            raise ValueError
    except ValueError:
        print("Ungültige Perioden. Beispiel: short=20, long=50 und short < long.")
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

    if len(df) < long_period + 10:
        print("Nicht genügend Daten.")
        return

    # --------------------------------------------------
    # Datum & Sortierung
    # --------------------------------------------------
    df["date"] = time_to_datetime(df["time"])
    df["date_str"] = df["date"].dt.strftime("%d.%m.%Y")
    df = df.sort_values("time").reset_index(drop=True)

    # --------------------------------------------------
    # EMAs berechnen
    # --------------------------------------------------
    short_col = f"EMA_{short_period}"
    long_col = f"EMA_{long_period}"

    df[short_col] = df["close"].ewm(span=short_period, adjust=False).mean()
    df[long_col] = df["close"].ewm(span=long_period, adjust=False).mean()

    # Optional: echtes Kreuzungssignal markieren
    df["crossover_up"] = (
            (df[short_col] > df[long_col]) &
            (df[short_col].shift(1) <= df[long_col].shift(1))
    ).astype(int)

    df["crossover_down"] = (
            (df[short_col] < df[long_col]) &
            (df[short_col].shift(1) >= df[long_col].shift(1))
    ).astype(int)

    # --------------------------------------------------
    # Aktueller Wert + Empfehlung
    # --------------------------------------------------
    last = df.iloc[-1]

    current_close = float(last["close"])
    current_short = float(last[short_col])
    current_long = float(last[long_col])

    print("\n=== Aktueller Moving Average Crossover ===")
    print(f"Datum        : {last['date_str']}")
    print(f"Close        : {current_close:.6f}")
    print(f"{short_col:<12}: {current_short:.6f}")
    print(f"{long_col:<12}: {current_long:.6f}")

    if int(last["crossover_up"]) == 1:
        print("✅ Signal: GOLDEN CROSS")
        print("Begründung   : Die kurze EMA hat die lange EMA aktuell von unten nach oben gekreuzt.")
        print("📈 Empfehlung: KAUFEN")
    elif int(last["crossover_down"]) == 1:
        print("⚠️ Signal: DEATH CROSS")
        print("Begründung   : Die kurze EMA hat die lange EMA aktuell von oben nach unten gekreuzt.")
        print("📉 Empfehlung: VERKAUFEN")
    elif current_short > current_long:
        print("✅ Signal: BULLISH")
        print("Begründung   : Die kurze EMA liegt über der langen EMA.")
        print("📈 Empfehlung: KAUFEN")
    elif current_short < current_long:
        print("⚠️ Signal: BEARISH")
        print("Begründung   : Die kurze EMA liegt unter der langen EMA.")
        print("📉 Empfehlung: VERKAUFEN")
    else:
        print("➖ Signal: NEUTRAL")
        print("Begründung   : Beide EMAs liegen auf ähnlichem Niveau.")
        print("🤝 Empfehlung: HALTEN")

    print("\nLetzte 5 Werte:")
    print(df[["date_str", "close", short_col, long_col, "crossover_up", "crossover_down"]].tail(5))

    # --------------------------------------------------
    # Ziel & Prognose
    # --------------------------------------------------
    df["direction"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # einfache Trendlogik:
    # bullish wenn kurze EMA > lange EMA
    df["pred_up"] = (df[short_col] > df[long_col]).astype(int)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    eval_df = df.dropna(subset=[short_col, long_col, "direction"]).copy()

    y_true = eval_df["direction"].to_numpy(dtype=int)
    y_pred = eval_df["pred_up"].to_numpy(dtype=int)

    tn, fp, fn, tp = confusion_counts(y_true, y_pred)

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    specificity = safe_div(tn, tn + fp)
    signal_rate = safe_div((y_pred == 1).sum(), len(y_pred))

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
        [
            "time", "date_str", "close",
            short_col, long_col,
            "crossover_up", "crossover_down",
            "direction", "pred_up"
        ]
    ].sort_values("time", ascending=False).copy()

    metrics_df = pd.DataFrame({
        "metric": [
            "short_period",
            "long_period",
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
            short_period,
            long_period,
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

    out_path = base_dir / f"{db_path.stem}_MA_CROSSOVER_{short_period}_{long_period}_BACKTEST.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="ma_crossover_backtest", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)

    print("\n✅ Backtest abgeschlossen")
    print(f"📁 Datei erstellt: {out_path}")


if __name__ == "__main__":
    main()