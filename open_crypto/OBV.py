#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OBV Backtest Tool

- lädt historic_rates_view aus SQLite
- benötigt time, close, volume
- berechnet On-Balance Volume (OBV)

Signal:
    pred_up = 1 wenn OBV steigt

Wahrheit:
    direction = 1 wenn close[t+1] > close[t]

Export:
    Sheet: obv_backtest
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
# OBV berechnen
# --------------------------------------------------
def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    close_diff = close.diff()

    direction = np.where(close_diff > 0, 1,
                         np.where(close_diff < 0, -1, 0))

    obv_change = volume * direction
    obv = pd.Series(obv_change, index=close.index).fillna(0).cumsum()

    return obv


#55515020,546464651

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
    print("=== OBV Backtest Tool ===")

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

    db_path = db_files[choice - 1]
    print(f"\nVerbinde mit: {db_path.name}")

    # --------------------------------------------------
    # Daten laden
    # --------------------------------------------------
    conn = sqlite3.connect(db_path)

    try:
        df = pd.read_sql_query(
            """
            SELECT time, close, volume
            FROM historic_rates_view
            WHERE close IS NOT NULL
              AND volume IS NOT NULL
            ORDER BY time
            """,
            conn
        )
    except Exception:
        conn.close()
        print("Die Tabelle historic_rates_view benötigt die Spalten: time, close, volume.")
        return

    conn.close()

    if len(df) < 20:
        print("Nicht genügend Daten.")
        return

    # --------------------------------------------------
    # Datum & Sortierung
    # --------------------------------------------------
    df["date"] = time_to_datetime(df["time"])
    df["date_str"] = df["date"].dt.strftime("%d.%m.%Y")
    df = df.sort_values("time").reset_index(drop=True)

    # --------------------------------------------------
    # OBV berechnen
    # --------------------------------------------------
    df["obv"] = compute_obv(df["close"], df["volume"])

    # Optional: geglättete OBV-Linie
    df["obv_sma_10"] = df["obv"].rolling(window=10).mean()

    # --------------------------------------------------
    # Aktueller Wert + Empfehlung
    # --------------------------------------------------
    last = df.iloc[-1]
    current_close = float(last["close"])
    current_obv = float(last["obv"])
    current_obv_sma = float(last["obv_sma_10"]) if not pd.isna(last["obv_sma_10"]) else np.nan

    print("\n=== Aktueller OBV-Wert ===")
    print(f"Datum        : {last['date_str']}")
    print(f"Close        : {current_close:.6f}")
    print(f"Volume       : {float(last['volume']):.6f}")
    print(f"OBV          : {current_obv:.6f}")

    if not np.isnan(current_obv_sma):
        print(f"OBV_SMA_10   : {current_obv_sma:.6f}")

    if len(df) >= 2:
        prev_obv = float(df.iloc[-2]["obv"])

        if not np.isnan(current_obv_sma):
            if current_obv > current_obv_sma and current_obv > prev_obv:
                print("✅ Signal: BULLISH")
                print("Begründung   : OBV steigt und liegt über seiner gleitenden Linie.")
                print("📈 Empfehlung: KAUFEN")
            elif current_obv < current_obv_sma and current_obv < prev_obv:
                print("⚠️ Signal: BEARISH")
                print("Begründung   : OBV fällt und liegt unter seiner gleitenden Linie.")
                print("📉 Empfehlung: VERKAUFEN")
            else:
                print("➖ Signal: NEUTRAL")
                print("Begründung   : Kein klares Volumensignal.")
                print("🤝 Empfehlung: HALTEN")
        else:
            if current_obv > prev_obv:
                print("✅ Signal: BULLISH")
                print("Begründung   : OBV steigt gegenüber der Vorperiode.")
                print("📈 Empfehlung: KAUFEN")
            elif current_obv < prev_obv:
                print("⚠️ Signal: BEARISH")
                print("Begründung   : OBV fällt gegenüber der Vorperiode.")
                print("📉 Empfehlung: VERKAUFEN")
            else:
                print("➖ Signal: NEUTRAL")
                print("Begründung   : OBV unverändert.")
                print("🤝 Empfehlung: HALTEN")

    print("\nLetzte 5 Werte:")
    print(df[["date_str", "close", "volume", "obv", "obv_sma_10"]].tail(5))

    # --------------------------------------------------
    # Ziel & Prognose
    # --------------------------------------------------
    df["direction"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Einfache Prognose:
    # bullish wenn OBV steigt
    df["pred_up"] = (df["obv"].diff() > 0).astype(int)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    eval_df = df.dropna(subset=["obv", "direction"]).copy()

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
        ["time", "date_str", "close", "volume", "obv", "obv_sma_10", "direction", "pred_up"]
    ].sort_values("time", ascending=False).copy()

    metrics_df = pd.DataFrame({
        "metric": [
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

    out_path = base_dir / f"{db_path.stem}_OBV_BACKTEST.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="obv_backtest", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)

    print("\n✅ Backtest abgeschlossen")
    print(f"📁 Datei erstellt: {out_path}")


if __name__ == "__main__":
    main()