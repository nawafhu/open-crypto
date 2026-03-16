#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CCI Backtest Tool

- lädt historic_rates_view aus SQLite
- benötigt time, high, low, close
- berechnet Commodity Channel Index (CCI)

Signal:
    pred_up = 1 wenn CCI < -100

Wahrheit:
    direction = 1 wenn close[t+1] > close[t]

Export:
    Sheet: cci_backtest
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
# CCI berechnen
# --------------------------------------------------
def compute_cci(high: pd.Series,
                low: pd.Series,
                close: pd.Series,
                period: int = 20) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    sma_tp = typical_price.rolling(window=period).mean()

    mad = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True
    )

    denom = (0.015 * mad).replace(0, np.nan)
    cci = (typical_price - sma_tp) / denom
    return cci


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
    print("=== CCI Backtest Tool ===")

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
        cci_period = int(input("CCI-Periode (z.B. 20) [Enter=20]: ").strip() or "20")
        if cci_period < 2:
            raise ValueError
    except ValueError:
        print("Ungültige CCI-Periode. Nutze 20.")
        cci_period = 20

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

    if len(df) < cci_period + 10:
        print("Nicht genügend Daten.")
        return

    # --------------------------------------------------
    # Datum & Sortierung
    # --------------------------------------------------
    df["date"] = time_to_datetime(df["time"])
    df["date_str"] = df["date"].dt.strftime("%d.%m.%Y")
    df = df.sort_values("time").reset_index(drop=True)

    # --------------------------------------------------
    # CCI berechnen
    # --------------------------------------------------
    cci_col = f"CCI_{cci_period}"
    df[cci_col] = compute_cci(df["high"], df["low"], df["close"], period=cci_period)

    # --------------------------------------------------
    # Aktueller Wert + Empfehlung
    # --------------------------------------------------
    last = df.iloc[-1]

    current_high = float(last["high"])
    current_low = float(last["low"])
    current_close = float(last["close"])
    current_cci = float(last[cci_col])

    print("\n=== Aktueller CCI-Wert ===")
    print(f"Datum        : {last['date_str']}")
    print(f"High         : {current_high:.6f}")
    print(f"Low          : {current_low:.6f}")
    print(f"Close        : {current_close:.6f}")
    print(f"{cci_col:<12}: {current_cci:.2f}")

    if current_cci < -100:
        print("✅ Signal: BULLISH")
        print("Begründung   : Der CCI liegt unter -100 und deutet auf eine überverkaufte Situation hin.")
        print("📈 Empfehlung: KAUFEN")
    elif current_cci > 100:
        print("⚠️ Signal: BEARISH")
        print("Begründung   : Der CCI liegt über +100 und deutet auf eine überkaufte Situation hin.")
        print("📉 Empfehlung: VERKAUFEN")
    else:
        print("➖ Signal: NEUTRAL")
        print("Begründung   : Der CCI liegt im neutralen Bereich.")
        print("🤝 Empfehlung: HALTEN")

    print("\nLetzte 5 Werte:")
    print(df[["date_str", "high", "low", "close", cci_col]].tail(5))

    # --------------------------------------------------
    # Ziel & Prognose
    # --------------------------------------------------
    df["direction"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Einfache Mean-Reversion-Logik:
    # überverkauft => pred_up = 1
    df["pred_up"] = (df[cci_col] < -100).astype(int)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    eval_df = df.dropna(subset=[cci_col, "direction"]).copy()

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
        ["time", "date_str", "high", "low", "close", cci_col, "direction", "pred_up"]
    ].sort_values("time", ascending=False).copy()

    metrics_df = pd.DataFrame({
        "metric": [
            "cci_period",
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
            cci_period,
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

    out_path = base_dir / f"{db_path.stem}_CCI{cci_period}_BACKTEST.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="cci_backtest", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)

    print("\n✅ Backtest abgeschlossen")
    print(f"📁 Datei erstellt: {out_path}")


if __name__ == "__main__":
    main()