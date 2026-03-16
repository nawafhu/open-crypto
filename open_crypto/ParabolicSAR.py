#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parabolic SAR Backtest Tool

- lädt historic_rates_view aus SQLite
- benötigt time, high, low, close
- berechnet Parabolic SAR

Signal:
    pred_up = 1 wenn close > psar

Wahrheit:
    direction = 1 wenn close[t+1] > close[t]

Export:
    Sheet: psar_backtest
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
# Parabolic SAR berechnen
# --------------------------------------------------
def compute_psar(high: pd.Series,
                 low: pd.Series,
                 step: float = 0.02,
                 max_step: float = 0.2) -> pd.Series:
    high = high.to_numpy(dtype=float)
    low = low.to_numpy(dtype=float)

    n = len(high)
    psar = np.zeros(n, dtype=float)

    if n < 2:
        return pd.Series(psar, index=range(n))

    # Anfangstrend festlegen
    bull = high[1] >= high[0]

    # Initialwerte
    psar[0] = low[0] if bull else high[0]
    ep = high[0] if bull else low[0]
    af = step

    for i in range(1, n):
        prev_psar = psar[i - 1]

        # Standard-PSAR-Fortschreibung
        current_psar = prev_psar + af * (ep - prev_psar)

        if bull:
            # SAR darf nicht über den letzten beiden Tiefs liegen
            if i >= 2:
                current_psar = min(current_psar, low[i - 1], low[i - 2])
            else:
                current_psar = min(current_psar, low[i - 1])

            # Trendwechsel?
            if low[i] < current_psar:
                bull = False
                current_psar = ep
                ep = low[i]
                af = step
            else:
                # Trend fortgesetzt
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)

        else:
            # SAR darf nicht unter den letzten beiden Hochs liegen
            if i >= 2:
                current_psar = max(current_psar, high[i - 1], high[i - 2])
            else:
                current_psar = max(current_psar, high[i - 1])

            # Trendwechsel?
            if high[i] > current_psar:
                bull = True
                current_psar = ep
                ep = high[i]
                af = step
            else:
                # Trend fortgesetzt
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)

        psar[i] = current_psar

    return pd.Series(psar, index=high.index if isinstance(high, pd.Series) else None)


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
    print("=== Parabolic SAR Backtest Tool ===")

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
        step = float(input("PSAR Schrittweite (z.B. 0.02) [Enter=0.02]: ").strip() or "0.02")
        max_step = float(input("PSAR Max-Step (z.B. 0.2) [Enter=0.2]: ").strip() or "0.2")
        if step <= 0 or max_step <= 0 or step > max_step:
            raise ValueError
    except ValueError:
        print("Ungültige Parameter. Nutze Standard: step=0.02, max_step=0.2.")
        step, max_step = 0.02, 0.2

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
    # Parabolic SAR berechnen
    # --------------------------------------------------
    df["psar"] = compute_psar(df["high"], df["low"], step=step, max_step=max_step)

    # --------------------------------------------------
    # Aktuelle Werte + Empfehlung
    # --------------------------------------------------
    last = df.iloc[-1]

    current_close = float(last["close"])
    current_psar = float(last["psar"])
    current_high = float(last["high"])
    current_low = float(last["low"])

    print("\n=== Aktueller Parabolic SAR ===")
    print(f"Datum        : {last['date_str']}")
    print(f"High         : {current_high:.6f}")
    print(f"Low          : {current_low:.6f}")
    print(f"Close        : {current_close:.6f}")
    print(f"PSAR         : {current_psar:.6f}")

    if current_close > current_psar:
        print("✅ Signal: BULLISH")
        print("Begründung   : Der Schlusskurs liegt über dem Parabolic SAR.")
        print("📈 Empfehlung: KAUFEN")
    elif current_close < current_psar:
        print("⚠️ Signal: BEARISH")
        print("Begründung   : Der Schlusskurs liegt unter dem Parabolic SAR.")
        print("📉 Empfehlung: VERKAUFEN")
    else:
        print("➖ Signal: NEUTRAL")
        print("Begründung   : Schlusskurs und Parabolic SAR liegen auf gleichem Niveau.")
        print("🤝 Empfehlung: HALTEN")

    print("\nLetzte 5 Werte:")
    print(df[["date_str", "high", "low", "close", "psar"]].tail(5))

    # --------------------------------------------------
    # Ziel & Prognose
    # --------------------------------------------------
    df["direction"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # bullish wenn close über PSAR
    df["pred_up"] = (df["close"] > df["psar"]).astype(int)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    eval_df = df.dropna(subset=["psar", "direction"]).copy()

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
        ["time", "date_str", "high", "low", "close", "psar", "direction", "pred_up"]
    ].sort_values("time", ascending=False).copy()

    metrics_df = pd.DataFrame({
        "metric": [
            "step",
            "max_step",
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
            step,
            max_step,
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

    out_path = base_dir / f"{db_path.stem}_PSAR_BACKTEST.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="psar_backtest", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)

    print("\n✅ Backtest abgeschlossen")
    print(f"📁 Datei erstellt: {out_path}")


if __name__ == "__main__":
    main()