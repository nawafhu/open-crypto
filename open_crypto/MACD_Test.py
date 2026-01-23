#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MACD Tool + Backtest
- DB auswählen
- time-Format (ms/s/days) erkennen
- MACD (fast/slow/signal) berechnen
- Prognose-Regel: MACD > Signal => Up
- Test: trifft das die Richtung morgen? (close[t+1] > close[t])
- Export nach Excel: macd_backtest + metrics
"""

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# time -> datetime erkennen (ms / s / days)
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
# MACD berechnen
# --------------------------------------------------
def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "macd": macd_line,
        "signal": signal_line,
        "hist": hist
    })


# --------------------------------------------------
# Metrics (ohne sklearn)
# --------------------------------------------------
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def roc_auc_score_simple(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Einfacher ROC-AUC über Rank-Statistik (Mann–Whitney U).
    Falls nur eine Klasse vorhanden: NaN.
    """
    y = y_true.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")

    scores = np.asarray(scores, dtype=float)
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Tie handling (durchschnittsrang)
    # Für gleiche Werte: Mittelwert der Ränge
    df_tmp = pd.DataFrame({"s": scores, "r": ranks})
    df_tmp["r_mean"] = df_tmp.groupby("s")["r"].transform("mean")
    ranks = df_tmp["r_mean"].to_numpy()

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


# --------------------------------------------------
# Hauptprogramm
# --------------------------------------------------
def main():
    print("=== MACD Tool + Backtest ===")

    base_dir = Path(__file__).resolve().parent
    db_files = sorted(base_dir.glob("*.db"))

    if not db_files:
        print("Keine SQLite-Datenbanken im Skriptordner gefunden.")
        return

    print("\nGefundene Datenbanken:")
    for i, db in enumerate(db_files, start=1):
        print(f"{i}: {db.name}")

    try:
        choice = int(input("\nNummer der Datenbank auswählen: ").strip())
        if choice < 1 or choice > len(db_files):
            raise ValueError
    except ValueError:
        print("Ungültige Auswahl.")
        return

    # MACD-Parameter abfragen
    try:
        fast = int(input("MACD fast EMA (z.B. 12) [Enter=12]: ").strip() or "12")
        slow = int(input("MACD slow EMA (z.B. 26) [Enter=26]: ").strip() or "26")
        sig = int(input("MACD Signal EMA (z.B. 9)  [Enter=9]: ").strip() or "9")
        if fast < 1 or slow < 1 or sig < 1 or fast >= slow:
            raise ValueError
    except ValueError:
        print("Ungültige Parameter. Nutze Standard: fast=12, slow=26, signal=9.")
        fast, slow, sig = 12, 26, 9

    db_path = db_files[choice - 1]
    print(f"\nVerbinde mit Datenbank: {db_path.name}")

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

    if len(df) < slow + sig + 50:
        print("Nicht genügend Daten für MACD + Backtest.")
        return

    # Datum
    df["date"] = time_to_datetime(df["time"])
    df["date_str"] = df["date"].dt.strftime("%d.%m.%Y")

    # Chronologisch sortieren (wichtig!)
    df = df.sort_values("time").reset_index(drop=True)

    # MACD
    macd_df = compute_macd(df["close"], fast=fast, slow=slow, signal=sig)
    df = pd.concat([df, macd_df], axis=1)

    # ----------------------------
    # Backtest: Prognose vs Wahrheit
    # ----------------------------
    # Wahrheit: steigt morgen?
    df["direction"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Prognose-Regel:
    # bullish wenn macd > signal
    df["pred_up"] = (df["macd"] > df["signal"]).astype(int)

    # Score für AUC: Histogramm (je größer, desto bullish)
    df["score"] = df["hist"].astype(float)

    # Für saubere Auswertung: letzte Zeile hat direction=NaN (kein nächster Tag)
    eval_df = df.dropna(subset=["macd", "signal", "hist", "direction"]).copy()

    y_true = eval_df["direction"].to_numpy(dtype=int)
    y_pred = eval_df["pred_up"].to_numpy(dtype=int)
    scores = eval_df["score"].to_numpy(dtype=float)

    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    precision_up = safe_div(tp, tp + fp)      # wenn wir "Up" sagen: wie oft stimmt es?
    recall_up = safe_div(tp, tp + fn)         # von allen echten Ups: wie viele treffen wir?
    specificity = safe_div(tn, tn + fp)       # von allen echten Downs: wie viele treffen wir?
    f1_up = safe_div(2 * precision_up * recall_up, precision_up + recall_up)
    auc = roc_auc_score_simple(y_true, scores)

    print("\n=== MACD Backtest (Regel: macd > signal => Up) ===")
    print(f"Rows evaluated: {len(eval_df)}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision Up : {precision_up:.4f}")
    print(f"Recall Up    : {recall_up:.4f}")
    print(f"F1 Up        : {f1_up:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print(f"ROC-AUC (hist score): {auc:.4f}" if not np.isnan(auc) else "ROC-AUC: n/a (nur 1 Klasse)")

    # Export-DF: neueste oben
    export_cols = [
        "time", "date_str", "close",
        "ema_fast", "ema_slow", "macd", "signal", "hist",
        "direction", "pred_up", "score"
    ]
    export_df = df[export_cols].sort_values("date_str", ascending=False).copy()

    metrics_df = pd.DataFrame({
        "metric": [
            "fast", "slow", "signal",
            "rows_evaluated",
            "accuracy", "precision_up", "recall_up", "f1_up", "specificity",
            "roc_auc_hist",
            "TN", "FP", "FN", "TP"
        ],
        "value": [
            fast, slow, sig,
            len(eval_df),
            acc, precision_up, recall_up, f1_up, specificity,
            auc,
            tn, fp, fn, tp
        ]
    })

    out_name = f"{db_path.stem}_MACD_BACKTEST_{fast}_{slow}_{sig}.xlsx"
    out_path = base_dir / out_name

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="macd_backtest", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)

    print(f"\n✅ Excel exportiert: {out_path}")


if __name__ == "__main__":
    main()