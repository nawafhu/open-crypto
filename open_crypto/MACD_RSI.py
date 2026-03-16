#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MACD + RSI Combo Backtest (RSI = SMA, gleitender Durchschnitt)

Was macht das Skript?
- sucht *.db im gleichen Ordner wie dieses Skript
- DB auswählen
- lädt time, close aus historic_rates_view
- erkennt time-Format (ms/s/days) automatisch
- berechnet MACD (EMA fast/slow, Signal, Histogramm)
- berechnet RSI (SMA) mit frei wählbarer Periode
- Kombi-Regel:
    pred_up = 1 wenn (macd > signal) UND (RSI < rsi_buy_threshold)
    sonst 0
- Wahrheit:
    direction = 1 wenn close[t+1] > close[t] sonst 0
- bewertet: accuracy, precision/recall/f1 (Up), specificity, ROC-AUC (Score)
- exportiert nach Excel:
    Sheet 1: combo_backtest
    Sheet 2: metrics

WICHTIG: Sortierung ist korrekt nach Datum (über time).
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
# RSI (SMA) berechnen
# --------------------------------------------------
def compute_rsi_sma(close: pd.Series, period: int = 21) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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
    ROC-AUC via Rank-Statistik (Mann–Whitney U).
    Falls nur eine Klasse: NaN.
    """
    y = y_true.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")

    scores = np.asarray(scores, dtype=float)
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Tie-handling: gleicher Score -> mittlerer Rang
    tmp = pd.DataFrame({"s": scores, "r": ranks})
    tmp["r_mean"] = tmp.groupby("s")["r"].transform("mean")
    ranks = tmp["r_mean"].to_numpy()

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    sum_ranks_pos = float(ranks[y == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


# --------------------------------------------------
# Hauptprogramm
# --------------------------------------------------
def main():
    print("=== MACD + RSI Combo Backtest (RSI = SMA) ===")

    base_dir = Path(__file__).resolve().parent
    db_files = sorted(base_dir.glob("*.db"))

    if not db_files:
        print("Keine SQLite-Datenbanken im Skriptordner gefunden.")
        return

    print("\nGefundene Datenbanken:")
    for i, db in enumerate(db_files, start=1):
        print(f"{i}: {db.name}")

    # DB auswählen
    try:
        choice = int(input("\nNummer der Datenbank auswählen: ").strip())
        if choice < 1 or choice > len(db_files):
            raise ValueError
    except ValueError:
        print("Ungültige Auswahl.")
        return

    # MACD Parameter
    try:
        fast = int(input("MACD fast EMA (z.B. 12) [Enter=12]: ").strip() or "12")
        slow = int(input("MACD slow EMA (z.B. 26) [Enter=26]: ").strip() or "26")
        sig = int(input("MACD Signal EMA (z.B. 9)  [Enter=9]: ").strip() or "9")
        if fast < 1 or slow < 1 or sig < 1 or fast >= slow:
            raise ValueError
    except ValueError:
        print("Ungültige MACD Parameter. Nutze Standard 12/26/9.")
        fast, slow, sig = 12, 26, 9

    # RSI Parameter
    try:
        rsi_period = int(input("RSI-Periode (z.B. 21) [Enter=21]: ").strip() or "21")
        if rsi_period < 2:
            raise ValueError
    except ValueError:
        print("Ungültige RSI-Periode. Nutze 21.")
        rsi_period = 21

    # RSI Buy Threshold
    try:
        rsi_buy = float(input("RSI Kauf-Schwelle (z.B. 30) [Enter=30]: ").strip() or "30")
    except ValueError:
        rsi_buy = 30.0

    db_path = db_files[choice - 1]
    print(f"\nVerbinde mit Datenbank: {db_path.name}")

    # Daten laden
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

    if len(df) < slow + sig + rsi_period + 50:
        print("Nicht genügend Daten für MACD+RSI Backtest.")
        return

    # Datums-Spalten
    df["date"] = time_to_datetime(df["time"])
    df["date_str"] = df["date"].dt.strftime("%d.%m.%Y")

    # Chronologisch sortieren (wichtig!)
    df = df.sort_values("time").reset_index(drop=True)

    # Indikatoren
    macd_df = compute_macd(df["close"], fast=fast, slow=slow, signal=sig)
    df = pd.concat([df, macd_df], axis=1)

    rsi_col = f"RSI_{rsi_period}"
    df[rsi_col] = compute_rsi_sma(df["close"], period=rsi_period)

        # Aktuelle Werte + Empfehlung
    last_row = df.iloc[-1]
    current_macd = float(last_row["macd"])
    current_signal = float(last_row["signal"])
    current_hist = float(last_row["hist"])
    current_rsi = float(last_row[rsi_col])
    current_close = float(last_row["close"])

    print("\n=== Aktuelle Indikatorwerte ===")
    print(f"Datum        : {last_row['date_str']}")
    print(f"Close        : {current_close:.6f}")
    print(f"MACD         : {current_macd:.6f}")
    print(f"Signal-Linie : {current_signal:.6f}")
    print(f"Histogramm   : {current_hist:.6f}")
    print(f"{rsi_col:<13}: {current_rsi:.2f}")

    if current_macd > current_signal and current_rsi < rsi_buy:
        print("✅ Signal: BULLISH")
        print("Begründung   : MACD > Signal und RSI unter Kauf-Schwelle.")
        print("📈 Empfehlung: KAUFEN")
    elif current_macd < current_signal:
        print("⚠️ Signal: BEARISH")
        print("Begründung   : MACD liegt unter der Signallinie.")
        print("📉 Empfehlung: VERKAUFEN")
    else:
        print("➖ Signal: UNEINDEUTIG")
        print("Begründung   : MACD zwar nicht bearish genug für Verkauf, aber kein vollständiges Kaufsignal.")
        print("🤝 Empfehlung: HALTEN")

    # Wahrheit (morgen höher?)
    df["direction"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Kombi-Regel: MACD bullish UND RSI unter Schwelle
    df["pred_up"] = ((df["macd"] > df["signal"]) & (df[rsi_col] < rsi_buy)).astype(int)

    # Score für AUC (einfacher Kombi-Score)
    df["score"] = df["hist"] - 0.1 * (df[rsi_col] - rsi_buy)

    # Auswertung (ohne NaNs)
    eval_df = df.dropna(subset=["macd", "signal", "hist", rsi_col, "direction"]).copy()

    y_true = eval_df["direction"].to_numpy(dtype=int)
    y_pred = eval_df["pred_up"].to_numpy(dtype=int)
    scores = eval_df["score"].to_numpy(dtype=float)

    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    precision_up = safe_div(tp, tp + fp)
    recall_up = safe_div(tp, tp + fn)
    f1_up = safe_div(2 * precision_up * recall_up, precision_up + recall_up)
    specificity = safe_div(tn, tn + fp)
    auc = roc_auc_score_simple(y_true, scores)
    signal_rate = safe_div((y_pred == 1).sum(), len(y_pred))

    print("\n=== Ergebnis ===")
    print(f"rows_evaluated : {len(eval_df)}")
    print(f"signal_rate    : {signal_rate:.4f}")
    print(f"accuracy       : {acc:.4f}")
    print(f"precision_up   : {precision_up:.4f}")
    print(f"recall_up      : {recall_up:.4f}")
    print(f"f1_up          : {f1_up:.4f}")
    print(f"specificity    : {specificity:.4f}")
    print(f"roc_auc_score  : {auc:.4f}" if not np.isnan(auc) else "roc_auc_score  : n/a")
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

    # Export: korrekt nach Datum sortiert (neueste oben) -> sort by time
    export_cols = [
        "time", "date_str", "close",
        "ema_fast", "ema_slow", "macd", "signal", "hist",
        rsi_col,
        "direction", "pred_up", "score"
    ]
    export_df = df[export_cols].sort_values("time", ascending=False).copy()

    metrics_df = pd.DataFrame({
        "metric": [
            "fast", "slow", "signal",
            "rsi_period", "rsi_buy_threshold",
            "rows_evaluated", "signal_rate",
            "accuracy", "precision_up", "recall_up", "f1_up", "specificity",
            "roc_auc_score",
            "TN", "FP", "FN", "TP"
        ],
        "value": [
            fast, slow, sig,
            rsi_period, rsi_buy,
            len(eval_df), signal_rate,
            acc, precision_up, recall_up, f1_up, specificity,
            auc,
            tn, fp, fn, tp
        ]
    })

    out_name = f"{db_path.stem}_MACD_RSI_BACKTEST_{fast}_{slow}_{sig}_RSI{rsi_period}_BUY{int(rsi_buy)}.xlsx"
    out_path = base_dir / out_name

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="combo_backtest", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)

    print(f"\n✅ Excel exportiert: {out_path}")


if __name__ == "__main__":
    main()