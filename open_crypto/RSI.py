#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSI Tool + Backtest Metrics
- sucht SQLite-Datenbanken im gleichen Ordner wie dieses Skript
- fragt interaktiv, welche DB verwendet werden soll
- erkennt das Zeitintervall anhand der Abstände
- erkennt das time-Format (ms / s / days) automatisch
- fragt RSI-Periode interaktiv ab (Default 21)
- berechnet RSI für ALLE Daten (chronologisch korrekt)
- erstellt ein RSI-Signal und testet es gegen "morgen steigt ja/nein"
- berechnet Metrics: Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, Confusion Matrix
- exportiert alles nach Excel (neueste oben)
"""

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# RSI-Berechnung (SMA)
# --------------------------------------------------
def compute_rsi(series: pd.Series, period: int = 21) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# --------------------------------------------------
# Zeitintervall erkennen (robust, Einheit egal)
# --------------------------------------------------
def detect_interval(time_series: pd.Series) -> tuple:
    diffs = time_series.sort_values().diff().dropna()
    if diffs.empty:
        return None, "unbekannt"
    median = float(diffs.median())

    candidates = [
        (1.0, "1 Tag (days)"),
        (86_400.0, "1 Tag (seconds)"),
        (86_400_000.0, "1 Tag (milliseconds)"),
        (3_600.0, "1 Stunde (seconds)"),
        (3_600_000.0, "1 Stunde (milliseconds)"),
        (60.0, "1 Minute (seconds)"),
        (60_000.0, "1 Minute (milliseconds)"),
    ]
    closest = min(candidates, key=lambda x: abs(x[0] - median))
    return median, closest[1]


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
# Metriken ohne sklearn (robust, keine Abhängigkeit)
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


def roc_auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    ROC-AUC über Rangstatistik (Mann–Whitney U).
    Wenn nur eine Klasse vorhanden -> NaN.
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
    print("=== RSI Tool + Backtest Metrics ===")

    base_dir = Path(__file__).resolve().parent
    db_files = sorted(base_dir.glob("*.db"))

    if not db_files:
        print("Keine SQLite-Datenbanken im Skriptordner gefunden.")
        return

    print("\nGefundene Datenbanken:")
    for i, db in enumerate(db_files, start=1):
        print(f"{i}: {db.name}")

    try:
        choice = int(input("\nNummer der Datenbank auswählen: "))
        if choice < 1 or choice > len(db_files):
            raise ValueError
    except ValueError:
        print("Ungültige Auswahl.")
        return

    # RSI-Periode abfragen (Default 21)
    try:
        period = int(input("RSI-Periode (z.B. 21) [Enter = 21]: ").strip() or "21")
        if period < 2:
            raise ValueError
    except ValueError:
        print("Ungültige Periode. Nutze 21.")
        period = 21

    # Buy-Schwelle für Signal (Default 30)
    try:
        buy_th = float(input("RSI Buy-Schwelle (z.B. 30) [Enter = 30]: ").strip() or "30")
    except ValueError:
        buy_th = 30.0

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

    if len(df) < period + 10:
        print(f"Nicht genügend Daten für RSI (brauche mindestens ~{period+10} Zeilen).")
        return

    # Intervall erkennen
    median_step, interval_name = detect_interval(df["time"])

    # time -> date
    df["date"] = time_to_datetime(df["time"])
    df["date_str"] = df["date"].dt.strftime("%d.%m.%Y")

    # Chronologisch sortieren (wichtig für Features & Ziel)
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    # RSI berechnen
    df["RSI"] = compute_rsi(df["close"], period=period)

    # Zielvariable: morgen steigt?
    df["y"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # RSI Signal: "überverkauft" => morgen eher up
    df["pred_up"] = (df["RSI"] < buy_th).astype(int)

    # Score für ROC-AUC: je niedriger RSI, desto bullischer
    # Daher score = buy_th - RSI (größer = bullischer)
    df["score"] = (buy_th - df["RSI"]).astype(float)

    # Für Backtest: nur Zeilen ohne NaNs und ohne letzte Zeile (hat kein y sinnvoll)
    eval_df = df.dropna(subset=["RSI", "y"]).copy()

    y_true = eval_df["y"].to_numpy(dtype=int)
    y_pred = eval_df["pred_up"].to_numpy(dtype=int)
    scores = eval_df["score"].to_numpy(dtype=float)

    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    specificity = safe_div(tn, tn + fp)
    auc = roc_auc_rank(y_true, scores)

    signal_rate = safe_div((y_pred == 1).sum(), len(y_pred))

    print("\n📏 Erkanntes Zeitintervall:")
    print(f"Median Δt: {median_step}  →  {interval_name}")

    # Ausgabe neueste oben
    df_show = df.sort_values("date", ascending=False)
    print(f"\nLetzte 20 RSI-Werte (Periode={period}) (neueste oben):")
    print(df_show.head(20)[["date_str", "close", "RSI"]])

    last_rsi = float(df_show.iloc[0]["RSI"])
    print(f"\nAktueller RSI (neuester): {last_rsi:.2f}")

    if last_rsi > 70:
        print("⚠️ Markt überkauft")
    elif last_rsi < 30:
        print("⚠️ Markt überverkauft")
    else:
        print("✅ Markt neutral")

    print("\n=== Backtest Metrics (Signal: RSI < Schwelle => Up) ===")
    print(f"RSI Periode    : {period}")
    print(f"Buy Schwelle   : {buy_th}")
    print(f"Signal Rate    : {signal_rate:.4f}")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision (Up) : {precision:.4f}")
    print(f"Recall (Up)    : {recall:.4f}")
    print(f"F1 (Up)        : {f1:.4f}")
    print(f"Specificity    : {specificity:.4f}")
    print(f"ROC-AUC        : {auc:.4f}" if not np.isnan(auc) else "ROC-AUC        : n/a")
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

    # --------------------------------------------------
    # Excel Export
    # --------------------------------------------------
    out_name = f"{db_path.stem}_RSI_{period}_BUY{int(buy_th)}_BACKTEST.xlsx"
    out_path = base_dir / out_name

    # Für Export: neueste oben
    rsi_values = df.sort_values("date", ascending=False)[["time", "date_str", "close", "RSI"]].copy()

    backtest = df.sort_values("date", ascending=False)[
        ["time", "date_str", "close", "RSI", "y", "pred_up", "score"]
    ].copy()

    metrics_df = pd.DataFrame({
        "metric": [
            "rsi_period", "buy_threshold",
            "rows_evaluated", "signal_rate",
            "accuracy", "precision_up", "recall_up", "f1_up", "specificity",
            "roc_auc",
            "TN", "FP", "FN", "TP"
        ],
        "value": [
            period, buy_th,
            len(eval_df), signal_rate,
            acc, precision, recall, f1, specificity,
            auc,
            tn, fp, fn, tp
        ]
    })

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        rsi_values.to_excel(writer, sheet_name="rsi_values", index=False)
        backtest.to_excel(writer, sheet_name="backtest", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)

    print(f"\n✅ Excel exportiert: {out_path}")


if __name__ == "__main__":
    main()