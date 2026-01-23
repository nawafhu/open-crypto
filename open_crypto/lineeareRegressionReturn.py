#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linear Regression Tool (Return Forecast) + SMA-RSI in Excel
- sucht SQLite-Datenbanken im gleichen Ordner wie dieses Skript
- fragt interaktiv, welche DB verwendet werden soll
- erkennt time-Format (ms / s / days) automatisch
- berechnet SMA-RSI (gleitender Durchschnitt)
- Ziel: y = log(close[t+1]/close[t]) (1-Tages-Log-Return, also "morgen-Return")
- Time-Split Train/Test (kein Shuffle)
- speichert Ergebnisse in Excel inkl. RSI-Spalte im predictions-Sheet
"""

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
# RSI mit gleitendem Durchschnitt (SMA)
# --------------------------------------------------
def compute_rsi_sma(series: pd.Series, period: int = 21) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def time_split(df: pd.DataFrame, test_size: float):
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def main():
    print("=== Linear Regression Tool (Return Forecast) ===")

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

    # RSI-Periode abfragen
    try:
        rsi_period = int(input("RSI-Periode (z.B. 21) [Enter = 21]: ").strip() or "21")
        if rsi_period < 2:
            raise ValueError
    except ValueError:
        print("Ungültige Periode. Nutze 21.")
        rsi_period = 21

    # Test-Anteil abfragen
    try:
        test_size = float(input("Test-Anteil (z.B. 0.2) [Enter = 0.2]: ").strip() or "0.2")
        if not (0.05 <= test_size <= 0.5):
            raise ValueError
    except ValueError:
        print("Ungültiger Test-Anteil. Nutze 0.2.")
        test_size = 0.2

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

    if len(df) < 250:
        print("Zu wenige Daten (empfohlen: >= 250 Zeilen).")
        return

    # Datum
    df["date"] = time_to_datetime(df["time"])
    df["date_str"] = df["date"].dt.strftime("%d.%m.%Y")

    # Chronologisch sortieren (wichtig!)
    df = df.sort_values("time").reset_index(drop=True)

    # Features
    df["ret_1"] = np.log(df["close"] / df["close"].shift(1))
    df["ret_5"] = np.log(df["close"] / df["close"].shift(5))
    df["vol_20"] = df["ret_1"].rolling(20).std()

    rsi_col = f"rsi_{rsi_period}"
    df[rsi_col] = compute_rsi_sma(df["close"], period=rsi_period)

    df["sma_20"] = df["close"].rolling(20).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    feature_cols = ["ret_1", "ret_5", "vol_20", rsi_col, "sma_20", "ema_20"]

    # Zielvariable: 1-Tages-Log-Return von heute -> morgen
    df["y_return_1d"] = np.log(df["close"].shift(-1) / df["close"])

    df_model = df.dropna(subset=feature_cols + ["y_return_1d"]).copy()
    if len(df_model) < 200:
        print("Nach DropNA sind zu wenige Zeilen übrig.")
        return

    # Time-Split
    train_df, test_df = time_split(df_model, test_size=test_size)

    X_train = train_df[feature_cols].values
    y_train = train_df["y_return_1d"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["y_return_1d"].values

    # Modell trainieren
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Vorhersagen
    y_pred = reg.predict(X_test)

    # Metriken
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Ergebnisse (Test) ===")
    print(f"RMSE: {rmse:.8f}")
    print(f"MAE : {mae:.8f}")
    print(f"R²  : {r2:.6f}")

    # Predictions Sheet (inkl. RSI)
    pred_out = test_df[["date", "date_str", "close", rsi_col]].copy()
    pred_out = pred_out.rename(columns={rsi_col: f"RSI_{rsi_period}"})
    pred_out["y_return_1d"] = y_test
    pred_out["y_pred_return_1d"] = y_pred
    pred_out["error"] = pred_out["y_pred_return_1d"] - pred_out["y_return_1d"]

    # Neueste oben
    pred_out = pred_out.sort_values("date", ascending=False)

    # Metrics Sheet
    metrics_df = pd.DataFrame({
        "metric": ["rmse", "mae", "r2", "test_size_frac", "train_size", "test_size"],
        "value": [rmse, mae, r2, test_size, len(train_df), len(test_df)]
    })

    # Coefficients Sheet
    coef_df = pd.DataFrame({"feature": feature_cols, "coef": reg.coef_}).sort_values("coef", ascending=False)
    intercept_df = pd.DataFrame({"intercept": [float(reg.intercept_)]})

    # Excel Export
    out_name = f"{db_path.stem}_LINREG_SMA_RSI{rsi_period}.xlsx"
    out_path = base_dir / out_name

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pred_out.to_excel(writer, sheet_name="predictions", index=False)
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        coef_df.to_excel(writer, sheet_name="coefficients", index=False)
        intercept_df.to_excel(writer, sheet_name="intercept", index=False)

    print(f"\n✅ Excel exportiert: {out_path}")


if __name__ == "__main__":
    main()