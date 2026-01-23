#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSI Tool
- sucht SQLite-Datenbanken im gleichen Ordner wie dieses Skript
- fragt interaktiv, welche DB verwendet werden soll
- erkennt das Zeitintervall anhand der Abstände
- erkennt das time-Format (ms / s / days) automatisch
- fragt RSI-Periode interaktiv ab (Default 21)
- berechnet RSI für ALLE Daten (chronologisch korrekt)
- gibt die letzten 20 Werte aus (neueste oben)
- exportiert ALLE Werte nach Excel (neueste oben) inkl. Datum dd.mm.yyyy
"""

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# RSI-Berechnung (für alle Daten)
# --------------------------------------------------
def compute_rsi(series: pd.Series, period: int = 21) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


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

    # Heuristik:
    # ms timestamps ~ 1e12
    # sec timestamps ~ 1e9
    # days since 1970 ~ 20_000
    if mx > 1e11:
        dt = pd.to_datetime(s, unit="ms", utc=True)
    elif mx > 1e8:
        dt = pd.to_datetime(s, unit="s", utc=True)
    else:
        dt = pd.to_datetime("1970-01-01", utc=True) + pd.to_timedelta(s, unit="D")

    return dt.dt.tz_convert(None)


# --------------------------------------------------
# Hauptprogramm
# --------------------------------------------------
def main():
    print("=== RSI Tool ===")

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

    db_path = db_files[choice - 1]
    print(f"\nVerbinde mit Datenbank: {db_path.name}")

    conn = sqlite3.connect(db_path)

    query = """
        SELECT time, close
        FROM historic_rates_view
        WHERE close IS NOT NULL
        ORDER BY time
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if len(df) < period + 2:
        print(f"Nicht genügend Daten für RSI (brauche mindestens ~{period+2} Zeilen).")
        return

    # Intervall erkennen
    median_step, interval_name = detect_interval(df["time"])

    # time -> date (für Anzeige/Export)
    df["date"] = time_to_datetime(df["time"])
    df["date_str"] = df["date"].dt.strftime("%d.%m.%Y")

    # RSI berechnen (chronologisch korrekt!)
    df["RSI"] = compute_rsi(df["close"], period=period)

    print("\n📏 Erkanntes Zeitintervall:")
    print(f"Median Δt: {median_step}  →  {interval_name}")

    # Für Ausgabe & Export: neuestes Datum nach oben
    df = df.sort_values("date", ascending=False)

    print(f"\nLetzte 20 RSI-Werte (Periode={period}) (neueste oben):")
    print(df.head(20)[["date_str", "close", "RSI"]])

    last_rsi = df.iloc[0]["RSI"]  # neuester RSI steht jetzt oben
    print(f"\nAktueller RSI (neuester): {last_rsi:.2f}")

    if last_rsi > 70:
        print("⚠️ Markt überkauft")
    elif last_rsi < 30:
        print("⚠️ Markt überverkauft")
    else:
        print("✅ Markt neutral")

    # Excel Export (ALLE Werte, neueste oben)
    out_name = f"{db_path.stem}_RSI_{period}.xlsx"
    out_path = base_dir / out_name

    export_cols = ["time", "date_str", "close", "RSI"]
    df[export_cols].to_excel(out_path, index=False)

    print(f"\n✅ Excel exportiert: {out_path}")


if __name__ == "__main__":
    main()