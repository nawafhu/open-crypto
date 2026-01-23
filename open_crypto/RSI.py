#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSI Tool
- sucht SQLite-Datenbanken im gleichen Ordner wie dieses Skript
- fragt interaktiv, welche DB verwendet werden soll
- erkennt das Zeitintervall anhand der ms-Abstände
- berechnet RSI auf Basis der historic_rates_view
"""

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# RSI-Berechnung
# --------------------------------------------------
def compute_rsi(series: pd.Series, period: int = 21) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# --------------------------------------------------
# Zeitintervall erkennen
# --------------------------------------------------
def detect_interval_ms(time_series: pd.Series) -> tuple:
    diffs = time_series.sort_values().diff().dropna()
    median_ms = int(diffs.median())

    intervals = {
        60_000: "1 Minute",
        300_000: "5 Minuten",
        900_000: "15 Minuten",
        3_600_000: "1 Stunde",
        86_400_000: "1 Tag"
    }

    closest = min(intervals.keys(), key=lambda x: abs(x - median_ms))
    return median_ms, intervals[closest]


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

    if len(df) < 20:
        print("Nicht genügend Daten für RSI.")
        return

    # Zeitintervall erkennen
    median_ms, interval_name = detect_interval_ms(df["time"])

    # RSI berechnen
    df["RSI"] = compute_rsi(df["close"])

    print("\n📏 Erkanntes Zeitintervall:")
    print(f"Median Δt: {median_ms} ms  →  {interval_name}")

    print("\nLetzte 20 RSI-Werte:")
    print(df.tail(20)[["time", "close", "RSI"]])

    last_rsi = df["RSI"].iloc[-1]

    print(f"\nAktueller RSI: {last_rsi:.2f}")
    if last_rsi > 70:
        print("⚠️ Markt überkauft")
    elif last_rsi < 30:
        print("⚠️ Markt überverkauft")
    else:
        print("✅ Markt neutral")


# --------------------------------------------------
# Wichtig für CLI & >>> Nutzung
# --------------------------------------------------
if __name__ == "__main__":
    main()