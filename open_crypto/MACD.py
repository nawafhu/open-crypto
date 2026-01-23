#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MACD Tool
- sucht SQLite-Datenbanken im gleichen Ordner wie dieses Skript
- fragt interaktiv, welche DB verwendet werden soll
- erkennt time-Format (ms / s / days) automatisch
- berechnet MACD (EMA fast/slow), Signal-Linie, Histogramm
- exportiert ALLE Werte nach Excel (neueste oben) inkl. Datum dd.mm.yyyy
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
# Hauptprogramm
# --------------------------------------------------
def main():
    print("=== MACD Tool ===")

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
        signal = int(input("MACD Signal EMA (z.B. 9)  [Enter=9]: ").strip() or "9")
        if fast < 1 or slow < 1 or signal < 1 or fast >= slow:
            raise ValueError
    except ValueError:
        print("Ungültige Parameter. Nutze Standard: fast=12, slow=26, signal=9.")
        fast, slow, signal = 12, 26, 9

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

    if len(df) < slow + signal + 10:
        print("Nicht genügend Daten für MACD.")
        return

    # Datum
    df["date"] = time_to_datetime(df["time"])
    df["date_str"] = df["date"].dt.strftime("%d.%m.%Y")

    # Chronologisch sortieren (wichtig!)
    df = df.sort_values("time").reset_index(drop=True)

    # MACD berechnen
    macd_df = compute_macd(df["close"], fast=fast, slow=slow, signal=signal)
    df = pd.concat([df, macd_df], axis=1)

    # Letzte 10 anzeigen (neueste oben)
    preview = df[["date_str", "close", "macd", "signal", "hist"]].tail(10).sort_values("date_str", ascending=False)
    print("\nLetzte 10 Werte (neueste oben):")
    print(preview)

    # Export (neueste oben)
    df_export = df.sort_values("date", ascending=False).copy()

    out_name = f"{db_path.stem}_MACD_{fast}_{slow}_{signal}.xlsx"
    out_path = base_dir / out_name

    export_cols = ["time", "date_str", "close", "ema_fast", "ema_slow", "macd", "signal", "hist"]
    df_export[export_cols].to_excel(out_path, index=False)

    print(f"\n✅ Excel exportiert: {out_path}")


if __name__ == "__main__":
    main()