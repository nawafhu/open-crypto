#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DB → Excel Export Tool
- sucht SQLite-Datenbanken im gleichen Ordner
- fragt interaktiv nach Tabelle
- optionaler Zeitraum-Filter
- exportiert alles nach Excel
"""

import sqlite3
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# time -> datetime (ms / s / days automatisch)
# --------------------------------------------------
def time_to_datetime(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mx = s.dropna().max()

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
    print("=== DB → Excel Export Tool ===")

    base_dir = Path(__file__).resolve().parent
    db_files = sorted(base_dir.glob("*.db"))

    if not db_files:
        print("Keine .db Dateien gefunden.")
        return

    print("\nGefundene Datenbanken:")
    for i, db in enumerate(db_files, 1):
        print(f"{i}: {db.name}")

    try:
        choice = int(input("\nNummer der Datenbank auswählen: "))
        db_path = db_files[choice - 1]
    except:
        print("Ungültige Auswahl.")
        return

    conn = sqlite3.connect(db_path)

    # Tabellen anzeigen
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    )

    if tables.empty:
        print("Keine Tabellen gefunden.")
        conn.close()
        return

    print("\nVerfügbare Tabellen:")
    for i, name in enumerate(tables["name"], 1):
        print(f"{i}: {name}")

    try:
        table_choice = int(input("\nNummer der Tabelle auswählen: "))
        table_name = tables["name"].iloc[table_choice - 1]
    except:
        print("Ungültige Auswahl.")
        conn.close()
        return

    print(f"\nLade Tabelle: {table_name}")

    # Zeitraum optional
    date_filter = input("Zeitraum filtern? (j/n) [Enter = n]: ").strip().lower()

    if date_filter == "j":
        from_date = input("Startdatum (YYYY-MM-DD): ")
        to_date = input("Enddatum   (YYYY-MM-DD): ")

        query = f"""
        SELECT *
        FROM {table_name}
        WHERE time IS NOT NULL
        """
        df = pd.read_sql(query, conn)

        # time -> datetime
        if "time" in df.columns:
            df["date"] = time_to_datetime(df["time"])
            df = df[(df["date"] >= from_date) & (df["date"] <= to_date)]

    else:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

        if "time" in df.columns:
            df["date"] = time_to_datetime(df["time"])

    conn.close()

    if df.empty:
        print("Keine Daten gefunden.")
        return

    # Neueste oben sortieren (falls date existiert)
    if "date" in df.columns:
        df = df.sort_values("date", ascending=False)

    # Excel speichern
    out_name = f"{db_path.stem}_{table_name}.xlsx"
    out_path = base_dir / out_name

    df.to_excel(out_path, index=False)

    print(f"\n✅ Export erfolgreich!")
    print(f"Datei gespeichert unter: {out_path}")
    print(f"Anzahl Zeilen: {len(df)}")


if __name__ == "__main__":
    main()