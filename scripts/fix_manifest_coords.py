#!/usr/bin/env python
"""
Fix koordinat manifest: flip lat positif ke negatif (Malang) + flag outliers lng.
Output: data/extracted/manifest_clean.csv + notes CSV.
"""
from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    inp = Path("data/extracted/manifest.csv")
    out = Path("data/extracted/manifest_clean.csv")
    rows = list(csv.DictReader(inp.open()))

    cleaned = []
    for r in rows:
        lat_str = r.get("lat","").strip()
        lng_str = r.get("long","").strip()
        lat = float(lat_str) if lat_str else None
        lng = float(lng_str) if lng_str else None

        # Flip positive lat to negative (should all be ~ -7.98 in Malang)
        if lat is not None and lat > 0:
            lat = -abs(lat)
        # Outlier lng < 112.3 → mark as outlier; keep value but add flag
        flag = ""
        if lng is not None and lng < 112.5:
            flag = "OUTLIER_LNG"
        if flag:
            note = {**r, "flag": flag}
            cleaned.append(note)
            continue

        row = {**r, "lat": f"{float(lat):.8f}" if lat is not None else "",
                "long": f"{lng:.8f}" if lng is not None else "", "flag": ""}
        cleaned.append(row)

    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames=list(next(iter(cleaned)).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned)

    n_outlier = sum(1 for r in cleaned if r["flag"])
    print("\nFixed:", len(cleaned), "rows ->", out)
    print("Outlier lng flagged:", n_outlier, "(review manually)")


if __name__ == "__main__":
    main()
