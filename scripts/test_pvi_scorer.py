#!/usr/bin/env python
"""Selftest mandiri untuk ``uvip_ai.scoring.pvi_scorer``.

Tidak butuh model ML, tidak butuh dataset - semua fixture ditanam.
Jalankan:  python scripts/test_pvi_scorer.py

Yang diuji:
  1. 5 foto acuan -> skor HARUS persis sama dengan tabel keputusan
  2. VDI > 1 di-clamp (bug #1)
  3. Tidak ada skor yang menyentuh 10 kecuali foto terbaik
  4. Kontrak output: 4 key, tipe float, rentang sah
  5. Ketahanan input kosong / None / nilai string
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from uvip_ai.scoring.pvi_scorer import (  # noqa: E402
    CALIBRATION,
    PVI_WEIGHTS,
    compute_pvi,
    compute_scores,
    compute_scores_from_metrics,
)

# ---------------------------------------------------------------------------
# Fixture: nilai indeks nyata (dari infer(excel_mode=True)) + hasil harapan
# ---------------------------------------------------------------------------
REFERENCE = [
    (
        "ALUN_ALUN_MERDEKA_BS-01",
        {
            "building_visibility_index": 0.2792,
            "vegetation_coverage_index": 0.3477,
            "sky_openness_index": 0.0223,
            "ground_accessibility_index": 0.3104,
            "human_activity_index": 0.0,
            "vehicle_intensity_index": 0.5869,
            "traffic_infrastructure_index": 0.0192,
            "heritage_dominance_index": 0.0,
        },
        (4.19, 7.30, 4.86, 0.53),
    ),
    (
        "ALUN_ALUN_MERDEKA_BS-02",
        {
            "building_visibility_index": 0.4368,
            "vegetation_coverage_index": 0.2568,
            "sky_openness_index": 0.0324,
            "ground_accessibility_index": 0.2589,
            "human_activity_index": 0.0,
            "vehicle_intensity_index": 0.3176,
            "traffic_infrastructure_index": 0.0066,
            "heritage_dominance_index": 0.0,
        },
        (2.85, 5.72, 3.53, 0.39),
    ),
    (
        "ALUN_ALUN_MERDEKA_BS-03",
        {
            "building_visibility_index": 0.3354,
            "vegetation_coverage_index": 0.1888,
            "sky_openness_index": 0.1174,
            "ground_accessibility_index": 0.2955,
            "human_activity_index": 0.0,
            "vehicle_intensity_index": 1.4273,  # > 1 -> di-clamp
            "traffic_infrastructure_index": 0.0264,
            "heritage_dominance_index": 0.0,
        },
        (7.39, 7.03, 3.81, 0.62),
    ),
    (
        "ALUN_ALUN_MERDEKA_BU-05",
        {
            "building_visibility_index": 0.4926,
            "vegetation_coverage_index": 0.2686,
            "sky_openness_index": 0.031,
            "ground_accessibility_index": 0.1454,
            "human_activity_index": 0.0,
            "vehicle_intensity_index": 0.9763,
            "traffic_infrastructure_index": 0.0243,
            "heritage_dominance_index": 0.0,
        },
        (5.98, 3.04, 3.70, 0.44),
    ),
    (
        "ALUN_ALUN_MERDEKA_UB-05",
        {
            "building_visibility_index": 0.3527,
            "vegetation_coverage_index": 0.3322,
            "sky_openness_index": 0.0514,
            "ground_accessibility_index": 0.1818,
            "human_activity_index": 0.0,
            "vehicle_intensity_index": 0.1355,
            "traffic_infrastructure_index": 0.0614,
            "heritage_dominance_index": 0.0,
        },
        (2.08, 4.65, 5.08, 0.38),
    ),
]

FIELDS = ("beauty_score", "safety_score", "comfort_score", "uvi_score")

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [OK]   {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}" + (f" -> {detail}" if detail else ""))


def main() -> int:
    print("=" * 72)
    print("SELFTEST uvip_ai.scoring.pvi_scorer")
    print("=" * 72)

    # -- 1. Foto acuan ------------------------------------------------------
    print("\n[1] 5 foto acuan (harus persis sesuai tabel keputusan)")
    for name, metrics, expected in REFERENCE:
        s = compute_scores(metrics)
        got = tuple(s[f] for f in FIELDS)
        check(f"{name:26s} {got}", got == expected,
              f"expected {expected}, got {got}")

    # -- 2. Clamp VDI -------------------------------------------------------
    print("\n[2] VDI di-clamp ke [0, 1] (bug #1)")
    base = dict(REFERENCE[2][1])
    vdi_over = dict(base, vehicle_intensity_index=1.4273)
    vdi_one = dict(base, vehicle_intensity_index=1.0)
    check("VDI=1.4273 == VDI=1.0 (clamp bekerja)",
          compute_pvi(vdi_over) == compute_pvi(vdi_one),
          f"{compute_pvi(vdi_over)} != {compute_pvi(vdi_one)}")
    check("VDI=-0.5 == VDI=0.0 (clamp bawah)",
          compute_pvi(dict(base, vehicle_intensity_index=-0.5))
          == compute_pvi(dict(base, vehicle_intensity_index=0.0)))
    check("VDI>1 tidak menghasilkan skor > 10",
          compute_scores(vdi_over)["beauty_score"] <= 10.0)

    # -- 3. Anti-saturasi ---------------------------------------------------
    print("\n[3] Anti-saturasi (tidak semua 10)")
    scores = [compute_scores(m) for _, m, _ in REFERENCE]
    tens = [s for s in scores if s["beauty_score"] == 10.0]
    check(f"hanya <=1 foto bernilai beauty 10 (dapat {len(tens)})",
          len(tens) <= 1)
    uniq = len({s["beauty_score"] for s in scores})
    check(f"beauty bervariasi (nilai unik = {uniq})", uniq >= 5)
    check("foto terbaik > foto terburuk",
          max(s["beauty_score"] for s in scores)
          > min(s["beauty_score"] for s in scores))

    # -- 4. Kontrak output --------------------------------------------------
    print("\n[4] Kontrak output")
    for name, metrics, _ in REFERENCE:
        s = compute_scores(metrics)
        check(f"{name:26s} 4 key + tipe float",
              all(f in s for f in FIELDS)
              and all(isinstance(s[f], float) for f in FIELDS))
        check(f"{name:26s} beauty/safety/comfort di [1,10]",
              1.0 <= s["beauty_score"] <= 10.0
              and 1.0 <= s["safety_score"] <= 10.0
              and 1.0 <= s["comfort_score"] <= 10.0)
        check(f"{name:26s} uvi di [0,1]", 0.0 <= s["uvi_score"] <= 1.0)
        check(f"{name:26s} uvi = 0.4b+0.3s+0.3c",
              abs(s["uvi_score"] - round(
                  (s["beauty_score"] / 10) * 0.4
                  + (s["safety_score"] / 10) * 0.3
                  + (s["comfort_score"] / 10) * 0.3, 2)) <= 0.01)

    check("compute_scores_from_metrics tanpa _raw",
          "_raw" not in compute_scores_from_metrics(REFERENCE[0][1]))
    check("compute_scores menyertakan _raw untuk audit",
          "_raw" in compute_scores(REFERENCE[0][1]))

    # -- 5. Ketahanan input -------------------------------------------------
    print("\n[5] Ketahanan input")
    empty = compute_scores({})
    check("input {} -> tetap 4 field, tidak error",
          all(f in empty for f in FIELDS))
    check("input {} -> skor di rentang sah",
          1.0 <= empty["beauty_score"] <= 10.0)
    none_ok = compute_scores({"building_visibility_index": None})
    check("input None -> tidak error", all(f in none_ok for f in FIELDS))
    str_ok = compute_scores({"sky_openness_index": "0.05"})
    check("input string numerik -> tidak error",
          all(f in str_ok for f in FIELDS))
    bad_ok = compute_scores({"sky_openness_index": "abc"})
    check("input string non-numerik -> fallback 0, tidak error",
          all(f in bad_ok for f in FIELDS))

    # -- 6. Konsistensi kalibrasi ------------------------------------------
    print("\n[6] Konsistensi anchor kalibrasi")
    for key, anchor in CALIBRATION.items():
        check(f"anchor {key:12s} max > min",
              anchor["max"] > anchor["min"])
    check("bobot PVI berjumlah 1.0",
          abs(sum(PVI_WEIGHTS.values()) - 1.0) < 1e-9,
          f"sum={sum(PVI_WEIGHTS.values())}")

    print("\n" + "=" * 72)
    print(f"HASIL: {_passed} lulus, {_failed} gagal")
    print("=" * 72)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
