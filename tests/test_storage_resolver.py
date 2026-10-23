"""Kontrak resolver artefak -> ``result_*.json``.

Kasus yang dijaga: klien mengirim path overlay/segmentation/mask (bentuk yang
dipakai `/llm/generate`), bukan path JSON; resolver wajib menemukan result
pasangannya, termasuk saat file artefak sudah terhapus dan saat nama result
punya akhiran bentrok ``_2``.
"""

from __future__ import annotations

from uvip_ai.utils import storage


def _dirs(tmp_path, monkeypatch):
    uploads = (tmp_path / "uploads").resolve()
    results = uploads / "results"
    segmentation = uploads / "segmentation"
    results.mkdir(parents=True)
    segmentation.mkdir(parents=True)
    monkeypatch.setattr(storage, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(storage, "RESULTS_DIR", results)
    return results, segmentation


def test_artifact_path_resolves_to_paired_result(tmp_path, monkeypatch):
    results, segmentation = _dirs(tmp_path, monkeypatch)
    stem = "76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4"
    stamp = "1792582425737"
    result = results / f"result_{stem}_{stamp}.json"
    storage.write_json_atomic(result, {"photo_id": "p1"})
    overlay = segmentation / f"overlay_{stem}_{stamp}.jpg"
    overlay.write_bytes(b"overlay")

    got, origin = storage.resolve_result_path(
        f"/uploads/segmentation/{overlay.name}")
    assert got == result
    assert origin.startswith("artifact:")

    # Artefak sudah dihapus, tetapi result pasangannya masih ada.
    overlay.unlink()
    got, _ = storage.resolve_result_path(
        f"/uploads/segmentation/{overlay.name}")
    assert got == result


def test_collision_suffix_maps_to_same_suffix_result(tmp_path, monkeypatch):
    results, segmentation = _dirs(tmp_path, monkeypatch)
    stem = "photo_123"
    stamp = "1792582425737"
    first = results / f"result_{stem}_{stamp}.json"
    second = results / f"result_{stem}_{stamp}_2.json"
    storage.write_json_atomic(first, {"n": 1})
    storage.write_json_atomic(second, {"n": 2})
    overlay = segmentation / f"overlay_{stem}_{stamp}_2.jpg"
    overlay.write_bytes(b"overlay")

    got, _ = storage.resolve_result_path(
        f"/uploads/segmentation/{overlay.name}")
    assert got == second

    # Proses lama menimpa result saat bentrok, jadi ``_2`` hanya ada di
    # artefak; resolver harus jatuh ke result tanpa akhiran.
    second.unlink()
    got, _ = storage.resolve_result_path(
        f"/uploads/segmentation/{overlay.name}")
    assert got == first
