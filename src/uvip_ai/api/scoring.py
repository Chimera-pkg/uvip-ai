"""Router tuning skor: slider/bobot, profil, preview, evaluasi sebaran.

Semua endpoint di sini **tanpa GPU**: skor dihitung dari 8 indeks yang sudah
tersimpan (``data/scoring/pvi_per_photo.csv`` hasil segmentasi 431 foto),
jadi menggeser slider bisa langsung membandingkan sebaran tanpa infer ulang.

Endpoint:

    GET    /ai/scoring/schema            definisi slider + default + preset
    GET    /ai/scoring/profiles          daftar profil tersimpan
    GET    /ai/scoring/profiles/{name}   satu profil (tuning lengkap)
    POST   /ai/scoring/profiles          simpan/timpa profil
    DELETE /ai/scoring/profiles/{name}   hapus profil
    POST   /ai/scoring/preview           skor dari indeks manual (respons AI lengkap)
    POST   /ai/scoring/evaluate          sebaran skor dataset referensi

Tuning yang sama bisa dipakai di ``/ai/process`` dan ``/ai/advise`` lewat
form field ``profile``/``tuning``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from uvip_ai.scoring import profiles

router = APIRouter(prefix="/ai/scoring", tags=["scoring"])


class PreviewRequest(BaseModel):
    """Hitung skor dari 8 indeks tanpa menyentuh foto/GPU."""

    indices: Dict[str, float]
    tuning: Optional[Dict[str, Any]] = None
    profile: Optional[str] = None


class ProfileSaveRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    tuning: Dict[str, Any]
    description: str = ""


class EvaluateRequest(BaseModel):
    tuning: Optional[Dict[str, Any]] = None
    profile: Optional[str] = None


def _resolve(tuning: Any, profile: Optional[str]) -> dict[str, Any]:
    """Tuning efektif -> spec siap pakai; error dipetakan ke HTTP."""
    try:
        return profiles.resolve(tuning=tuning, profile=profile)
    except profiles.TuningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"profil '{profile}' tidak ditemukan") from None


@router.get("/schema", summary="Definisi slider tuning + default + preset")
def scoring_schema() -> dict[str, Any]:
    return profiles.schema()


@router.get("/profiles", summary="Daftar profil tuning tersimpan")
def scoring_profiles() -> dict[str, Any]:
    return {"profiles": profiles.list_profiles()}


@router.get("/profiles/{name}", summary="Detail satu profil")
def scoring_profile(name: str) -> dict[str, Any]:
    try:
        return profiles.load_profile(name)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"profil '{name}' tidak ditemukan") from None
    except profiles.TuningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/profiles", status_code=201, summary="Simpan profil tuning")
def save_scoring_profile(payload: ProfileSaveRequest) -> dict[str, Any]:
    try:
        return profiles.save_profile(payload.name, payload.tuning, payload.description)
    except profiles.TuningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/profiles/{name}", status_code=204, summary="Hapus profil tuning")
def delete_scoring_profile(name: str) -> Response:
    try:
        removed = profiles.delete_profile(name)
    except profiles.TuningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail=f"profil '{name}' tidak ditemukan")
    return Response(status_code=204)


@router.post("/preview", summary="Skor PVI dari 8 indeks (tanpa GPU)")
def preview_scores(payload: PreviewRequest) -> dict[str, Any]:
    spec = _resolve(payload.tuning, payload.profile)
    try:
        return profiles.score_indices(
            payload.indices,
            spec["tuning"],
            profile=spec["profile"],
            normalized=spec["normalized"],
            warnings=spec["warnings"],
        )
    except profiles.TuningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/evaluate", summary="Sebaran skor seluruh dataset referensi")
def evaluate_scores(payload: EvaluateRequest) -> dict[str, Any]:
    spec = _resolve(payload.tuning, payload.profile)
    try:
        report = profiles.evaluate_reference(spec["tuning"])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except profiles.TuningError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "profile": spec["profile"],
        "profile_hash": spec["profile_hash"],
        **report,
        "warnings": spec["warnings"] + report["warnings"],
    }
