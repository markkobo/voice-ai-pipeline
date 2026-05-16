"""
FastAPI dependency providers.

Routes never instantiate services or repositories directly — they ask for
them via `Depends(get_recordings_service)`. Tests swap implementations via
`app.dependency_overrides[get_recordings_service] = lambda: fake_service`.

Singletons live on `app.state` and are constructed lazily on first request.
This avoids touching disk at import time and keeps test isolation clean.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import Request

from app.services.corpus import (
    CorpusService,
    IngestionService,
    JsonCorpusRepository,
)
from app.services.listeners import JsonListenerRepository, ListenersService
from app.services.personas import JsonPersonaRepository, PersonasService
from app.services.recordings.repository import JsonRecordingsRepository, RecordingsRepository
from app.services.recordings.service import IdValidator, RecordingsService
from app.services.training_service.audio_resolver import RecordingsAudioResolver
from app.services.training_service.repository import JsonTrainingRepository
from app.services.training_service.service import TrainingService


# ---------------------------------------------------------------------------
# Concrete validators — used by RecordingsService and TrainingService to
# check persona_id / listener_id. Each validator wraps a fresh repository
# instance per call, which is cheap (just a JSON load under shared lock).
# ---------------------------------------------------------------------------
class PersonaValidator:
    """Validates persona_ids against the personas repository."""

    def __init__(self, repository: Optional[JsonPersonaRepository] = None) -> None:
        self._repository = repository

    def _repo(self) -> JsonPersonaRepository:
        if self._repository is not None:
            return self._repository
        return JsonPersonaRepository(_resolve_data_root())

    def is_valid(self, id_: str) -> bool:
        return self._repo().exists(id_)

    def list_ids(self) -> set[str]:
        return {p.persona_id for p in self._repo().list()}


class ListenerValidator:
    """Validates listener_ids against the listeners repository."""

    def __init__(self, repository: Optional[JsonListenerRepository] = None) -> None:
        self._repository = repository

    def _repo(self) -> JsonListenerRepository:
        if self._repository is not None:
            return self._repository
        return JsonListenerRepository(_resolve_data_root())

    def is_valid(self, id_: str) -> bool:
        return self._repo().exists(id_)

    def list_ids(self) -> set[str]:
        return {l.listener_id for l in self._repo().list()}


# ---------------------------------------------------------------------------
# Path resolution: services need a data_root. Honor DATA_ROOT env var, fall
# back to the legacy hardcoded path. Phase 1.3 will fold this into a central
# config module.
# ---------------------------------------------------------------------------
def _resolve_data_root() -> Path:
    """Thin wrapper around app.config.data_root() — kept for back-compat with
    callers that imported this name directly."""
    from app import config as _cfg
    return _cfg.data_root()


# ---------------------------------------------------------------------------
# Singletons — lazy, attached to app.state so they survive across requests.
# ---------------------------------------------------------------------------
def _get_or_create_recordings_service(app_state) -> RecordingsService:
    existing: Optional[RecordingsService] = getattr(app_state, "_recordings_service", None)
    if existing is not None:
        return existing

    data_root = _resolve_data_root()
    repo: RecordingsRepository = JsonRecordingsRepository(data_root)
    service = RecordingsService(
        repository=repo,
        persona_validator=PersonaValidator(),
        listener_validator=ListenerValidator(),
        audio_root=data_root / "recordings" / "raw",
    )
    app_state._recordings_service = service
    return service


# ---------------------------------------------------------------------------
# FastAPI dependency callables. Routes write:
#     service: RecordingsService = Depends(get_recordings_service)
# ---------------------------------------------------------------------------
def get_recordings_service(request: Request) -> RecordingsService:
    return _get_or_create_recordings_service(request.app.state)


def get_persona_validator() -> IdValidator:
    return PersonaValidator()


def get_listener_validator() -> IdValidator:
    return ListenerValidator()


# ---------------------------------------------------------------------------
# Training service singleton (Phase 1.2).
# ---------------------------------------------------------------------------
def _resolve_models_dir() -> Path:
    from app import config as _cfg
    return _cfg.models_dir()


def _get_or_create_training_service(app_state) -> TrainingService:
    existing: Optional[TrainingService] = getattr(app_state, "_training_service", None)
    if existing is not None:
        return existing

    data_root = _resolve_data_root()
    models_dir = _resolve_models_dir()
    recordings_svc = _get_or_create_recordings_service(app_state)
    repo = JsonTrainingRepository(models_dir)
    audio_resolver = RecordingsAudioResolver(
        recordings_service=recordings_svc,
        audio_root=data_root / "recordings" / "raw",
    )
    service = TrainingService(
        repository=repo,
        persona_validator=PersonaValidator(),
        audio_resolver=audio_resolver,
        models_dir=models_dir,
    )
    app_state._training_service = service
    return service


def get_training_service(request: Request) -> TrainingService:
    return _get_or_create_training_service(request.app.state)


# ---------------------------------------------------------------------------
# Personas + listeners services (Phase 1.3).
# ---------------------------------------------------------------------------
def _get_or_create_personas_service(app_state) -> PersonasService:
    existing: Optional[PersonasService] = getattr(app_state, "_personas_service", None)
    if existing is not None:
        return existing
    service = PersonasService(JsonPersonaRepository(_resolve_data_root()))
    app_state._personas_service = service
    return service


def _get_or_create_listeners_service(app_state) -> ListenersService:
    existing: Optional[ListenersService] = getattr(app_state, "_listeners_service", None)
    if existing is not None:
        return existing
    service = ListenersService(JsonListenerRepository(_resolve_data_root()))
    app_state._listeners_service = service
    return service


def get_personas_service(request: Request) -> PersonasService:
    return _get_or_create_personas_service(request.app.state)


def get_listeners_service(request: Request) -> ListenersService:
    return _get_or_create_listeners_service(request.app.state)


# ---------------------------------------------------------------------------
# Corpus service singleton (RFC_M6 Phase 0).
# ---------------------------------------------------------------------------
def _get_or_create_corpus_service(app_state) -> CorpusService:
    existing: Optional[CorpusService] = getattr(app_state, "_corpus_service", None)
    if existing is not None:
        return existing
    from app import config as _cfg
    repo = JsonCorpusRepository(_cfg.personas_dir())
    service = CorpusService(repository=repo)
    app_state._corpus_service = service
    return service


def get_corpus_service(request: Request) -> CorpusService:
    return _get_or_create_corpus_service(request.app.state)


def _get_or_create_ingestion_service(app_state) -> IngestionService:
    existing: Optional[IngestionService] = getattr(app_state, "_ingestion_service", None)
    if existing is not None:
        return existing
    # Share the same repository as CorpusService — both write to the
    # same files. Atomic-rename + flock in the repo keeps them safe.
    corpus_svc = _get_or_create_corpus_service(app_state)
    service = IngestionService(repository=corpus_svc.repository)
    app_state._ingestion_service = service
    return service


def get_ingestion_service(request: Request) -> IngestionService:
    return _get_or_create_ingestion_service(request.app.state)


# ---------------------------------------------------------------------------
# Test helpers — let pytest construct an in-test service without going through
# the request lifecycle.
# ---------------------------------------------------------------------------
def make_recordings_service_for_testing(data_root: Path) -> RecordingsService:
    repo = JsonRecordingsRepository(data_root)
    return RecordingsService(
        repository=repo,
        persona_validator=PersonaValidator(),
        listener_validator=ListenerValidator(),
        audio_root=data_root / "recordings" / "raw",
    )


def make_training_service_for_testing(
    data_root: Path,
    recordings_service: Optional[RecordingsService] = None,
    job_factory=None,
) -> TrainingService:
    """
    Build a TrainingService against a per-test data dir.

    By default uses a no-op job_factory so tests don't spawn real subprocess
    training threads. Pass a custom factory to assert on job-start calls.
    """
    if recordings_service is None:
        recordings_service = make_recordings_service_for_testing(data_root)
    models_dir = data_root / "models"
    repo = JsonTrainingRepository(models_dir)
    audio_resolver = RecordingsAudioResolver(
        recordings_service=recordings_service,
        audio_root=data_root / "recordings" / "raw",
    )
    if job_factory is None:

        class _NullJob:
            def start(self): return None
            def cancel(self): return None
            def is_running(self): return False

        def job_factory(**_kw):  # type: ignore[no-redef]
            return _NullJob()

    return TrainingService(
        repository=repo,
        persona_validator=PersonaValidator(),
        audio_resolver=audio_resolver,
        models_dir=models_dir,
        job_factory=job_factory,
    )


def make_personas_service_for_testing(data_root: Path) -> PersonasService:
    return PersonasService(JsonPersonaRepository(data_root))


def make_listeners_service_for_testing(data_root: Path) -> ListenersService:
    return ListenersService(JsonListenerRepository(data_root))


def make_corpus_service_for_testing(data_root: Path) -> CorpusService:
    return CorpusService(JsonCorpusRepository(data_root / "personas"))


def make_ingestion_service_for_testing(
    data_root: Path,
    corpus_service: Optional[CorpusService] = None,
) -> IngestionService:
    if corpus_service is None:
        corpus_service = make_corpus_service_for_testing(data_root)
    return IngestionService(repository=corpus_service.repository)
