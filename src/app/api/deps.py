"""Общие FastAPI-зависимости."""
from __future__ import annotations

from fastapi import Request

from app.services.llm import DraftService
from app.services.loadguard import IncidentDeduplicator, LLMBudget
from app.services.retrieval import KnowledgeBaseIndex


def get_kb_index(request: Request) -> KnowledgeBaseIndex:
    return request.app.state.kb_index


def get_draft_service(request: Request) -> DraftService:
    return request.app.state.draft_service


def get_llm_budget(request: Request) -> LLMBudget:
    return request.app.state.llm_budget


def get_dedup(request: Request) -> IncidentDeduplicator:
    return request.app.state.dedup
