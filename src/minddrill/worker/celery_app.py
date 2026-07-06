"""Celery app for the ingestion path (broker + result backend = Redis).

Ingestion only — the query path is synchronous and never queued.
"""

from celery import Celery

from minddrill.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "minddrill",
    broker=_settings.redis_url,
    backend=_settings.redis_url,
    include=["minddrill.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Redeliver a job if a worker dies mid-task; combined with idempotency this
    # makes a redeploy resume rather than drop work.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)
