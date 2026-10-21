from app.core.celery_app import celery_app


@celery_app.task(name="sessions.process_training_session")
def process_training_session(session_id: int):
    from app.features.pipeline.service import process_session

    process_session(session_id)
