from app.core.celery_app import celery_app


@celery_app.task(name="uploads.validate_self_upload")
def validate_self_upload(upload_id: int):
    from app.features.uploads.validation import run_validation

    run_validation(upload_id)


@celery_app.task(name="uploads.process_self_upload")
def process_self_upload(upload_id: int):
    from app.features.pipeline.service import process_self_upload as run

    run(upload_id)
