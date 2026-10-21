from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.deps import get_current_student
from app.db.session import get_db
from app.features.students.models import Student
from app.features.subscriptions.models import Subscription
from app.features.subscriptions.schemas import SubscriptionActivateRequest, SubscriptionOut

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])


@router.get("/me", response_model=SubscriptionOut | None)
def my_subscription(
    current_student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """Lets the app show 'subscribe' vs 'active until X' without hitting a gated route."""
    return (
        db.query(Subscription)
        .filter(Subscription.student_id == current_student.id)
        .order_by(Subscription.expires_at.desc())
        .first()
    )


@router.post("/activate", response_model=SubscriptionOut)
def activate_subscription(
    payload: SubscriptionActivateRequest,
    current_student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    PROTOTYPE ONLY: activates a subscription directly, no payment.
    Replace with a real payment-provider webhook (Stripe/Razorpay) before
    launch — that webhook should call this same activation logic after
    verifying the payment, not be triggered directly by the client.
    """
    subscription = Subscription(
        student_id=current_student.id,
        plan_type=payload.plan_type,
        status="active",
        expires_at=datetime.now(timezone.utc) + timedelta(days=payload.duration_days),
    )
    db.add(subscription)
    db.commit()
    db.refresh(subscription)
    return subscription
