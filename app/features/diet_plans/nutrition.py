"""
Deterministic calorie/macro targets. The LLM writes meals AROUND these numbers; it never chooses them.

Design rules (this is guidance for young athletes, so it stays conservative):
  * targets aim at maintaining/supporting growth and training - never a weight-loss deficit
  * youth (9-18) use the US Institute of Medicine EER equations, adults use Mifflin-St Jeor x activity
  * needs age + gender; without them we refuse rather than guess
"""
from dataclasses import dataclass
from datetime import date

ADULT_ACTIVITY = 1.6  # training most days
YOUTH_PA = {"male": 1.26, "female": 1.31}  # IOM "active" physical-activity coefficients


class ProfileIncomplete(Exception):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__("Profile incomplete: " + ", ".join(missing))


class AgeOutOfRange(Exception):
    pass


@dataclass
class Targets:
    age: int
    is_minor: bool
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int
    hydration_l: float
    method: str


def age_years(dob: date, today: date) -> int:
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _round50(x: float) -> int:
    return int(round(x / 50.0) * 50)


def compute_targets(dob: date | None, gender: str | None, height_cm: float, weight_kg: float, today: date) -> Targets:
    missing = [n for n, v in (("date_of_birth", dob), ("gender", gender)) if not v]
    if missing:
        raise ProfileIncomplete(missing)
    age = age_years(dob, today)
    if age < 9:
        raise AgeOutOfRange("Diet plans are available from age 9.")
    h_m = height_cm / 100.0
    if age < 19:
        if gender == "male":
            kcal = 88.5 - 61.9 * age + YOUTH_PA["male"] * (26.7 * weight_kg + 903 * h_m) + 25
        else:
            kcal = 135.3 - 30.8 * age + YOUTH_PA["female"] * (10 * weight_kg + 934 * h_m) + 25
        method, protein_per_kg = "IOM EER (youth, active) incl. growth allowance", 1.4
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if gender == "male" else -161)
        kcal, method, protein_per_kg = bmr * ADULT_ACTIVITY, "Mifflin-St Jeor x 1.6 (training most days)", 1.6
    calories = _round50(kcal)
    protein = int(round(weight_kg * protein_per_kg))
    fat = int(round(calories * 0.28 / 9))
    carbs = int(round((calories - protein * 4 - fat * 9) / 4))
    return Targets(age, age < 18, calories, protein, carbs, fat, round(weight_kg * 0.035 + 0.5, 1), method)
