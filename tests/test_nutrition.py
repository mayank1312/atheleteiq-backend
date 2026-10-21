from datetime import date

import pytest

from app.features.diet_plans.nutrition import AgeOutOfRange, ProfileIncomplete, compute_targets

TODAY = date(2026, 9, 28)


def test_youth_boy_matches_iom_equation():
    t = compute_targets(date(2010, 3, 1), "male", 175, 65, TODAY)  # age 16
    assert t.is_minor and t.age == 16 and t.calories == 3300 and t.protein_g == 91
    assert abs(t.calories - (t.protein_g * 4 + t.carbs_g * 4 + t.fat_g * 9)) < 60  # macros add up to the calories


def test_youth_girl_and_adult():
    assert compute_targets(date(2011, 1, 1), "female", 160, 52, TODAY).calories == 2350
    a = compute_targets(date(2001, 5, 5), "male", 175, 70, TODAY)
    assert not a.is_minor and a.calories == 2700 and a.protein_g == 112


def test_refuses_to_guess_missing_profile_or_tiny_children():
    with pytest.raises(ProfileIncomplete) as e:
        compute_targets(None, None, 170, 60, TODAY)
    assert set(e.value.missing) == {"date_of_birth", "gender"}
    with pytest.raises(AgeOutOfRange):
        compute_targets(date(2020, 1, 1), "male", 110, 20, TODAY)
