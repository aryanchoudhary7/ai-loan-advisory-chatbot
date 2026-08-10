import pytest

from src.calculators.emi_calculator import calculate_emi


def test_emi_calculation():
    result = calculate_emi(
        principal=5000000,
        annual_interest_rate=8.5,
        tenure_years=20,
    )

    assert result["emi"] == pytest.approx(43391.16, abs=0.01)
    assert result["total_repayment"] == pytest.approx(
        10413878.80,
        abs=0.01,
    )
    assert result["total_interest"] == pytest.approx(
        5413878.80,
        abs=0.01,
    )

def test_zero_interest_rate():
    result = calculate_emi(
        principal=120000,
        annual_interest_rate=0,
        tenure_years=1,
    )

    assert result["emi"] == 10000
    assert result["total_repayment"] == 120000
    assert result["total_interest"] == 0


def test_invalid_principal():
    with pytest.raises(ValueError):
        calculate_emi(
            principal=0,
            annual_interest_rate=8.5,
            tenure_years=20,
        )


def test_invalid_interest_rate():
    with pytest.raises(ValueError):
        calculate_emi(
            principal=5000000,
            annual_interest_rate=-1,
            tenure_years=20,
        )


def test_invalid_tenure():
    with pytest.raises(ValueError):
        calculate_emi(
            principal=5000000,
            annual_interest_rate=8.5,
            tenure_years=0,
        )