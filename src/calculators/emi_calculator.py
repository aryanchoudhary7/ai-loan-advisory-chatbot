def calculate_emi(
    principal: float,
    annual_interest_rate: float,
    tenure_years: int,
) -> dict[str, float]:
    """
    Calculate monthly EMI, total interest, and total repayment.

    Args:
        principal: Loan amount.
        annual_interest_rate: Annual interest rate in percentage.
        tenure_years: Loan tenure in years.

    Returns:
        Dictionary containing EMI, total interest, and total repayment.
    """

    if principal <= 0:
        raise ValueError("Principal must be greater than zero.")

    if annual_interest_rate < 0:
        raise ValueError("Interest rate cannot be negative.")

    if tenure_years <= 0:
        raise ValueError("Tenure must be greater than zero.")

    months = tenure_years * 12
    monthly_rate = annual_interest_rate / (12 * 100)

    if monthly_rate == 0:
        emi = principal / months
    else:
        factor = (1 + monthly_rate) ** months
        emi = principal * monthly_rate * factor / (factor - 1)

    total_repayment = emi * months
    total_interest = total_repayment - principal

    return {
        "emi": round(emi, 2),
        "total_interest": round(total_interest, 2),
        "total_repayment": round(total_repayment, 2),
    }