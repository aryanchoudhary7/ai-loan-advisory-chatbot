from src.llm.gemini_client import GeminiClient
from src.llm.prompts import LOAN_ADVISORY_SYSTEM_PROMPT


client = GeminiClient()

prompt = """
Retrieved Context:

[Source 1]
Source: sbi_home_loan_mitc.pdf
Page: 1

Loan to Value Ratio (LTV):
For loan amount upto Rs.20 Lacs, maximum permissible LTV ratio is 90%
of the assessed value of the property. For loan amount greater than
Rs.20 Lacs and upto Rs.75 Lacs, maximum permissible LTV ratio is 80%.
A maximum permissible LTV ratio of 75% is applicable on a loan amount
above Rs.75 Lacs.

User Question:

What is the current SBI home loan interest rate?

Answer the question using only the retrieved context.
If the context does not contain the answer, clearly state that
the available documents do not contain enough information.
"""

response = client.generate(
    prompt=prompt,
    system_instruction=LOAN_ADVISORY_SYSTEM_PROMPT,
)

print("\nGemini Response:")
print(response)