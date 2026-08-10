from pathlib import Path
import csv


RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("metadata/document_inventory.csv")


def get_document_type(institution: str, loan_type: str) -> str:
    if institution == "RBI":
        return "regulatory_guideline"

    if institution == "SBI":
        return "MITC"

    return "terms_and_conditions"


def build_inventory():
    documents = []

    for pdf_path in RAW_DIR.rglob("*.pdf"):
        relative_path = pdf_path.relative_to(RAW_DIR)
        parts = relative_path.parts

        institution = parts[0].upper()

        if institution == "RBI":
            loan_type = "general"
        else:
            loan_type = parts[1]

        documents.append(
            {
                "institution": institution,
                "institution_type": (
                    "regulator" if institution == "RBI" else "bank"
                ),
                "loan_type": loan_type,
                "document_type": get_document_type(
                    institution, loan_type
                ),
                "source": pdf_path.name,
                "relative_path": str(relative_path),
            }
        )

    documents.sort(
        key=lambda x: (
            x["institution"],
            x["loan_type"],
            x["source"],
        )
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "institution",
                "institution_type",
                "loan_type",
                "document_type",
                "source",
                "relative_path",
            ],
        )

        writer.writeheader()
        writer.writerows(documents)

    print(f"Documents found: {len(documents)}")
    print(f"Inventory created: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_inventory()