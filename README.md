# AI Loan Advisory Chatbot

An AI-powered Loan Advisory Agent that allows users to ask loan-related questions in natural language and receive accurate, source-backed answers. The system processes financial documents such as bank loan terms and RBI guidelines, retrieves the most relevant information using semantic search, and generates clear responses using Retrieval-Augmented Generation (RAG).

## Problem Statement

The AI Loan Advisory Agent is designed to help users understand loan-related information without manually reading lengthy financial documents.

The system processes documents such as loan agreements, bank terms and conditions, and RBI guidelines. It retrieves the most relevant information for a user's question and generates a concise response grounded in the retrieved documents.

The application supports queries related to:

- Loan eligibility
- Loan terms and conditions
- RBI loan guidelines
- Interest and repayment-related information
- EMI estimation

The application also includes a deterministic EMI calculator that calculates monthly EMI, total interest, and total repayment based on the loan amount, annual interest rate, and tenure.

## Key Features

- Natural-language loan question answering
- Retrieval-Augmented Generation (RAG)
- Multi-document financial document corpus
- Semantic retrieval using embeddings
- Local FAISS vector search
- Gemini-powered response generation
- Context-grounded responses
- Response validation and safe fallbacks
- Source and page-level information
- Loan eligibility and RBI guideline queries
- Deterministic EMI calculator
- Streamlit interface
- Automated testing

## Document Corpus

The application uses 14 publicly available financial documents from:

- SBI
- Axis Bank
- ICICI Bank
- Reserve Bank of India (RBI)

The corpus covers:

- Home loans
- Personal loans
- Education loans
- RBI loan-related guidelines

Each document is associated with metadata such as institution, loan type, document type, source, and page number.

## System Architecture

```text
                         User
                          │
             ┌────────────┴────────────┐
             │                         │
       Loan Question               EMI Inputs
             │                         │
             ▼                         ▼
       RAG Pipeline              EMI Calculator
             │                         │
             ▼                         ▼
      Query Embedding         EMI / Interest /
             │                  Total Repayment
             ▼
       FAISS Retrieval
             │
             ▼
    Relevant Document Chunks
             │
             ▼
       Context Builder
             │
             ▼
        Gemini LLM
             │
             ▼
    Response Validation
             │
             ▼
      Answer + Sources
```

## RAG Flow

```text
PDF Documents
      │
      ▼
Document Ingestion
      │
      ▼
Text Cleaning & Chunking
      │
      ▼
Sentence Embeddings
      │
      ▼
FAISS Vector Store
      │
      │
User Question
      │
      ▼
Query Embedding
      │
      ▼
Semantic Retrieval
      │
      ▼
Relevant Context
      │
      ▼
Gemini
      │
      ▼
Response Validation
      │
      ▼
Answer + Sources
```

The system retrieves relevant information from the financial document corpus and provides the retrieved context to Gemini for grounded response generation.

## EMI Calculator

The application includes a deterministic EMI calculator.

**Inputs:**

- Loan amount
- Annual interest rate
- Loan tenure

**Outputs:**

- Monthly EMI
- Total interest
- Total repayment

The calculation is performed programmatically rather than by the LLM. The calculator also handles zero-interest loans and validates invalid inputs.

## Response Validation

Generated responses are validated before being returned to the user.

The validation layer handles:

- Empty responses
- Empty retrieved context
- Whitespace
- Insufficient information

If the available documents do not contain enough information, the system returns a safe fallback instead of inventing information.

## Technologies Used

- **Python**
- **Streamlit**
- **LangChain**
- **FAISS**
- **Sentence Transformers**
- **Gemini API**
- **PyMuPDF**
- **Pytest**
- **NumPy**

## Installation

Clone the repository:

```bash
git clone https://github.com/aryanchoudhary7/ai-loan-advisory-chatbot.git
cd ai-loan-advisory-chatbot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file based on `.env.example` and add:

```text
GEMINI_API_KEY=your_api_key_here
```

The `.env` file is excluded from Git and is not included in the repository.

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application provides:

- Loan question answering
- Document-grounded responses
- Source and page information
- Loan eligibility information
- RBI guideline information
- EMI calculation

## Testing

Run the complete test suite:

```bash
python -m pytest -v
```

Current test result:

```text
38 passed, 1 warning
```

The warning is a dependency deprecation warning and does not cause test failures.

## Limitations

- Responses are limited to the indexed document corpus.
- If the required information is not available in the indexed documents, the system does not assume or invent missing information.
- Loan policies and regulatory guidelines may change over time.
- EMI calculations are estimates and may not include bank-specific charges.
- The system is intended for informational purposes and not as a substitute for professional financial advice.
