import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="AI governance refers to the frameworks, policies, and principles that guide the development and deployment of artificial intelligence systems.",
            metadata={"source": "governance_framework.pdf", "page": 1},
        ),
        Document(
            page_content="Machine learning models require oversight mechanisms to ensure they operate within acceptable ethical and legal boundaries.",
            metadata={"source": "ml_oversight.pdf", "page": 2},
        ),
        Document(
            page_content="Transparency in AI systems is essential for public trust. Stakeholders must be able to understand how decisions are made by automated systems.",
            metadata={"source": "transparency_report.pdf", "page": 3},
        ),
        Document(
            page_content="Algorithmic bias can perpetuate existing social inequalities. Fairness audits are a critical component of responsible AI deployment.",
            metadata={"source": "governance_framework.pdf", "page": 2},
        ),
        Document(
            page_content="Data privacy regulations such as GDPR and CCPA govern how AI systems must handle personal information collected from users.",
            metadata={"source": "ml_oversight.pdf", "page": 5},
        ),
        Document(
            page_content="The OECD AI Principles emphasize that AI should be robust, secure, and safe throughout its entire lifecycle.",
            metadata={"source": "oecd_principles.pdf", "page": 1},
        ),
        Document(
            page_content="Human oversight of AI systems is a key safeguard against unintended consequences. Humans must remain accountable for AI-assisted decisions.",
            metadata={"source": "transparency_report.pdf", "page": 7},
        ),
    ]


@pytest.fixture
def tmp_chroma_dir(tmp_path):
    return str(tmp_path / "chroma_db")
