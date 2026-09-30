from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents() -> list[str]:

    knowledge_path = Path(
        "data/clinical_knowledge.txt"
    )

    papers_path = Path(
        "data/clinical_papers.txt"
    )

    knowledge_text = knowledge_path.read_text(
        encoding="utf-8"
    )

    papers_text = papers_path.read_text(
        encoding="utf-8"
    )

    documents = []

    documents.extend(
        document.strip()
        for document in knowledge_text.split("\n\n\n")
        if document.strip()
    )

    documents.extend(
        document.strip()
        for document in papers_text.split("\n\n\n")
        if document.strip()
    )

    return documents


def create_embeddings(documents: list[str]):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(documents, normalize_embeddings=True)

    return model, embeddings