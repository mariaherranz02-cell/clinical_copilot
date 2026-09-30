from src.embeddings import load_documents, create_embeddings
from src.retrieval import retrieve
from src.generation import generate_answer


def build_rag():
    documents = load_documents()

    model, embeddings = create_embeddings(documents)

    return documents, model, embeddings


def ask(
    query: str,
    documents,
    model,
    embeddings,
    top_k: int = 3,
):
    retrieved_documents = retrieve(
        query=query,
        documents=documents,
        embeddings=embeddings,
        model=model,
        top_k=top_k,
    )

    answer = generate_answer(
        query=query,
        retrieved_documents=retrieved_documents,
    )

    return answer, retrieved_documents