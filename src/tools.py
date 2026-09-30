from src.rag import build_rag
from src.retrieval import retrieve, extract_metadata


# Build the RAG once when the tools module is loaded.
documents, embedding_model, document_embeddings = build_rag()


def search(query: str, top_k: int = 3) -> list[dict]:
    """
    Search the existing RAG for documents relevant to the query.

    This function does not implement a new retrieval system.
    It simply exposes the existing retrieve() function as an agent tool.
    """

    results = retrieve(
        query=query,
        documents=documents,
        embeddings=document_embeddings,
        model=embedding_model,
        top_k=top_k,
    )

    formatted_results = []

    for result in results:
        document = result["document"]
        score = result["score"]

        metadata = extract_metadata(document)

        formatted_results.append(
            {
                "document": document,
                "score": float(score),
                "metadata": metadata,
            }
        )

    return formatted_results


def metadata(query: str, top_k: int = 5) -> list[dict]:
    """
    Find bibliographic metadata for papers relevant to the query.

    This reuses the existing RAG retrieval and extract_metadata()
    functionality.
    """

    results = search(query, top_k=top_k)

    papers = []

    for result in results:
        metadata_info = result["metadata"]

        # Only include documents that contain paper metadata.
        if metadata_info.get("title"):
            papers.append(metadata_info)

    return papers