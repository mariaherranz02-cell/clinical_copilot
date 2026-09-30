import argparse

from src.rag import build_rag, ask


def main():
    parser = argparse.ArgumentParser(
        description="Clinical Copilot - local RAG application"
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question to ask the clinical copilot",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents retrieved",
    )

    args = parser.parse_args()

    print("Loading knowledge base...")
    documents, model, embeddings = build_rag()

    print("Searching relevant documents...\n")

    answer, retrieved_documents = ask(
        query=args.question,
        documents=documents,
        model=model,
        embeddings=embeddings,
        top_k=args.top_k,
    )

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)

    print("\n" + "=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    for i, result in enumerate(
    retrieved_documents,
    start=1
):
        metadata = result["metadata"]

    print(
        f"\nSource {i} "
        f"— similarity: {result['score']:.3f}"
    )

    print(
        f"Title: {metadata['title']}"
    )

    print(
        f"Year: {metadata['year']}"
    )

    print(
        f"Journal: {metadata['journal']}"
    )

    print(
        f"PMID: {metadata['pmid']}"
    )

    print(
        f"DOI: {metadata['doi']}"
    )


if __name__ == "__main__":
    main()