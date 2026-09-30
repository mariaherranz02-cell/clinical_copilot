import json

from src.rag import build_rag
from src.retrieval import retrieve


def evaluate():

    with open(
        "data/evaluation_questions.json",
        "r",
        encoding="utf-8"
    ) as f:

        questions = json.load(f)

    documents, model, embeddings = build_rag()

    total = len(questions)

    passed = 0

    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    for i, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        expected_topics = [
            topic.lower()
            for topic in item["expected_topics"]
        ]

        results = retrieve(
            query=question,
            documents=documents,
            embeddings=embeddings,
            model=model,
            top_k=3,
        )

        retrieved_text = " ".join(
            result["document"].lower()
            for result in results
        )

        matches = [
            topic
            for topic in expected_topics
            if topic in retrieved_text
        ]

        score = len(matches) / len(
            expected_topics
        )

        if score >= 0.5:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print()
        print(f"{i}. {question}")
        print(f"Status: {status}")
        print(
            f"Topic coverage: "
            f"{len(matches)}/{len(expected_topics)}"
        )

        print("Retrieved:")

        for result in results:

            title = result["metadata"]["title"]

            similarity = result["score"]

            print(
                f"  - {title} "
                f"({similarity:.3f})"
            )

    print()
    print("=" * 70)

    print(
        f"RESULT: {passed}/{total} "
        f"questions passed"
    )

    print("=" * 70)


if __name__ == "__main__":
    evaluate()