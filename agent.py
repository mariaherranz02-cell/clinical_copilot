import argparse

from src.agent import run_agent


DEFAULT_MODEL = "PetrosStav/gemma3-tools:4b"


def main():

    parser = argparse.ArgumentParser(
        description="Clinical Copilot - Agentic RAG"
    )

    parser.add_argument(
        "question",
        nargs="+",
        help="Question for Clinical Copilot.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model to use.",
    )

    args = parser.parse_args()

    question = " ".join(args.question)

    print("=" * 60)
    print("CLINICAL COPILOT")
    print("=" * 60)

    print(f"\nQuestion:\n{question}")

    result = run_agent(
        question=question,
        model=args.model,
    )

    print("\n" + "-" * 60)
    print("ANSWER")
    print("-" * 60)

    print(result["answer"])

    print("\n" + "-" * 60)
    print("AGENT TRACE")
    print("-" * 60)

    for item in result["trace"]:
        print(
            f"Step {item['step']}: "
            f"{item['tool']}({item['arguments']})"
        )

    if result["sources"]:

        print("\n" + "-" * 60)
        print("SOURCES")
        print("-" * 60)

        for i, source in enumerate(result["sources"], 1):

            print(f"\n{i}. {source['title']}")

            if source.get("year"):
                print(f"   Year: {source['year']}")

            if source.get("pmid"):
                print(f"   PMID: {source['pmid']}")

            if source.get("doi"):
                print(f"   DOI: {source['doi']}")

    print(
        f"\nAgent steps used: {result['steps']}/{4}"
    )


if __name__ == "__main__":
    main()