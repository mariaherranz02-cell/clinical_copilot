import numpy as np


def extract_metadata(document: str) -> dict:

    metadata = {
        "title": "",
        "authors": "",
        "year": "",
        "journal": "",
        "pmid": "",
        "doi": "",
        "topics": "",
    }

    for line in document.splitlines():

        if line.startswith("TITLE:"):
            metadata["title"] = line.replace(
                "TITLE:", "", 1
            ).strip()

        elif line.startswith("AUTHORS:"):
            metadata["authors"] = line.replace(
                "AUTHORS:", "", 1
            ).strip()

        elif line.startswith("YEAR:"):
            metadata["year"] = line.replace(
                "YEAR:", "", 1
            ).strip()

        elif line.startswith("JOURNAL:"):
            metadata["journal"] = line.replace(
                "JOURNAL:", "", 1
            ).strip()

        elif line.startswith("PMID:"):
            metadata["pmid"] = line.replace(
                "PMID:", "", 1
            ).strip()

        elif line.startswith("DOI:"):
            metadata["doi"] = line.replace(
                "DOI:", "", 1
            ).strip()

        elif line.startswith("TOPICS:"):
            metadata["topics"] = line.replace(
                "TOPICS:", "", 1
            ).strip()

    return metadata


def retrieve(
    query: str,
    documents: list[str],
    embeddings,
    model,
    top_k: int = 3,
):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        document = documents[index]

        metadata = extract_metadata(
            document
        )

        results.append(
            {
                "document": document,
                "score": float(scores[index]),
                "metadata": metadata,
            }
        )

    return results