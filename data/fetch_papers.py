import json
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path


INPUT_FILE = Path("data/paper_ids.json")
OUTPUT_FILE = Path("data/clinical_papers.txt")

URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def get_text(element):
    if element is None:
        return ""

    return "".join(element.itertext()).strip()


def fetch_paper(pmid):

    response = requests.get(
        URL,
        params={
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        },
        timeout=30
    )

    response.raise_for_status()

    return response.text


def parse_paper(xml_text, topic):

    root = ET.fromstring(xml_text)

    article = root.find(".//PubmedArticle")

    pmid = get_text(
        article.find(".//PMID")
    )

    title = get_text(
        article.find(".//ArticleTitle")
    )

    journal = get_text(
        article.find(".//Journal/Title")
    )

    year = get_text(
        article.find(".//PubDate/Year")
    )

    authors = []

    for author in article.findall(
        ".//AuthorList/Author"
    ):

        lastname = get_text(
            author.find("LastName")
        )

        initials = get_text(
            author.find("Initials")
        )

        if lastname:

            if initials:
                authors.append(
                    f"{lastname} {initials}"
                )
            else:
                authors.append(lastname)

    abstract_parts = []

    for abstract_text in article.findall(
        ".//AbstractText"
    ):

        label = abstract_text.attrib.get(
            "Label"
        )

        text = get_text(abstract_text)

        if label:
            abstract_parts.append(
                f"{label}: {text}"
            )
        else:
            abstract_parts.append(text)

    abstract = "\n".join(
        abstract_parts
    )

    doi = ""

    for article_id in article.findall(
        ".//ArticleId"
    ):

        if article_id.attrib.get(
            "IdType"
        ) == "doi":

            doi = get_text(article_id)

            break

    return {
        "pmid": pmid,
        "doi": doi,
        "title": title,
        "authors": authors,
        "year": year,
        "journal": journal,
        "topic": topic,
        "abstract": abstract
    }


def main():

    papers = json.loads(
        INPUT_FILE.read_text(
            encoding="utf-8"
        )
    )

    documents = []

    for i, paper in enumerate(
        papers,
        start=1
    ):

        pmid = paper["pmid"]

        print(
            f"[{i}/{len(papers)}] "
            f"Fetching PMID {pmid}..."
        )

        xml = fetch_paper(pmid)

        parsed = parse_paper(
            xml,
            paper["topic"]
        )

        document = f"""
TITLE: {parsed["title"]}

AUTHORS: {", ".join(parsed["authors"])}

YEAR: {parsed["year"]}

JOURNAL: {parsed["journal"]}

PMID: {parsed["pmid"]}

DOI: {parsed["doi"]}

TOPICS: {parsed["topic"]}

ABSTRACT:
{parsed["abstract"]}
""".strip()

        documents.append(document)

        time.sleep(0.4)

    OUTPUT_FILE.write_text(
        "\n\n\n".join(documents),
        encoding="utf-8"
    )

    print()
    print(
        f"Saved {len(documents)} papers to "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()