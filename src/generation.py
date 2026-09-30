import ollama


MODEL_NAME = "gemma3"


def generate_answer(query: str, retrieved_documents: list[dict]) -> str:
    context = "\n\n".join(
        [
            f"[Document {i + 1}]\n{result['document']}"
            for i, result in enumerate(retrieved_documents)
        ]
    )

    prompt = f"""
You are a clinical information assistant.

Answer the user's question using ONLY the information contained
in the provided documents.

If the documents do not contain enough information to answer the
question, say:

"Not enough information in the provided documents."

Do not invent medical facts.
Do not provide a diagnosis.
Do not recommend treatment beyond what is explicitly stated
in the documents.

Documents:

{context}

User question:

{query}

Answer:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"]