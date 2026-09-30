import json
import ollama

from src.tools import search, metadata


MAX_STEPS = 4


SYSTEM_PROMPT = """
You are Clinical Copilot, a local clinical literature assistant.

You have access to two tools:

1. search:
   Search the clinical knowledge base and scientific literature.

2. metadata:
   Retrieve bibliographic information for relevant scientific papers.

Rules:

- Use the search tool when the question requires information from
  the clinical knowledge base or scientific literature.
- Use the metadata tool when bibliographic information is needed.
- You may use tools multiple times if necessary.
- Do not invent scientific facts.
- Do not invent papers, PMIDs, or DOIs.
- Base literature-related answers on the information returned by the tools.
- If the available evidence is insufficient, explicitly say that there
  is not enough information in the available sources.
- Include relevant paper references in the final answer.
- Keep the final answer clear and concise.
"""


def search_tool(query: str, top_k: int = 3) -> str:
    """Execute the RAG search tool."""

    results = search(query, top_k=top_k)

    return json.dumps(results, ensure_ascii=False, indent=2)


def metadata_tool(query: str, top_k: int = 5) -> str:
    """Execute the paper metadata tool."""

    results = metadata(query, top_k=top_k)

    return json.dumps(results, ensure_ascii=False, indent=2)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the clinical knowledge base and scientific "
                "literature for information relevant to a question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The information to search for.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of documents to retrieve.",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "metadata",
            "description": (
                "Retrieve bibliographic metadata for scientific papers "
                "relevant to a query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Paper topic, title, or subject.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of papers.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


TOOL_FUNCTIONS = {
    "search": search_tool,
    "metadata": metadata_tool,
}


def _extract_sources(tool_result):
    """
    Extract bibliographic metadata from a tool result.

    Sources are collected from both search and metadata results.
    """

    sources = []

    try:
        data = json.loads(tool_result)
    except (json.JSONDecodeError, TypeError):
        return sources

    if not isinstance(data, list):
        return sources

    for item in data:

        if not isinstance(item, dict):
            continue

        # Search results contain metadata inside "metadata".
        if "metadata" in item and isinstance(item["metadata"], dict):
            source = item["metadata"]

        # Metadata results are already metadata dictionaries.
        else:
            source = item

        if not source.get("title"):
            continue

        sources.append(
            {
                "title": source.get("title"),
                "year": source.get("year"),
                "pmid": source.get("pmid"),
                "doi": source.get("doi"),
            }
        )

    return sources


def _deduplicate_sources(sources):
    """Remove duplicate sources while preserving order."""

    unique = []
    seen = set()

    for source in sources:

        key = (
            source.get("doi")
            or source.get("pmid")
            or source.get("title")
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        unique.append(source)

    return unique


def run_agent(
    question: str,
    model: str,
    conversation_history=None,
):
    """
    Run the bounded agent loop.

    conversation_history is optional. If provided, it allows the
    interface to maintain short-term conversational context.

    The agent itself does not persist memory between executions.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # Add previous conversation only when explicitly provided.
    if conversation_history:
        messages.extend(conversation_history)

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    trace = []
    sources = []

    for step in range(MAX_STEPS):

        response = ollama.chat(
            model=model,
            messages=messages,
            tools=TOOLS,
        )

        message = response["message"]

        messages.append(message)

        tool_calls = message.get("tool_calls")

        # The model decided that it has enough information.
        if not tool_calls:

            return {
                "answer": message.get("content", ""),
                "sources": _deduplicate_sources(sources),
                "trace": trace,
                "steps": step + 1,
            }

        for tool_call in tool_calls:

            function = tool_call["function"]

            tool_name = function["name"]
            arguments = function.get("arguments", {})

            trace.append(
                {
                    "step": step + 1,
                    "tool": tool_name,
                    "arguments": arguments,
                }
            )

            if tool_name not in TOOL_FUNCTIONS:

                tool_result = json.dumps(
                    {
                        "error": f"Unknown tool: {tool_name}"
                    }
                )

            else:

                try:
                    tool_result = TOOL_FUNCTIONS[tool_name](**arguments)

                except Exception as exc:

                    tool_result = json.dumps(
                        {
                            "error": str(exc)
                        }
                    )

            # Collect bibliographic information for the UI.
            sources.extend(_extract_sources(tool_result))

            messages.append(
                {
                    "role": "tool",
                    "content": (
                        tool_result
                        if isinstance(tool_result, str)
                        else json.dumps(
                            tool_result,
                            ensure_ascii=False,
                            indent=2,
                        )
                    ),
                }
            )

    return {
        "answer": (
            "The agent reached the maximum number of steps "
            "without producing a final answer."
        ),
        "sources": _deduplicate_sources(sources),
        "trace": trace,
        "steps": MAX_STEPS,
    }