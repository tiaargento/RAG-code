"""Generation layer: synthesize answers from retrieved chunks via Ollama."""

DEFAULT_GENERATION_MODEL = "llama3.2:3b"


def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """Build a grounded prompt from a user query and retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown source")
        title = meta.get("title", "untitled")
        text = chunk.get("text", "")
        context_parts.append(
            f"[Source {i}] title={title} source={source}\n{text}"
        )

    context = "\n\n".join(context_parts)
    # The prompt explicitly constrains the model to retrieved context only.
    return f"""You are answering a question using retrieved document chunks.

Use only the context below. If the context is not enough to answer, say so.
Answer clearly and briefly. Include the most relevant source titles at the end.

Question:
{query}

Context:
{context}

Answer:
"""


def generate_answer(
    query: str,
    chunks: list[dict],
    model: str = DEFAULT_GENERATION_MODEL,
) -> str:
    """Call a local Ollama model to answer using the retrieved chunks."""
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "The optional 'ollama' package is not installed. Run: pip install ollama"
        ) from exc

    prompt = build_rag_prompt(query, chunks)
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ollama generation failed. Make sure Ollama is running and run: ollama pull {model}"
        ) from exc

    return response["message"]["content"].strip()
