"""
Query enhancer: classifies questions and decomposes complex multi-topic
queries into sub-queries for better cross-chapter retrieval.

Only triggers decomposition for genuinely complex questions (comparisons,
multi-chapter, multi-concept). Simple factual questions pass through unchanged.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

from config import MODEL

logger = logging.getLogger(__name__)

CLASSIFIER_PROMPT = """You are a query complexity classifier for a bioprocess engineering textbook RAG system.

Given a user question, decide if it needs to be decomposed into multiple search queries for better retrieval.

DECOMPOSE ONLY when the question:
- Explicitly compares 2+ distinct topics/technologies (e.g., "Compare BRoC vs stirred tank")
- Requires information from multiple chapters/sections (e.g., "How do upstream and downstream processes interact?")
- Asks about relationships between distinct concepts from different areas (e.g., "How does microbial kinetics affect bioreactor scale-up?")
- Lists multiple items needing separate lookups (e.g., "Explain batch, fed-batch, and continuous fermentation")

DO NOT DECOMPOSE when the question:
- Asks about a single concept, even if complex (e.g., "Explain the Del factor in sterilization")
- Asks for a definition, mechanism, or explanation of one topic
- Is about a single chapter's content, even if detailed
- Is a calculation question about one process
- Asks about a single table, figure, or diagram

Respond with ONLY valid JSON, no markdown:
- If simple: {"decompose": false}
- If complex: {"decompose": true, "sub_queries": ["query1", "query2", ...]}

Sub-query rules:
- Maximum 4 sub-queries
- Each sub-query should target a DIFFERENT chapter/topic area
- Make sub-queries specific and searchable (not vague)
- Include the original question's key terms in relevant sub-queries

Examples:

Question: "What is the Del factor and how is it used in sterilization?"
{"decompose": false}

Question: "Compare the advantages of stirred-tank bioreactors versus airlift bioreactors for mammalian cell culture"
{"decompose": true, "sub_queries": ["stirred-tank bioreactor design advantages mammalian cell culture", "airlift bioreactor design advantages mammalian cell culture"]}

Question: "How do microbial growth kinetics influence bioreactor design and scale-up strategies?"
{"decompose": true, "sub_queries": ["microbial growth kinetics Monod equation specific growth rate", "bioreactor design parameters for microbial cultivation", "scale-up strategies criteria dimensionless numbers"]}

Question: "What are the six phases of microbial batch growth?"
{"decompose": false}

Question: "Explain the role of mass transfer, heat transfer, and mixing in bioreactor performance"
{"decompose": true, "sub_queries": ["mass transfer oxygen transfer rate kLa bioreactor", "heat transfer cooling bioreactor temperature control", "mixing impeller power input bioreactor homogeneity"]}
"""

REWRITE_PROMPT = """
You are a query rewrite assistant for a bioprocess engineering textbook RAG system.

   it currently performs poorly on multi chapter references , cause similliar keywords sometimes get more content from one chapter rather than other .
   based on query , think of an ideal answer . and based on that answer write the best query you can that retrieves the most relevant chunks to answer .
   i will be passing your generated query directly to the file search tool , so make sure it is a valid query.
"""

RETRIEVAL_ONLY_PROMPT = """Retrieve and summarize the most relevant information from the uploaded books to answer this question. Focus on extracting specific facts, data, equations, and details. Be thorough but concise."""

SYNTHESIS_PROMPT_TEMPLATE = """You have retrieved the following content from textbook chapters to answer the user's question. Use ALL of this retrieved content as your primary source.

═══ RETRIEVED CONTENT ═══
{chunks_text}
═══ END RETRIEVED CONTENT ═══

{system_prompt}"""


def rewrite_query(client: genai.Client, question: str) -> str:
    """Rewrite a user question into an optimized search query for better retrieval."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=REWRITE_PROMPT,
                temperature=0,
                max_output_tokens=2048,
            ),
        )
        rewritten = response.text.strip()
        logger.info(f"Query rewrite: '{question[:60]}...' → '{rewritten[:80]}...'")
        return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed ({e}), using original")
        return question


def query_rewritten(client: genai.Client, store_name: str, question: str,
                    mode: str = "strict", top_k: int = 10) -> tuple[object, dict]:
    """
    Rewrite the query for better retrieval, then run normal single-query search.
    The rewritten query is used for retrieval, but the original question is what
    the model answers.

    Returns (response, metadata) where metadata has rewrite info.
    """
    from query_engine import STRICT_PROMPT, AUGMENTED_PROMPT

    rewritten = rewrite_query(client, question)
    system_prompt = STRICT_PROMPT if mode == "strict" else AUGMENTED_PROMPT

    # Use rewritten query as the content (for better retrieval) but prepend
    # the original question so the model knows what to actually answer
    combined_prompt = f"{rewritten}"

    response = client.models.generate_content(
        model=MODEL,
        contents=combined_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_name],
                    )
                )
            ],
        ),
    )

    metadata = {
        "rewritten": True,
        "original_query": question,
        "rewritten_query": rewritten,
    }
    return response, metadata


def classify_query(client: genai.Client, question: str) -> dict:
    """Classify whether a question needs decomposition. Returns dict with 'decompose' and optionally 'sub_queries'."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=CLASSIFIER_PROMPT,
                temperature=0,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )
        text = response.text.strip()
        result = json.loads(text)
        logger.info(f"Query classification: {result}")
        return result
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Classification failed ({e}), falling back to simple query")
        return {"decompose": False}


def _retrieve_chunks_for_subquery(client: genai.Client, store_name: str, sub_query: str, top_k: int = 10):
    """Run a single file search retrieval call and extract chunks."""
    response = client.models.generate_content(
        model=MODEL,
        contents=sub_query,
        config=types.GenerateContentConfig(
            system_instruction=RETRIEVAL_ONLY_PROMPT,
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_name],
                    )
                )
            ],
        ),
    )

    # Extract grounding chunks
    candidate = response.candidates[0] if response.candidates else None
    if not candidate:
        return []
    metadata = getattr(candidate, "grounding_metadata", None)
    if not metadata:
        return []
    raw_chunks = getattr(metadata, "grounding_chunks", None) or []

    chunks = []
    for chunk in raw_chunks:
        ctx = getattr(chunk, "retrieved_context", None)
        chunks.append({
            "text": ctx.text if ctx and ctx.text else "",
            "title": ctx.title if ctx and ctx.title else "",
            "uri": ctx.uri if ctx and ctx.uri else "",
        })
    return chunks


def _deduplicate_chunks(all_chunks: list[dict]) -> list[dict]:
    """Deduplicate chunks by text content (exact match on first 200 chars)."""
    seen = set()
    unique = []
    for chunk in all_chunks:
        key = chunk["text"][:200]
        if key and key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


def retrieve_multi_query(client: genai.Client, store_name: str, sub_queries: list[str], top_k: int = 10) -> list[dict]:
    """Run multiple retrieval queries in parallel and merge/deduplicate chunks."""
    all_chunks = []

    with ThreadPoolExecutor(max_workers=len(sub_queries)) as executor:
        futures = {
            executor.submit(_retrieve_chunks_for_subquery, client, store_name, sq, top_k): sq
            for sq in sub_queries
        }
        for future in as_completed(futures):
            sq = futures[future]
            try:
                chunks = future.result()
                logger.info(f"Sub-query '{sq[:50]}...' returned {len(chunks)} chunks")
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Sub-query '{sq[:50]}...' failed: {e}")

    deduped = _deduplicate_chunks(all_chunks)
    logger.info(f"Multi-query: {len(all_chunks)} total chunks → {len(deduped)} after dedup")
    return deduped


def synthesize_with_chunks(client: genai.Client, question: str, chunks: list[dict], system_prompt: str) -> object:
    """Generate final answer using pre-retrieved chunks as context (no file_search tool)."""
    # Format chunks into context text
    chunks_text = ""
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "")
        text = chunk.get("text", "")
        chunks_text += f"[Chunk {i}] {title}\n{text}\n\n"

    full_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        chunks_text=chunks_text,
        system_prompt=system_prompt,
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=full_prompt,
        ),
    )
    return response


def query_enhanced(client: genai.Client, store_name: str, question: str,
                   mode: str = "strict", top_k: int = 10,
                   force_decompose: bool = False) -> tuple[object, dict]:
    """
    Enhanced query with optional multi-query decomposition.

    Returns (response, enhancer_metadata) where metadata contains:
    - decomposed: bool
    - sub_queries: list[str] or None
    - total_chunks: int
    - deduped_chunks: int
    """
    from query_engine import query, STRICT_PROMPT, AUGMENTED_PROMPT

    # Step 1: Classify
    if force_decompose:
        classification = {"decompose": True, "sub_queries": [question]}
    else:
        classification = classify_query(client, question)

    if not classification.get("decompose"):
        # Simple query — pass through to existing query engine
        response = query(client, store_name, question, mode=mode, top_k=top_k)
        return response, {"decomposed": False, "sub_queries": None, "total_chunks": 0, "deduped_chunks": 0}

    # Step 2: Multi-query retrieval
    sub_queries = classification.get("sub_queries", [question])
    # Also include the original question as a sub-query to not miss direct matches
    if question not in sub_queries:
        sub_queries.append(question)

    chunks = retrieve_multi_query(client, store_name, sub_queries, top_k=top_k)

    if not chunks:
        # Fallback: if multi-query returned nothing, try single query
        logger.warning("Multi-query returned no chunks, falling back to single query")
        response = query(client, store_name, question, mode=mode, top_k=top_k)
        return response, {"decomposed": True, "sub_queries": sub_queries, "total_chunks": 0, "deduped_chunks": 0, "fallback": True}

    # Step 3: Synthesize with all chunks
    system_prompt = STRICT_PROMPT if mode == "strict" else AUGMENTED_PROMPT
    response = synthesize_with_chunks(client, question, chunks, system_prompt)

    metadata = {
        "decomposed": True,
        "sub_queries": sub_queries,
        "deduped_chunks": len(chunks),
        "retrieved_chunks": chunks,  # pass chunks through for eval
    }

    return response, metadata
