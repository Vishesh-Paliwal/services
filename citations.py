from page_finder import find_pages_for_chunks


def format_citations(response):
    """Extract citations from grounding_metadata and return formatted text + references."""
    candidate = response.candidates[0] if response.candidates else None
    if not candidate:
        return response.text, []

    metadata = getattr(candidate, "grounding_metadata", None)
    if not metadata:
        return response.text, []

    chunks = getattr(metadata, "grounding_chunks", None) or []
    supports = getattr(metadata, "grounding_supports", None) or []

    if not chunks:
        return response.text, []

    # Find real page numbers via local PDF search
    page_numbers = find_pages_for_chunks(chunks)

    # Build reference list per chunk
    chunk_refs = []
    for i, chunk in enumerate(chunks):
        ctx = getattr(chunk, "retrieved_context", None)
        title = ctx.title if ctx and ctx.title else "Unknown Source"
        uri = ctx.uri if ctx and ctx.uri else ""
        page = page_numbers[i]
        page_info = f"p.{page}" if page else None
        chunk_refs.append({
            "title": title,
            "uri": uri,
            "page_info": page_info,
        })

    # Build deduplicated reference list (title + page_info as key)
    references = []
    seen = {}
    chunk_to_ref = {}
    for i, cref in enumerate(chunk_refs):
        key = (cref["title"], cref["page_info"])
        if key not in seen:
            ref_num = len(references) + 1
            seen[key] = ref_num
            references.append({
                "index": ref_num,
                "title": cref["title"],
                "uri": cref["uri"],
                "page_info": cref["page_info"],
            })
        chunk_to_ref[i] = seen[key]

    # Insert inline citations
    text = response.text or ""
    if supports:
        sorted_supports = sorted(
            supports,
            key=lambda s: s.segment.end_index if s.segment and s.segment.end_index else 0,
            reverse=True,
        )
        for support in sorted_supports:
            if not support.segment or support.segment.end_index is None:
                continue
            indices = support.grounding_chunk_indices or []
            ref_nums = sorted(set(chunk_to_ref.get(idx, idx + 1) for idx in indices))
            if not ref_nums:
                continue
            marker = "[" + ",".join(str(n) for n in ref_nums) + "]"
            end = support.segment.end_index
            text = text[:end] + marker + text[end:]

    return text, references


def format_references_markdown(references):
    """Format references as markdown for display."""
    if not references:
        return ""
    lines = ["**References:**"]
    for ref in references:
        location = f", {ref['page_info']}" if ref.get("page_info") else ""
        lines.append(f"- [{ref['index']}] {ref['title']}{location}")
    return "\n".join(lines)
