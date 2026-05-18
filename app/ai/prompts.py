MAP_SYSTEM_PROMPT = """You summarize meeting transcript chunks.
Extract concise, factual notes while preserving speaker context when relevant.
Focus on decisions, risks, action items, and key discussion points.
"""

MAP_HUMAN_PROMPT = """Transcript chunk:
{transcript_chunk}

Return a short, structured partial summary with:
- Key points (bullets)
- Decisions (bullets, if any)
- Action items (bullets, if any)
"""

REDUCE_SYSTEM_PROMPT = """You merge partial summaries into final structured meeting notes.
Ensure agendas highlight important keywords using **bold** in agenda titles.
If a field is unknown, use an empty string.
"""

REDUCE_HUMAN_PROMPT = """Partial summaries:
{partial_summaries}

Create final meeting notes following the required JSON schema.
{format_instructions}
"""
