import re

FILLER_WORDS = [
    "uh",
    "um",
    "erm",
    "like",
    "you know",
    "i mean",
    "sort of",
    "kind of",
]

FILLER_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in FILLER_WORDS) + r")\b",
    flags=re.IGNORECASE,
)

NOISE_PATTERN = re.compile(
    r"\[(?:inaudible|laughter|laughs|crosstalk|noise|applause|music)[^\]]*\]"
    r"|\((?:inaudible|laughter|laughs|crosstalk|noise|applause|music)[^)]*\)",
    flags=re.IGNORECASE,
)

SPEAKER_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _\-.]{0,40}):\s*(.*)$")


def _clean_text(text: str) -> str:
    cleaned = NOISE_PATTERN.sub(" ", text)
    cleaned = FILLER_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def clean_transcript(transcript: str) -> str:
    if not transcript:
        return ""

    lines: list[str] = []
    last_blank = False

    for raw_line in transcript.splitlines():
        if not raw_line.strip():
            if not last_blank:
                lines.append("")
                last_blank = True
            continue

        last_blank = False
        speaker_match = SPEAKER_PATTERN.match(raw_line)

        if speaker_match:
            speaker = speaker_match.group(1).strip()
            content = _clean_text(speaker_match.group(2))
            if content:
                lines.append(f"{speaker}: {content}")
            else:
                lines.append(f"{speaker}:")
            continue

        cleaned_line = _clean_text(raw_line)
        if cleaned_line:
            lines.append(cleaned_line)

    return "\n".join(lines).strip()
