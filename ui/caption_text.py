import re
from textwrap import wrap


# Break Korean and other sentence endings, including punctuation inside quotes.
_SENTENCE_BREAK = re.compile(r'(?<=[.!?…])(?=\s|$)|(?<=[.!?…。！？][”’"\'])(?=\s|$)|(?<=[。！？])(?![”’"\'])')
MAX_CAPTION_LENGTH = 48


def caption_parts(text: str) -> list[str]:
    """Split a growing caption into short display items, keeping its words."""
    parts = []
    for sentence in _SENTENCE_BREAK.split(' '.join(text.split())):
        parts.extend(wrap(sentence, width=MAX_CAPTION_LENGTH, break_on_hyphens=False))
    return parts
