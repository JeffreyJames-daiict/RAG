"""Technical and domain-aware tokenizer for sparse BM25 retrieval.

Preserves:
- Patent numbers (e.g., 'US 11,234,567 B2', 'US20210123456A1', 'WO/2020/123456', 'EP3456789')
- Product codes & models (e.g., 'GTX-4090', 'SKU-1234-X', 'RES-470k-1%', 'i7-13700K')
- Company names & acronyms (e.g., 'AT&T', '3M', 'Johnson & Johnson', 'OpenAI')
- Claim numbers (e.g., 'Claim 1', 'Claim 14(a)', '[Claim 12]')
- Exact quoted phrases (e.g., '"quantum error correction"')
- Unicode normalization (NFC) and safe stopword handling
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Set, Tuple

# Safe stopwords: only functional words.
# Excludes technical/legal words like: claim, patent, table, fig, section, model, system, code, etc.
SAFE_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
    "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their",
    "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves",
}

# Regex to detect patent identifiers:
# e.g., 'US 11,234,567 B2', 'US20210123456A1', 'WO/2020/123456', 'EP3456789', 'RE45,123'
PATENT_PATTERN = re.compile(
    r"\b(?:US|EP|WO|JP|CN|KR|CA|GB)\s*[-/]?\s*\d{1,3}(?:,\d{3})+(?:\s*[A-Z]\d?)?\b"
    r"|\b(?:US|EP|WO|JP|CN|KR|CA|GB)\s*[-/]?\s*\d{6,14}(?:[A-Z]\d?)?\b"
    r"|\b(?:US|EP|WO|JP|CN|KR|CA|GB)\s*[-/]?\s*\d{4}[-/]?\d{4,8}(?:[A-Z]\d?)?\b"
    r"|\bRE\s*[-/]?\s*\d{2,6}\b",
    re.IGNORECASE,
)

# Regex to detect claim references:
# e.g., 'Claim 1', 'claim 14(a)', '[Claim 12]'
CLAIM_PATTERN = re.compile(
    r"\bclaims?\s+(\d+(?:\([a-z0-9]+\))?)\b",
    re.IGNORECASE,
)

# Regex for company names & acronyms with symbols:
# e.g., 'AT&T', '3M', 'Johnson & Johnson'
COMPANY_PATTERN = re.compile(
    r"\b(?:AT&T|3M|Johnson\s*&\s*Johnson)\b",
    re.IGNORECASE,
)

# Regex for product codes / technical compound tokens:
# e.g., 'GTX-4090', 'SKU-1234-X', 'RES-470k-1%', 'i7-13700K', 'v1.0'
PRODUCT_CODE_PATTERN = re.compile(
    r"\b[A-Za-z0-9]+(?:[-_/.][A-Za-z0-9]+)+\b"
)

# General word token pattern
WORD_PATTERN = re.compile(r"\b[A-Za-z0-9]+(?:'[a-z]+)?\b")

# Quoted phrase pattern
QUOTED_PHRASE_PATTERN = re.compile(r'["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]')


def normalize_text(text: str) -> str:
    """Normalize unicode, convert smart quotes to standard ASCII, strip zero-width chars."""
    if not text:
        return ""
    # NFC Unicode normalization
    text = unicodedata.normalize("NFC", text)
    # Convert curly quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # Strip invisible control chars
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u2060", "\u00ad"):
        text = text.replace(ch, "")
    return text


def extract_phrases(text: str) -> Tuple[str, List[str]]:
    """Extract quoted phrases from query or text.
    
    Returns:
        Tuple of (text_with_quotes_removed, list_of_phrases_lowercased)
    """
    phrases: List[str] = []
    normalized = normalize_text(text)

    def _replace_phrase(match: re.Match) -> str:
        phrase_content = match.group(1).strip().lower()
        if phrase_content:
            phrases.append(phrase_content)
        return " " + phrase_content + " "

    remaining_text = QUOTED_PHRASE_PATTERN.sub(_replace_phrase, normalized)
    return remaining_text, phrases


def normalize_patent_id(patent_str: str) -> Tuple[str, str]:
    """Produce canonical and hyphenated forms for a patent number string.
    
    e.g., 'US 11,234,567 B2' -> ('us11234567b2', 'us-11234567-b2')
    """
    cleaned = re.sub(r"[\s,/-]+", "", patent_str).lower()
    # Also create a hyphenated version: e.g. us-11234567-b2
    m = re.match(r"^([a-z]{2,3})(\d+)([a-z0-9]*)$", cleaned)
    if m:
        hyphenated = f"{m.group(1)}-{m.group(2)}"
        if m.group(3):
            hyphenated += f"-{m.group(3)}"
    else:
        hyphenated = cleaned
    return cleaned, hyphenated


def tokenize(
    text: str,
    remove_stopwords: bool = True,
    preserve_identifiers: bool = True,
) -> List[str]:
    """Tokenize text into search tokens, preserving technical identifiers and patent numbers.

    Emits:
      - Canonical and hyphenated patent tokens
      - Normalized claim tokens (e.g., 'claim-1', 'claim-14a')
      - Preserved company abbreviations ('at&t', 'att', '3m')
      - Preserved product codes ('gtx-4090' + subwords 'gtx', '4090')
      - General words (filtered for stopwords when remove_stopwords=True)

    Args:
        text: Input string.
        remove_stopwords: Whether to remove common functional English stopwords.
        preserve_identifiers: Whether to extract specialized patent and product identifiers.

    Returns:
        List of lowercase string tokens.
    """
    if not text:
        return []

    normalized = normalize_text(text)
    tokens: List[str] = []
    seen: Set[str] = set()

    def _add_token(t: str) -> None:
        lowered = t.lower().strip()
        if lowered and lowered not in seen:
            tokens.append(lowered)
            seen.add(lowered)

    if preserve_identifiers:
        # 1. Patent numbers
        for p_match in PATENT_PATTERN.finditer(normalized):
            p_str = p_match.group(0)
            canonical, hyphenated = normalize_patent_id(p_str)
            _add_token(canonical)
            _add_token(hyphenated)
            # Also extract standalone numeric segment (e.g. '11234567')
            num_match = re.search(r"\d{1,3}(?:,\d{3})+|\d{6,14}", p_str)
            if num_match:
                num_only = re.sub(r"[^\d]", "", num_match.group(0))
                if len(num_only) >= 5:
                    _add_token(num_only)

        # 2. Company names & symbols
        for c_match in COMPANY_PATTERN.finditer(normalized):
            c_str = c_match.group(0).lower()
            if "at&t" in c_str:
                _add_token("at&t")
                _add_token("att")
            elif "3m" in c_str:
                _add_token("3m")
            elif "johnson" in c_str:
                _add_token("johnson&johnson")
                _add_token("johnson")

        # 3. Claims
        for cl_match in CLAIM_PATTERN.finditer(normalized):
            raw_claim = cl_match.group(1).lower()
            clean_claim = raw_claim.replace("(", "").replace(")", "")
            _add_token(f"claim-{clean_claim}")
            _add_token(f"claim_{clean_claim}")
            _add_token(f"claim-{raw_claim}")
            _add_token(clean_claim)
            num_part = re.sub(r"[^\d]", "", clean_claim)
            if num_part:
                _add_token(num_part)
                _add_token(f"claim-{num_part}")

        # 4. Product codes & alphanumeric compounds (e.g. GTX-4090, SKU-1234-X)
        for pc_match in PRODUCT_CODE_PATTERN.finditer(normalized):
            pc_str = pc_match.group(0).lower()
            _add_token(pc_str)
            # Add subwords for partial matching
            for sub in re.split(r"[-_/.]", pc_str):
                if len(sub) > 1 and (not remove_stopwords or sub not in SAFE_STOPWORDS):
                    _add_token(sub)

    # 5. General word tokens
    for w_match in WORD_PATTERN.finditer(normalized):
        word = w_match.group(0).lower()
        # Protect tokens that have digits or are already in seen
        if any(c.isdigit() for c in word):
            _add_token(word)
            continue
        if remove_stopwords and word in SAFE_STOPWORDS:
            continue
        if len(word) > 1 or word in ("r", "c", "x", "y", "z"):  # Allow variable letters
            _add_token(word)

    return tokens


def extract_search_terms(query: str) -> Dict[str, Any]:
    """Extract search tokens, phrases, and identifiers from user query for BM25 search.

    Returns:
        Dictionary containing:
          - 'tokens': List of BM25 search tokens
          - 'phrases': List of exact quoted phrases
          - 'identifiers': List of detected patent/product identifiers
    """
    clean_query = normalize_text(query)
    remaining_text, phrases = extract_phrases(clean_query)
    tokens = tokenize(clean_query, remove_stopwords=True, preserve_identifiers=True)

    # Detect patent/product identifiers
    identifiers: List[str] = []
    for p_match in PATENT_PATTERN.finditer(clean_query):
        identifiers.append(p_match.group(0))
    for pc_match in PRODUCT_CODE_PATTERN.finditer(clean_query):
        identifiers.append(pc_match.group(0))

    return {
        "tokens": tokens,
        "phrases": phrases,
        "identifiers": identifiers,
    }
