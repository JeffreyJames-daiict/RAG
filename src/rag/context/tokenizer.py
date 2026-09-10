"""Token calculation, budgeting, and text slicing utilities using tiktoken with fallback."""

from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False


class ContextTokenizer:
    """Accurate token estimator and text truncator backed by tiktoken with heuristic fallback."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self._encoder = None
        if _HAS_TIKTOKEN:
            try:
                self._encoder = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(
                    "Could not load tiktoken encoding '%s' (%s). Falling back to cl100k_base or heuristics.",
                    encoding_name,
                    e,
                )
                try:
                    self._encoder = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self._encoder = None

    def count_tokens(self, text: Optional[str]) -> int:
        """Calculate the exact or estimated token count of a given string."""
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text, disallowed_special=()))
            except Exception as e:
                logger.debug("tiktoken encoding error: %s. Using heuristic fallback.", e)

        # Heuristic fallback: average ~3.8 characters per token for English / code
        char_estimate = int(len(text) / 3.8)
        word_estimate = int(len(text.split()) * 1.3)
        return max(1, max(char_estimate, word_estimate))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> Tuple[str, int]:
        """Truncate text so its token count does not exceed max_tokens.

        Returns:
            Tuple of (truncated_text, final_token_count)
        """
        if not text:
            return "", 0
        if max_tokens <= 0:
            return "", 0

        current_count = self.count_tokens(text)
        if current_count <= max_tokens:
            return text, current_count

        if self._encoder is not None:
            try:
                tokens = self._encoder.encode(text, disallowed_special=())
                if len(tokens) > max_tokens:
                    truncated_tokens = tokens[:max_tokens]
                    truncated_str = self._encoder.decode(truncated_tokens)
                    return truncated_str, len(truncated_tokens)
            except Exception as e:
                logger.debug("tiktoken truncation failed (%s), using char fallback", e)

        # Character ratio fallback
        target_chars = int(max_tokens * 3.5)
        sliced = text[:target_chars]
        # Avoid splitting words awkwardly if space is nearby
        last_space = sliced.rfind(" ")
        if last_space > int(target_chars * 0.8):
            sliced = sliced[:last_space]
        final_count = self.count_tokens(sliced)
        return sliced, final_count
