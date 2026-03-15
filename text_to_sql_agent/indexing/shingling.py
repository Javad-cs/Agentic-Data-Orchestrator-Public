"""Shingling utilities for LSH matching."""

from typing import Set
import re


def normalize_text(
    text: str, 
    remove_punctuation: bool = True,
    normalize_separators: bool = True,
) -> str:
    """
    Normalize text for consistent matching.
    
    Steps:
    1. Lowercase
    2. Normalize separators (underscore/hyphen to space) if enabled
    3. Strip leading/trailing whitespace
    4. Collapse multiple spaces
    5. Optionally remove punctuation
    
    Args:
        text: Input string
        remove_punctuation: Whether to remove punctuation
        normalize_separators: Whether to convert _ and - to spaces
            (Useful for matching "user_id" with "user id")
    
    Examples:
        "L.A." → "la"
        "St. Louis" → "st louis"
        "user_id" (with normalize_separators) → "user id"
        "San_Francisco" (with normalize_separators) → "san francisco"
        "first-name" (with normalize_separators) → "first name"
    """
    # Lowercase
    text = text.lower()
    
    # Normalize separators (underscore, hyphen → space)
    if normalize_separators:
        text = text.replace("_", " ")
        text = text.replace("-", " ")
    
    # Remove or replace punctuation
    if remove_punctuation:
        # Replace punctuation with space, then collapse
        text = re.sub(r'[^\w\s]', ' ', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip
    text = text.strip()
    
    return text


def create_shingles(
    text: str, 
    k: int = 3, 
    normalize: bool = True,
    normalize_separators: bool = True,
) -> Set[str]:
    """
    Create character-level k-shingles with proper fallback.
    
    Args:
        text: Input string
        k: Shingle size (default 3)
        normalize: Whether to normalize text first
        normalize_separators: Whether to normalize _ and - to spaces
        
    Returns:
        Set of shingles
        
    Examples:
        "hello" (k=3) → {"hel", "ell", "llo"}
        "hi" (k=3) → {"hi"}
        "L.A." (normalized, k=3) → {"la"}
        "user_id" (normalized, k=3) → {"use", "ser", "er ", "r i", " id"}
    """
    if normalize:
        text = normalize_text(text, normalize_separators=normalize_separators)
    
    if not text:
        return set()
    
    # Fallback for short strings
    if len(text) < k:
        return {text}
    
    # Standard k-shingles
    shingles = set()
    for i in range(len(text) - k + 1):
        shingles.add(text[i:i+k])
    
    return shingles


def stable_hash(text: str) -> str:
    """
    Create a stable, deterministic hash for a string.
    Unlike Python's hash(), this is consistent across runs.
    
    Args:
        text: String to hash
        
    Returns:
        Hex string (16 characters)
    """
    import hashlib
    return hashlib.blake2b(text.encode('utf-8'), digest_size=8).hexdigest()