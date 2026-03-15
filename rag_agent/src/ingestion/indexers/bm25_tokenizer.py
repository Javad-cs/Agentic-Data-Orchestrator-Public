import re
from typing import List, Set, Dict
from collections import Counter


class BM25Tokenizer:
    """
    Tokenizer for BM25 indexing.
    
    Handles both English and Korean text.
    
    Features:
    - Lowercase normalization (optional)
    - Stopword removal (optional)
    - Korean/English/number handling
    - Punctuation removal
    """
    
    # English stopwords
    ENGLISH_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
        'what', 'when', 'where', 'who', 'which', 'why', 'how'
    }
    
    # Korean stopwords
    KOREAN_STOPWORDS = {
        '은', '는', '이', '가', '을', '를', '에', '와', '과', '도', '만',
        '의', '로', '으로', '에서', '부터', '까지', '에게', '한테', '께',
        '이다', '있다', '없다', '하다', '되다', '이', '그', '저', '것'
    }
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        min_token_length: int = 2,
        max_token_length: int = 50
    ):
        """
        Initialize tokenizer.
        
        Args:
            lowercase: Convert to lowercase
            remove_stopwords: Remove stopwords
            min_token_length: Minimum token length
            max_token_length: Maximum token length
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        
        # Combine stopwords
        self.stopwords = self.ENGLISH_STOPWORDS | self.KOREAN_STOPWORDS
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Build regex pattern based on lowercase setting
        if self.lowercase:
            # Lowercase first, then extract
            text = text.lower()
            # Pattern: Korean (Hangul) | lowercase English | numbers
            pattern = r'[\uac00-\ud7a3]+|[a-z]+|[0-9]+'
        else:
            # Pattern: Korean | English (both cases) | numbers
            pattern = r'[\uac00-\ud7a3]+|[A-Za-z]+|[0-9]+'
        
        # Extract tokens
        tokens = re.findall(pattern, text, re.UNICODE)
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Apply lowercase if needed (for case-insensitive comparison with stopwords)
            # but we already lowercased above if lowercase=True
            token_for_comparison = token.lower() if not self.lowercase else token
            
            # Length filter
            if len(token) < self.min_token_length or len(token) > self.max_token_length:
                continue
            
            # Stopword filter (compare lowercased version)
            if self.remove_stopwords and token_for_comparison in self.stopwords:
                continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def tokenize_with_frequencies(self, text: str) -> Dict[str, int]:
        """
        Tokenize and count term frequencies.
        
        Args:
            text: Input text
            
        Returns:
            Dict mapping term to frequency
        """
        tokens = self.tokenize(text)
        return dict(Counter(tokens))
    
    def get_unique_terms(self, text: str) -> Set[str]:
        """
        Get unique terms from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of unique terms
        """
        return set(self.tokenize(text))