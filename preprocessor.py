import re
import unicodedata
import html
from typing import List, Optional, Dict, Tuple, Any, Union
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ftfy
from colab_print import header, table, info, error, success, warning
from contractions import fix
import emoji
from textblob import TextBlob
from tqdm import tqdm
import pandas as pd
import numpy as np

# --- Core Preprocessing Functions ---
def fix_broken_unicode(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Fixes common unicode corruption issues (e.g., Mojibake).
    Uses ftfy library to automatically detect and fix any broken unicode.

    Args:
        text: The input string potentially containing unicode errors.

    Returns:
        Tuple of (cleaned string, metrics dictionary)
    """
    metrics = {"original": text, "total_chars": len(text), "fixed": 0, "failed": 0}
    
    # Check for unicode issues by comparing with fixed version
    fixed_text = ftfy.fix_text(text)
    # Apply additional normalization for compatibility
    fixed_text = unicodedata.normalize('NFKC', fixed_text)
    
    # Count differences
    if fixed_text != text:
        diff_count = sum(1 for a, b in zip(text, fixed_text) if a != b)
        diff_count += abs(len(text) - len(fixed_text))
        metrics["fixed"] = diff_count
    
    # Calculate how many potential issues existed
    metrics["issues"] = metrics["fixed"]
    metrics["failed"] = 0  # We assume ftfy fixes everything it can
    metrics["percentage"] = (metrics["issues"] / metrics["total_chars"] * 100) if metrics["total_chars"] > 0 else 0
    
    return fixed_text, metrics

def remove_invalid_chars(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Removes HTML entities, null bytes, escape codes, and control characters.

    Args:
        text: The input string.

    Returns:
        Tuple of (cleaned string, metrics dictionary)
    """
    metrics = {"original": text, "total_chars": len(text), "fixed": 0, "failed": 0}
    
    # Count HTML entities
    html_entities_count = len(re.findall(r'&[a-zA-Z0-9#]+;', text))
    
    # Remove HTML entities
    text_html_cleaned = html.unescape(text)
    
    # Count control characters (except whitespace)
    control_chars_count = len(re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', text_html_cleaned))
    
    # Count escape sequences like \x1A
    escape_seq_count = text_html_cleaned.count('\x1A')
    
    # Count HTML tags like <br>
    html_tags_count = len(re.findall(r'<br\s*/?>', text_html_cleaned))
    
    # Calculate total issues
    total_issues = html_entities_count + control_chars_count + escape_seq_count + html_tags_count
    
    # Remove control characters (except whitespace like \n, \t, \r)
    cleaned_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text_html_cleaned)
    # Remove specific escape sequences if necessary (e.g., \x1A)
    cleaned_text = cleaned_text.replace('\x1A', '')
    # Remove specific HTML tags like <br> if they weren't caught by unescape
    cleaned_text = re.sub(r'<br\s*/?>', ' ', cleaned_text)
    
    # Calculate how many issues were fixed (difference in length)
    original_len = len(text)
    cleaned_len = len(cleaned_text)
    metrics["issues"] = total_issues
    metrics["fixed"] = total_issues  # We assume all identified issues are fixed
    metrics["failed"] = 0  # Assuming the regex fixes everything it matches
    metrics["percentage"] = (metrics["issues"] / metrics["total_chars"] * 100) if metrics["total_chars"] > 0 else 0
    
    return cleaned_text, metrics

def normalize_whitespace(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Collapses multiple whitespace characters into a single space
    and trims leading/trailing whitespace.

    Args:
        text: The input string.

    Returns:
        Tuple of (string with normalized whitespace, metrics dictionary)
    """
    metrics = {"original": text, "total_chars": len(text), "fixed": 0, "failed": 0}
    
    # Count multiple whitespace occurrences
    multi_whitespace_count = len(re.findall(r'\s{2,}', text))
    
    # Count leading/trailing whitespace
    leading_trailing_count = 1 if text.strip() != text else 0
    
    # Total issues
    total_issues = multi_whitespace_count + leading_trailing_count
    
    # Replace multiple whitespace chars (space, tab, newline, etc.) with a single space
    normalized_text = re.sub(r'\s+', ' ', text)
    # Trim leading/trailing whitespace
    normalized_text = normalized_text.strip()
    
    metrics["issues"] = total_issues
    metrics["fixed"] = total_issues  # We assume all whitespace issues are fixed
    metrics["failed"] = 0  # Assuming the regex fixes everything it matches
    metrics["percentage"] = (metrics["issues"] / metrics["total_chars"] * 100) if metrics["total_chars"] > 0 else 0
    
    return normalized_text, metrics

def remove_emojis_emoticons(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Removes emojis and common text-based emoticons.

    Args:
        text: The input string.

    Returns:
        Tuple of (string with emojis and emoticons removed, metrics dictionary)
    """
    metrics = {"original": text, "total_chars": len(text), "fixed": 0, "failed": 0}
    emoji_count = 0
    emoticon_count = 0
    
    try:
        # Count emojis
        emoji_count = emoji.emoji_count(text)
        # Try to use emoji library for emoji removal
        cleaned_text = emoji.replace_emoji(text, replace='')
    except ImportError:
        # Fallback to regex pattern for basic emoji detection if emoji library is not available
        try:
            # Unicode ranges for common emoji categories
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F700-\U0001F77F"  # alchemical symbols
                "\U0001F780-\U0001F7FF"  # Geometric Shapes
                "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                "\U0001FA00-\U0001FA6F"  # Chess Symbols
                "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                "\U00002702-\U000027B0"  # Dingbats
                "\U000024C2-\U0001F251" 
                "]", flags=re.UNICODE)
            emoji_matches = emoji_pattern.findall(text)
            emoji_count = len(emoji_matches)
            cleaned_text = emoji_pattern.sub(r'', text)
        except Exception as e:
            # If all else fails, log warning and return original text
            warning(f"Failed to remove emojis: {str(e)}")
            cleaned_text = text
            metrics["failed"] += emoji_count

    # Define common emoticons (add more as needed)
    emoticon_pattern = r"""
        (?:
          [:=;] # Eyes
          ['`\-]? # Optional nose
          [)\]\(\[dDpP/:}{@|\\] # Mouth
        )|(?:
          [)\]\(\[dDpP/:}{@|\\] # Mouth
          ['`\-]? # Optional nose
          [:=;] # Eyes
        )|(?:
            <3 # Heart
        )|(?:
            \^_\^ # Happy face
        )|(?:
            \(o\.o\) # Surprised face
        )|(?:
            -_- # Annoyed face
        )|(?:
            ;p # Winking tongue out
        )
    """
    
    # Count emoticons
    emoticon_matches = re.findall(emoticon_pattern, cleaned_text, flags=re.VERBOSE | re.IGNORECASE)
    emoticon_count = len(emoticon_matches)
    
    # Remove emoticons
    final_text = re.sub(emoticon_pattern, '', cleaned_text, flags=re.VERBOSE | re.IGNORECASE)
    
    # Update metrics
    total_issues = emoji_count + emoticon_count
    metrics["issues"] = total_issues
    metrics["fixed"] = total_issues - metrics["failed"]
    metrics["percentage"] = (metrics["issues"] / metrics["total_chars"] * 100) if metrics["total_chars"] > 0 else 0
    
    return final_text, metrics

def lowercase_text(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Converts the entire text to lowercase.

    Args:
        text: The input string.

    Returns:
        Tuple of (lowercased string, metrics dictionary)
    """
    metrics = {"original": text, "total_chars": len(text), "fixed": 0, "failed": 0}
    
    # Count uppercase characters
    uppercase_count = sum(1 for char in text if char.isupper())
    
    lowercased = text.lower()
    
    metrics["issues"] = uppercase_count
    metrics["fixed"] = uppercase_count  # All uppercase chars are converted
    metrics["failed"] = 0  # .lower() should never fail
    metrics["percentage"] = (metrics["issues"] / metrics["total_chars"] * 100) if metrics["total_chars"] > 0 else 0
    
    return lowercased, metrics

def normalize_unusual_word_forms(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Handles specific word normalization tasks like merging hyphenated words,
    underscore-separated words, slash-separated alternatives, and other
    common text artifacts.

    Args:
        text: The input string.

    Returns:
        Tuple of (string with unusual forms normalized, metrics dictionary)
    """
    metrics = {"original": text, "total_chars": len(text), "fixed": 0, "failed": 0}
    
    # Count underscore-separated words
    underscore_count = len(re.findall(r'\b(\w+)_(\w+)\b', text))
    
    # Count hyphenated words
    hyphen_count = len(re.findall(r'\b(\w+)-(\w+)\b', text, flags=re.IGNORECASE))
    
    # Count slash-separated alternatives
    slash_count = len(re.findall(r'\b(\w+)/(\w+)\b', text, flags=re.IGNORECASE))
    
    # Count specific common cases
    replacements = {
        r'\bcovid-19\b': 'covid',
        r'\be-mail\b': 'email',
        r'\bwifi\b': 'wifi',
        r'\bwi-fi\b': 'wifi',
        r'\bwi_fi\b': 'wifi',
        r'\be-commerce\b': 'ecommerce',
        r'\bt-shirt\b': 'tshirt',
        r'\bt_shirt\b': 'tshirt',
    }
    
    specific_cases_count = sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) 
                             for pattern in replacements.keys())
    
    # Calculate total issues
    total_issues = underscore_count + hyphen_count + slash_count + specific_cases_count
    
    # Handle underscore-separated words (e.g., foot_ball → football)
    processed_text = re.sub(r'\b(\w+)_(\w+)\b', r'\1\2', text)
    
    # Handle hyphenated words (e.g., foot-ball → football)
    processed_text = re.sub(r'\b(\w+)-(\w+)\b', r'\1\2', processed_text, flags=re.IGNORECASE)
    
    # Handle slash-separated alternatives (e.g., football/soccer → football)
    # This keeps the first alternative and removes the second
    processed_text = re.sub(r'\b(\w+)/(\w+)\b', r'\1', processed_text, flags=re.IGNORECASE)
    
    # Apply specific common cases
    for pattern, replacement in replacements.items():
        processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
    
    metrics["issues"] = total_issues
    metrics["fixed"] = total_issues  # We assume all identified issues are fixed
    metrics["failed"] = 0  # Assuming the regex fixes everything it matches
    metrics["percentage"] = (metrics["issues"] / metrics["total_chars"] * 100) if metrics["total_chars"] > 0 else 0
    
    return processed_text, metrics

def expand_contractions(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Expands contractions like "I'm" to "I am".
    Requires the 'contractions' library.

    Args:
        text: The input string.

    Returns:
        Tuple of (string with contractions expanded, metrics dictionary)
    """
    metrics = {"original": text, "total_chars": len(text), "fixed": 0, "failed": 0}
    
    # Common contractions patterns for estimation
    contraction_pattern = r"(?i)\b(won't|can't|don't|isn't|haven't|hasn't|hadn't|couldn't|shouldn't|wouldn't|aren't|weren't|wasn't|didn't|that's|what's|it's|i'm|i've|i'll|i'd|you're|you've|you'll|you'd|he's|he'll|he'd|she's|she'll|she'd|we're|we've|we'll|we'd|they're|they've|they'll|they'd|there's|here's|who's|who'll|who'd|what's|where's|when's|why's|how's|let's|that's)\b"
    
    # Count contractions
    contractions_matches = re.findall(contraction_pattern, text)
    contractions_count = len(contractions_matches)
    
    try:
        expanded_text = fix(text)
        metrics["issues"] = contractions_count
        metrics["fixed"] = contractions_count  # Assuming all contractions are fixed
        metrics["failed"] = 0
    except ImportError:
        warning("Optional library 'contractions' not installed. Skipping contraction expansion.")
        expanded_text = text
        metrics["issues"] = contractions_count
        metrics["fixed"] = 0
        metrics["failed"] = contractions_count
    except Exception as e:
        error(f"Error during contraction expansion: {e}")
        expanded_text = text  # Return original text on error
        metrics["issues"] = contractions_count
        metrics["fixed"] = 0
        metrics["failed"] = contractions_count
    
    metrics["percentage"] = (metrics["issues"] / metrics["total_chars"] * 100) if metrics["total_chars"] > 0 else 0
    
    return expanded_text, metrics

def normalize_punctuation(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Normalizes different types of quotes and repeated punctuation.
    Fixes spacing around punctuation.

    Args:
        text: The input string.

    Returns:
        Tuple of (string with normalized punctuation, metrics dictionary)
    """
    metrics = {"original": text, "total_chars": len(text), "fixed": 0, "failed": 0}
    
    # Count abnormal quotes
    curly_quotes_count = text.count('"') + text.count('"') + text.count("'") + text.count("'")
    
    # Count repeated punctuation
    repeated_periods = len(re.findall(r'\.{2,}', text))
    repeated_exclamation = len(re.findall(r'!{2,}', text))
    repeated_question = len(re.findall(r'\?{2,}', text))
    
    # Count spacing issues
    missing_space_after = len(re.findall(r'([.!?])([a-zA-Z0-9])', text))
    extra_space_before = len(re.findall(r'\s+([,.!?])', text))
    missing_space_after_comma = len(re.findall(r'(,)([a-zA-Z0-9])', text))
    
    # Calculate total issues
    total_issues = (curly_quotes_count + repeated_periods + repeated_exclamation + 
                  repeated_question + missing_space_after + extra_space_before + 
                  missing_space_after_comma)
    
    # Normalize curly quotes to standard quotes
    processed_text = text.replace('"', '"').replace('"', '"')
    processed_text = processed_text.replace("'", "'").replace("'", "'")

    # Collapse multiple periods (e.g., ellipsis) into one, preserving sentence boundary indication
    processed_text = re.sub(r'\.{2,}', '.', processed_text)
    # Collapse multiple exclamation marks or question marks
    processed_text = re.sub(r'!{2,}', '!', processed_text)
    processed_text = re.sub(r'\?{2,}', '?', processed_text)

    # Ensure space after sentence-ending punctuation if followed by a letter/number
    processed_text = re.sub(r'([.!?])([a-zA-Z0-9])', r'\1 \2', processed_text)
    # Remove space before sentence-ending punctuation or commas
    processed_text = re.sub(r'\s+([,.!?])', r'\1', processed_text)
    # Ensure space after commas if followed by a letter/number
    processed_text = re.sub(r'(,)([a-zA-Z0-9])', r'\1 \2', processed_text)
    
    metrics["issues"] = total_issues
    metrics["fixed"] = total_issues  # We assume all identified issues are fixed
    metrics["failed"] = 0  # Assuming the regex fixes everything it matches
    metrics["percentage"] = (metrics["issues"] / metrics["total_chars"] * 100) if metrics["total_chars"] > 0 else 0
    
    return processed_text, metrics

def remove_stopwords(tokens: List[str], language: str = 'english') -> Tuple[List[str], Dict[str, Any]]:
    """
    Removes common stopwords from a list of tokens.
    Requires NLTK stopwords data.

    Args:
        tokens: A list of word tokens.
        language: The language of the stopwords list to use.

    Returns:
        Tuple of (list of tokens with stopwords removed, metrics dictionary)
    """
    metrics = {"original": tokens, "total_tokens": len(tokens), "fixed": 0, "failed": 0}
    
    try:
        stop_words = set(stopwords.words(language))
        stopword_count = sum(1 for token in tokens if token in stop_words)
        
        filtered_tokens = [token for token in tokens if token not in stop_words]
        
        metrics["issues"] = stopword_count
        metrics["fixed"] = stopword_count
        metrics["failed"] = 0
        metrics["percentage"] = (metrics["issues"] / metrics["total_tokens"] * 100) if metrics["total_tokens"] > 0 else 0
        
        return filtered_tokens, metrics
    except LookupError:
        warning(f"Stopwords for '{language}' not found. Ensure NLTK data is downloaded.")
        metrics["issues"] = 0
        metrics["fixed"] = 0
        metrics["failed"] = 0
        metrics["percentage"] = 0
        return tokens, metrics
    except Exception as e:
         error(f"Error removing stopwords: {e}")
         metrics["issues"] = 0
         metrics["fixed"] = 0
         metrics["failed"] = 0
         metrics["percentage"] = 0
         return tokens, metrics

def strip_platform_noise(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Removes common platform-specific noise patterns like timestamps or user comments.
    Adapts logic from fd_dataset_creator_script.preprocess_sentence.

    Args:
        text: The input string.

    Returns:
        Tuple of (string with platform noise removed, metrics dictionary)
    """
    metrics = {"original": text, "total_chars": len(text), "fixed": 0, "failed": 0}
    
    # Count platform noise patterns
    comment_pattern = r'^.*? commented\s*'
    date_pattern = r'\s*on \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} on .*?\.?$'
    outer_quotes = False
    
    comment_matches = re.findall(comment_pattern, text, flags=re.IGNORECASE)
    comment_count = len(comment_matches)
    
    date_matches = re.findall(date_pattern, text, flags=re.IGNORECASE)
    date_count = len(date_matches)
    
    if len(text) >= 2 and ((text.startswith('"') and text.endswith('"')) or 
                          (text.startswith("'") and text.endswith("'"))):
        outer_quotes = True
    
    total_issues = comment_count + date_count + (1 if outer_quotes else 0)
    
    # Remove the starting part '<name> commented' (case-insensitive)
    processed_text = re.sub(r'^.*? commented\s*', '', text, flags=re.IGNORECASE)
    # Remove the ending part 'on <date> on <platform>.' (case-insensitive)
    processed_text = re.sub(r'\s*on \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} on .*?\.?$', '', processed_text, flags=re.IGNORECASE)

    # If the sentence was in quotes, remove only the outer quotes if they remain
    # Handle potential errors if text becomes too short after stripping
    if len(processed_text) >= 2 and processed_text.startswith('"') and processed_text.endswith('"'):
        processed_text = processed_text[1:-1]
    elif len(processed_text) >= 2 and processed_text.startswith("'") and processed_text.endswith("'"):
         processed_text = processed_text[1:-1]

    metrics["issues"] = total_issues
    metrics["fixed"] = total_issues  # We assume all identified issues are fixed
    metrics["failed"] = 0
    metrics["percentage"] = (metrics["issues"] / metrics["total_chars"] * 100) if metrics["total_chars"] > 0 else 0
    
    return processed_text.strip(), metrics # Ensure trimming after removal

def tokenize_text(text: str, keep_punctuation: bool = False) -> Tuple[List[str], Dict[str, Any]]:
    """
    Tokenizes text into words using NLTK's tokenizer.
    Optionally removes punctuation tokens and performs additional cleaning.

    Args:
        text: The input string.
        keep_punctuation: If False, removes tokens that are purely punctuation.

    Returns:
        Tuple of (list of word tokens, metrics dictionary)
    """
    metrics = {"original": text, "total_chars": len(text), "fixed": 0, "failed": 0, "issues": 0}
    
    try:
        # Initial tokenization with NLTK
        initial_tokens = word_tokenize(text)
        token_count = len(initial_tokens)
        
        # Enhanced tokenization
        cleaned_tokens = []
        issues_count = 0
        
        for token in initial_tokens:
            # Handle hashtags - separate # from word
            if token.startswith('#') and len(token) > 1:
                if token[-1] in '.!?,;:)]}':  # If hashtag ends with punctuation
                    hashtag_word = token[1:-1]
                    cleaned_tokens.append('#')
                    cleaned_tokens.append(hashtag_word)
                    cleaned_tokens.append(token[-1])
                    issues_count += 1
                else:
                    hashtag_word = token[1:]
                    cleaned_tokens.append('#')
                    cleaned_tokens.append(hashtag_word)
                    issues_count += 1
                continue
            
            # Handle quotes
            if token.startswith('"') and len(token) > 1:
                cleaned_tokens.append('"')
                if token.endswith('"') and len(token) > 2:  # Double-quoted single word
                    cleaned_tokens.append(token[1:-1])
                    cleaned_tokens.append('"')
                    issues_count += 1
                else:
                    cleaned_tokens.append(token[1:])
                    issues_count += 1
                continue
            
            if token.endswith('"') and len(token) > 1 and not token.startswith('"'):
                cleaned_tokens.append(token[:-1])
                cleaned_tokens.append('"')
                issues_count += 1
                continue
            
            # Handle punctuation attached to words (like word. or word!)
            if any(token.endswith(p) for p in '.!?,;:') and len(token) > 1 and not token.startswith('#'):
                # Special case for abbreviations like U.S. or a.m.
                if re.match(r'^[a-zA-Z](\.[a-zA-Z])+\.$', token):
                    cleaned_tokens.append(token)
                    continue
                    
                # Special case for numerical patterns like dates, times, IP addresses
                if re.match(r'^[\d]+([-:/\.]\d+)+$', token):
                    cleaned_tokens.append(token)
                    continue
                
                # Regular case: separate punctuation from word
                base_word = token[:-1]
                punctuation = token[-1]
                cleaned_tokens.append(base_word)
                cleaned_tokens.append(punctuation)
                issues_count += 1
                continue
            
            # Handle punctuation at the beginning of words
            if any(token.startswith(p) for p in '([{') and len(token) > 1:
                cleaned_tokens.append(token[0])
                cleaned_tokens.append(token[1:])
                issues_count += 1
                continue
                
            # Add token as is if no special handling applied
            cleaned_tokens.append(token)
        
        # Filter out punctuation if requested
        if not keep_punctuation:
            punct_tokens = [token for token in cleaned_tokens if re.fullmatch(r'[\W_]+', token)]
            punct_count = len(punct_tokens)
            filtered_tokens = [token for token in cleaned_tokens if not re.fullmatch(r'[\W_]+', token)]
            issues_count += punct_count
            metrics["fixed"] = issues_count
            return filtered_tokens, metrics
        
        metrics["issues"] = issues_count
        metrics["fixed"] = issues_count
        metrics["failed"] = 0
        metrics["percentage"] = (metrics["issues"] / token_count * 100) if token_count > 0 else 0
        
        return cleaned_tokens, metrics
    except LookupError:
        warning("NLTK 'punkt' tokenizer model not found. Using basic split(). Ensure NLTK data is downloaded.")
        tokens = text.split()
        metrics["issues"] = 0
        metrics["fixed"] = 0
        metrics["failed"] = 0
        metrics["percentage"] = 0
        return tokens, metrics
    except Exception as e:
        error(f"Error during tokenization: {e}")
        tokens = text.split()
        metrics["issues"] = 0
        metrics["fixed"] = 0
        metrics["failed"] = 0
        metrics["percentage"] = 0
        return tokens, metrics

def correct_spelling(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Corrects spelling errors using TextBlob.
    Note: This can be slow and sometimes inaccurate.

    Args:
        text: The input string.

    Returns:
        Tuple of (string with spelling potentially corrected, metrics dictionary)
    """
    metrics = {"original": text, "total_chars": len(text), "fixed": 0, "failed": 0}
    
    try:
        # Create a TextBlob to detect potential misspellings
        blob = TextBlob(text)
        words = blob.words
        
        # Count potential misspellings (simplified approach)
        # Note: This is approximate as TextBlob doesn't expose misspelling detection separately
        corrected_blob = blob.correct()
        corrected_text = str(corrected_blob)
        
        # Count changes
        if corrected_text != text:
            word_diff = sum(1 for w1, w2 in zip(blob.words, corrected_blob.words) if w1 != w2)
            metrics["issues"] = word_diff
            metrics["fixed"] = word_diff
            metrics["failed"] = 0
        else:
            metrics["issues"] = 0
            metrics["fixed"] = 0
            metrics["failed"] = 0
            
        metrics["percentage"] = (metrics["issues"] / len(words) * 100) if words else 0
        
        return corrected_text, metrics
    except ImportError:
        warning("Optional library 'textblob' not installed. Skipping spell correction.")
        metrics["issues"] = 0
        metrics["fixed"] = 0
        metrics["failed"] = 0
        metrics["percentage"] = 0
        return text, metrics
    except Exception as e:
        error(f"Error during spell correction: {e}")
        metrics["issues"] = 0
        metrics["fixed"] = 0
        metrics["failed"] = 0
        metrics["percentage"] = 0
        return text, metrics # Return original text on error


# --- Main Preprocessing Pipeline ---

DEFAULT_PREPROCESSING_CONFIG = {
    "fix_unicode": True,
    "remove_invalid": True,
    "normalize_whitespace": True,
    "remove_emojis": True,
    "lowercase": True,
    "normalize_words": True,
    "expand_contractions": True, 
    "normalize_punctuation": True,
    "strip_platform_noise": True,
    "remove_stopwords": False, # Optional step
    "tokenize": False,         # Optional step (returns list if True)
    "keep_punctuation_tokens": False, # Only if tokenize is True
    "spell_correct": False     # Optional step (expensive)
}

def preprocess_text(text: str, config: Optional[dict] = None) -> Union[str, List[str], Tuple[Union[str, List[str]], Dict[str, Dict[str, Any]]]]:
    """
    Applies a sequence of NLP preprocessing steps to the input text based on configuration.

    Args:
        text: The raw input string.
        config: A dictionary specifying which preprocessing steps to apply.
                Defaults to DEFAULT_PREPROCESSING_CONFIG.

    Returns:
        If metrics=True in config:
            Tuple of (processed text or tokens, metrics dictionary)
        Else:
            The processed text as a string, or a list of tokens if config["tokenize"] is True.
    """
    if not isinstance(text, str):
        warning(f"Input is not a string (type: {type(text)}), attempting conversion.")
        try:
            text = str(text)
        except Exception:
             error("Failed to convert input to string. Returning empty string.")
             return ""

    cfg = DEFAULT_PREPROCESSING_CONFIG.copy()
    if config:
        cfg.update(config)
    
    collect_metrics = cfg.get("collect_metrics", True)
    all_metrics = {}

    processed_text = text
    
    # Define which steps to show in tqdm based on enabled config
    enabled_steps = [(key, value) for key, value in cfg.items() 
                    if value is True and key in DEFAULT_PREPROCESSING_CONFIG]
    
    # Create a progress bar for the preprocessing steps
    for step_name, _ in tqdm(enabled_steps, desc="Preprocessing steps", leave=False):
        if step_name == "strip_platform_noise" and cfg["strip_platform_noise"]:
            processed_text, metrics = strip_platform_noise(processed_text)
            if collect_metrics:
                all_metrics["strip_platform_noise"] = metrics
            # info("Stripped platform noise.")

        elif step_name == "fix_unicode" and cfg["fix_unicode"]:
            processed_text, metrics = fix_broken_unicode(processed_text)
            if collect_metrics:
                all_metrics["fix_unicode"] = metrics
            # info("Fixed unicode.")

        elif step_name == "remove_invalid" and cfg["remove_invalid"]:
            processed_text, metrics = remove_invalid_chars(processed_text)
            if collect_metrics:
                all_metrics["remove_invalid"] = metrics
            # info("Removed invalid characters.")

        elif step_name == "remove_emojis" and cfg["remove_emojis"]:
            processed_text, metrics = remove_emojis_emoticons(processed_text)
            if collect_metrics:
                all_metrics["remove_emojis"] = metrics
            # info("Removed emojis and emoticons.")

        # Lowercasing often comes before or after contraction expansion depending on the library
        elif step_name == "lowercase" and cfg["lowercase"]:
             processed_text, metrics = lowercase_text(processed_text)
             if collect_metrics:
                 all_metrics["lowercase"] = metrics
             # info("Lowercased text.")

        elif step_name == "expand_contractions" and cfg["expand_contractions"]:
            processed_text, metrics = expand_contractions(processed_text)
            if collect_metrics:
                all_metrics["expand_contractions"] = metrics
            # info("Expanded contractions.")

        elif step_name == "normalize_words" and cfg["normalize_words"]:
            processed_text, metrics = normalize_unusual_word_forms(processed_text)
            if collect_metrics:
                all_metrics["normalize_words"] = metrics
            # info("Normalized unusual word forms.")

        elif step_name == "normalize_punctuation" and cfg["normalize_punctuation"]:
            processed_text, metrics = normalize_punctuation(processed_text)
            if collect_metrics:
                all_metrics["normalize_punctuation"] = metrics
            # info("Normalized punctuation.")

        # Whitespace normalization is often best done near the end
        elif step_name == "normalize_whitespace" and cfg["normalize_whitespace"]:
            processed_text, metrics = normalize_whitespace(processed_text)
            if collect_metrics:
                all_metrics["normalize_whitespace"] = metrics
            # info("Normalized whitespace.")

    # --- Optional Steps (Potentially Token-Based) ---
    tokens = None
    if cfg["tokenize"] or cfg["remove_stopwords"]:
        tokens, metrics = tokenize_text(processed_text, keep_punctuation=cfg["keep_punctuation_tokens"])
        if collect_metrics:
            all_metrics["tokenize"] = metrics
        # info(f"Tokenized text into {len(tokens)} tokens.")

    if cfg["remove_stopwords"] and tokens is not None:
        original_count = len(tokens)
        tokens, metrics = remove_stopwords(tokens)
        if collect_metrics:
            all_metrics["remove_stopwords"] = metrics
        # info(f"Removed {original_count - len(tokens)} stopwords.")

    # Decide final output format
    if tokens is not None and cfg["tokenize"]:
         # If tokenization was the goal, return tokens
         final_output = tokens
         # info("Preprocessing complete. Returning tokens.")
    elif tokens is not None:
         # If tokenization was intermediate, join back to string
         final_output = ' '.join(tokens)
         # info("Preprocessing complete. Returning processed string from tokens.")
    else:
         # If no tokenization occurred, use the processed string
         final_output = processed_text
         # info("Preprocessing complete. Returning processed string.")
    # Spell correction is usually last and operates on the string
    if cfg["spell_correct"]:
         if isinstance(final_output, list):
             warning("Spell correction requested but output is tokens. Joining tokens to perform correction.")
             final_output = ' '.join(final_output)
         final_output, metrics = correct_spelling(final_output)
         if collect_metrics:
             all_metrics["spell_correct"] = metrics
         # info("Applied spell correction.")
    
    if collect_metrics:
        return final_output, all_metrics
    else:
        return final_output

def preprocess_df(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applies a sequence of NLP preprocessing steps to each row of a DataFrame.

    Args:
        df: A pandas DataFrame containing text data.
        config: A dictionary specifying which preprocessing steps to apply.

    Returns:
        Tuple of (processed_df, overall_metrics_df, instance_metrics_df):
            - processed_df: DataFrame with added "Processed Text" column
            - overall_metrics_df: DataFrame with aggregated metrics for all instances
            - instance_metrics_df: DataFrame with metrics for each instance
    """
    df_copy = df.copy()
    if "Sentence Fragment" not in df_copy.columns:
        raise ValueError("DataFrame must contain a 'Sentence Fragment' column.")
    
    # Ensure metrics are collected
    if config is None:
        config = {}
    config["collect_metrics"] = True
    
    # Process each text and collect metrics
    processed_texts = []
    all_metrics = []
    
    info("Starting preprocessing pipeline...")
    # Use tqdm for progress tracking
    for idx, row in tqdm(df_copy.iterrows(), total=len(df_copy), desc="Processing DataFrame", leave=False):
        text = row["Sentence Fragment"]
        result, metrics = preprocess_text(text, config)
        processed_texts.append(result)
        
        # Add row identifier to metrics
        instance_metrics = {
            "instance_id": idx
        }
        
        # Flatten the metrics dictionary
        for step, step_metrics in metrics.items():
            for metric_key, metric_value in step_metrics.items():
                if metric_key in ["issues", "fixed", "failed", "percentage"]:
                    instance_metrics[f"{step}_{metric_key}"] = metric_value
        
        all_metrics.append(instance_metrics)
    
    # Add processed text to dataframe
    df_copy["Processed Text"] = processed_texts
    
    # Create instance metrics dataframe
    df_instances_matrix = pd.DataFrame(all_metrics)
    
    # Create overall metrics dataframe (aggregated)
    # Get all metric columns (excluding instance_id)
    metric_columns = [col for col in df_instances_matrix.columns if col != "instance_id"]
    
    # Get unique preprocessing steps from the column names
    step_names = set()
    for col in metric_columns:
        parts = col.split('_')
        if len(parts) >= 2:  # Ensure we have step name and metric type
            # The step name is everything except the last part (which is the metric type)
            step_name = '_'.join(parts[:-1])
            step_names.add(step_name)
    
    # Calculate aggregated statistics for each metric
    overall_data = []
    
    for step in step_names:
        step_data = {"step": step}
        
        # Find all columns for this step
        for metric_type in ["issues", "fixed", "failed", "percentage"]:
            col_name = f"{step}_{metric_type}"
            if col_name in df_instances_matrix.columns:
                # Calculate statistics
                step_data[f"{metric_type}_total"] = df_instances_matrix[col_name].sum()
                step_data[f"{metric_type}_mean"] = df_instances_matrix[col_name].mean()
                step_data[f"{metric_type}_median"] = df_instances_matrix[col_name].median()
                step_data[f"{metric_type}_min"] = df_instances_matrix[col_name].min()
                step_data[f"{metric_type}_max"] = df_instances_matrix[col_name].max()
        
        overall_data.append(step_data)
    
    df_overall_matrix = pd.DataFrame(overall_data)
    
    # Add summary statistics
    if len(df_overall_matrix) > 0:
        # Calculate totals across all steps
        total_row = {"step": "TOTAL"}
        for col in df_overall_matrix.columns:
            if col != "step" and "_total" in col:
                total_row[col] = df_overall_matrix[col].sum()
            elif col != "step" and any(stat in col for stat in ["_mean", "_median", "_min", "_max"]):
                # For aggregate metrics, we'll take the mean
                total_row[col] = df_overall_matrix[col].mean()
        
        # Append total row
        df_overall_matrix = pd.concat([df_overall_matrix, pd.DataFrame([total_row])], ignore_index=True)
    
    return df_copy, df_overall_matrix, df_instances_matrix
