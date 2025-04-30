import re
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict
from fd_linguistic_features import (
    AUXILIARY_VERBS, COMMON_EXPRESSIONS, CONJUNCTIONS, TEMPORAL_WORDS,
    ADVERBS_OPINION, SENTENCE_STARTERS, ADVERB_PATTERNS, FEATURE_DESCRIPTIONS,
    PAST_VERB_PATTERNS, GERUND_PATTERNS
)
from colab_print import Printer, header, list_, table, dfd, info, error, success, warning, title


class FeatureExtractor:
    """
    Extracts linguistic features from sentences based on predefined patterns and word lists.
    Used to analyze text for specific grammatical and structural elements.
    """
    def __init__(self):
        # Use imported word lists from fd_linguistic_features
        self.auxiliary_verbs = AUXILIARY_VERBS
        self.common_expressions = COMMON_EXPRESSIONS
        self.conjunctions = CONJUNCTIONS
        self.temporal_words = TEMPORAL_WORDS
        self.adverbs_opinion = ADVERBS_OPINION
        self.sentence_starters = SENTENCE_STARTERS
        self.adverb_patterns = ADVERB_PATTERNS
        self.past_verb_patterns = PAST_VERB_PATTERNS
        self.gerund_patterns = GERUND_PATTERNS
    
    def extract_features(self, sentence: str) -> Dict[str, bool]:
        """
        Extracts linguistic features from a given sentence.
        
        Args:
            sentence: The sentence to analyze
            
        Returns:
            Dictionary with feature names as keys and boolean values
        """
        # Convert to lowercase for case-insensitive matching, but keep original for punctuation
        original = sentence
        sentence = sentence.lower()
        
        # Split into words and remove punctuation for word-level features
        words = [re.sub(r'[^\w\s]', '', word) for word in sentence.split()]
        words = [word for word in words if word]  # Remove empty strings
        
        # Initialize features dictionary
        features = {}
        
        # Check for punctuation marks
        features["has_fullstop"] = "." in original
        features["has_question_mark"] = "?" in original
        features["has_exclamation_mark"] = "!" in original
        features["has_comma"] = "," in original
        features["has_semicolon"] = ";" in original
        features["has_colon"] = ":" in original
        features["has_quotation"] = '"' in original or "'" in original
        
        # Check for word-level features
        features["has_auxiliary"] = any(word in self.auxiliary_verbs for word in words)
        features["has_expression"] = any(word in self.common_expressions for word in words)
        features["has_conjunction"] = any(word in self.conjunctions for word in words)
        features["has_temporal"] = any(word in self.temporal_words for word in words)
        features["has_opinion_adverb"] = any(word in self.adverbs_opinion for word in words)
        features["has_starter"] = any(starter in sentence for starter in self.sentence_starters)
        
        # Check for pattern-based features
        features["has_adverb"] = any(re.search(pattern, sentence) for pattern in self.adverb_patterns)
        features["has_past_verb"] = any(re.search(pattern, sentence) for pattern in self.past_verb_patterns)
        features["has_gerund"] = any(re.search(pattern, sentence) for pattern in self.gerund_patterns)
        
        # Capitalization feature
        features["starts_capitalized"] = bool(original and original[0].isupper())
        
        return features


def preprocess_sentence(sentence: str) -> str:
    """
    Remove the starting '<name> commented' and ending 'on <date> on <platform>.' parts,
    and trim leading/trailing whitespaces.
    
    Args:
        sentence: The original sentence to process
        
    Returns:
        The cleaned sentence with patterns removed
    """
    # Remove the starting part '<name> commented'
    sentence = re.sub(r'^.*? commented\s*', '', sentence)
    
    # Remove the ending part 'on <date> on <platform>.'
    sentence = re.sub(r'\s*on \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} on .*?\.?$', '', sentence)
    
    # If the sentence was in quotes, remove only the outer quotes if they remain
    if sentence.startswith('"') and sentence.endswith('"'):
        sentence = sentence[1:-1]
    
    # Trim leading/trailing whitespaces
    return sentence.strip()


def expand_dataset(input_file: str, output_file: str) -> None:
    """
    Process the dataset, add linguistic feature columns, and save the expanded dataset.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the output CSV file
    """
    try:
        # Read the dataset
        info(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        
        if 'Sentence Fragment' not in df.columns:
            error(f"Input file must contain a 'Sentence Fragment' column.")
            return
        
        # Create a feature extractor
        extractor = FeatureExtractor()
        
        # Process each sentence and add features
        info(f"Extracting features from {len(df)} sentences...")
        
        # Initialize new columns with empty lists
        feature_columns = {
            "has_auxiliary": [],
            "has_fullstop": [],
            "has_question_mark": [],
            "has_exclamation_mark": [],
            "has_comma": [],
            "has_semicolon": [],
            "has_colon": [],
            "has_quotation": [],
            "has_expression": [],
            "has_conjunction": [],
            "has_temporal": [],
            "has_opinion_adverb": [],
            "has_adverb": [],
            "has_starter": [],
            "has_past_verb": [],
            "has_gerund": [],
            "starts_capitalized": []
        }
        
        # Process sentences with a progress bar
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sentences"):
            # Get and preprocess the sentence
            sentence = row['Sentence Fragment']
            processed_sentence = preprocess_sentence(sentence)
            
            # Extract features
            features = extractor.extract_features(processed_sentence)
            
            # Add each feature to its corresponding column
            for feature, value in features.items():
                feature_columns[feature].append(value)
        
        # Add feature columns to the DataFrame
        for feature, values in feature_columns.items():
            df[feature] = values
        
        # Use imported feature descriptions
        descriptions = FEATURE_DESCRIPTIONS
        
        # Save column descriptions as DataFrame metadata (only visible in code)
        for col, desc in descriptions.items():
            if col in df.columns:
                df[col].attrs['description'] = desc
        
        # Save to CSV
        info(f"Saving expanded dataset to: {output_file}")
        df.to_csv(output_file, index=False)
        
        # Print column descriptions for reference
        header("\nColumn Descriptions:")
        desc_list = []
        for col, desc in descriptions.items():
            if col in df.columns:
                desc_list.append(f"{col}: {desc}")
        list_(desc_list)
        
        success(f"Processing complete. Expanded dataset saved to {output_file}")
        
    except Exception as e:
        error(f"An error occurred: {str(e)}")