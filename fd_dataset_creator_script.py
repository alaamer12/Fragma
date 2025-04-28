import re
import pandas as pd
import random
from typing import Tuple, List, Dict, Optional, Union
from tqdm.auto import tqdm


class SmartExpander:
    def __init__(self):

        
        # 1. Auxiliary Verbs (verbs indicating tense, state, or activity)
        self.auxiliary_verbs = [
            "is", "are", "was", "were", "be", "been", "am", "being", 
            "has", "have", "had", "does", "do", "did"
        ]
        
        # 2. Common Expressions / Exclamations
        self.common_expressions = [
            "wow", "oops", "amazing", "fantastic", "unbelievable", "incredible", 
            "interesting", "sadly", "fortunately", "surprisingly", "finally"
        ]
        
        # 3. Conjunctions (links between ideas)
        self.conjunctions = [
            "and", "but", "or", "so", "because", "although", "however", 
            "yet", "still", "therefore", "meanwhile", "whereas"
        ]
        
        # 4. Temporal Words (expressions indicating time changes)
        self.temporal_words = [
            "then", "after", "before", "later", "suddenly", "soon", 
            "eventually", "earlier", "now", "today", "tonight", "tomorrow"
        ]
        
        # 5. Adverbs of Opinion / Degree - Common explicit ones
        # These are specific common adverbs we want to explicitly check for
        self.adverbs_opinion = [
            "clearly", "obviously", "probably", "certainly", "absolutely", 
            "seriously", "undoubtedly"
        ]
        
        # 6. Sentence Starters / Fillers
        self.sentence_starters = [
            "well", "so", "anyway", "besides", "actually", "by the way", 
            "in fact", "as a matter of fact"
        ]
        
        # Regex patterns for linguistic features
        self.adverb_patterns = [
            r'\b\w+ly\b',         # Words ending with 'ly' (carefully, quickly, etc.)
            r'\b\w+ily\b',        # Words ending with 'ily' (happily, easily, etc.)
            r'\b\w+ally\b',       # Words ending with 'ally' (basically, naturally, etc.)
            r'\b\w+ically\b',     # Words ending with 'ically' (specifically, dramatically, etc.)
        ]
        
        # Common adverbs that don't end with 'ly' but should be excluded from pattern matching
        self.non_splitting_adverbs = [
            "very", "quite", "rather", "too", "so", "just", "only",
            "almost", "nearly", "really", "pretty", "even"
        ]
        
        # Keyword categories in priority order
        self.keyword_categories = [
            self.common_expressions,
            self.sentence_starters,
            self.conjunctions,
            self.auxiliary_verbs,
            self.temporal_words,
            self.adverbs_opinion
        ]
    
    def is_adverb_by_pattern(self, word: str) -> bool:
        """
        Check if a word matches adverb patterns (typically ending with 'ly').
        Excludes words that are in the non-splitting adverbs list.
        
        Args:
            word: The word to check
            
        Returns:
            True if the word matches adverb patterns, False otherwise
        """
        # Clean the word from punctuation
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        
        # Skip checking if it's in our non-splitting adverbs list
        if clean_word in self.non_splitting_adverbs:
            return False
            
        # Check against all adverb patterns
        for pattern in self.adverb_patterns:
            if re.match(pattern, clean_word):
                return True
                
        return False
    
    def find_split_point(self, sentence: str) -> Optional[Tuple[int, str]]:
        """
        Find the optimal split point in a sentence based on keyword priority and regex patterns.
        
        Args:
            sentence: The sentence to analyze for split points
            
        Returns:
            A tuple (position, keyword) if a split point is found, None otherwise
        """
        # Convert to lowercase for case-insensitive matching
        words = sentence.lower().split()
        original_words = sentence.split()
        
        # Search for keywords in priority order
        for keyword_list in self.keyword_categories:
            for i, word in enumerate(words):
                # Clean the word from punctuation for matching
                clean_word = re.sub(r'[^\w\s]', '', word)
                
                if clean_word in keyword_list:
                    # Return the position and the original keyword
                    return i, original_words[i]
        
        # If no keyword match, check for adverbs by pattern
        for i, word in enumerate(words):
            if self.is_adverb_by_pattern(word):
                # We found an adverb by pattern
                return i, original_words[i]
        
        # No keyword or pattern found
        return None
    
    def split_sentence(self, sentence: str) -> List[str]:
        """
        Split a sentence intelligently into two fragments.
        
        Args:
            sentence: The sentence to split
            
        Returns:
            A list of two sentence fragments
        """
        # Find a split point based on keywords or regex patterns
        split_info = self.find_split_point(sentence)
        
        if split_info:
            # Split at the keyword or pattern match
            idx, keyword = split_info
            words = sentence.split()
            
            # Create two fragments: before and after the keyword (including it)
            fragment1 = ' '.join(words[:idx]).strip()
            fragment2 = ' '.join(words[idx:]).strip()
            
            # Handle very short fragments (minimum 2 words where possible)
            if len(fragment1.split()) < 2 and len(words) > 4:
                # If fragment1 is too short, move one more word to it
                fragment1 = ' '.join(words[:idx+1]).strip()
                fragment2 = ' '.join(words[idx+1:]).strip()
            
            # If fragment2 would be too short, keep it with fragment1
            if len(fragment2.split()) < 2 and len(words) > 4:
                fragment1 = sentence
                fragment2 = ""
        else:
            # Fallback: split at the middle word
            words = sentence.split()
            middle_idx = len(words) // 2
            
            fragment1 = ' '.join(words[:middle_idx]).strip()
            fragment2 = ' '.join(words[middle_idx:]).strip()
        
        # Clean up fragments: remove leading/trailing punctuation
        fragments = []
        for fragment in [fragment1, fragment2]:
            if fragment:  # Only process non-empty fragments
                # Remove leading punctuation except opening quotes
                fragment = re.sub(r'^[^\w"\']+', '', fragment)
                
                # If fragment starts with a quote but doesn't end with one, add it
                if (fragment.startswith('"') and not fragment.endswith('"')) or \
                   (fragment.startswith("'") and not fragment.endswith("'")):
                    pass  # The closing quote will be in the other fragment, which is fine
                    
                # Ensure sentences end with proper punctuation
                if not re.search(r'[.!?]$', fragment) and fragment:
                    fragment = fragment.rstrip(',;:-') + '.'
                    
                fragments.append(fragment)
        
        return [f for f in fragments if f]  # Return only non-empty fragments


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


def extract_fragments(sentence: str) -> List[Dict[str, bool]]:
    """
    Process the sentence and extract fragments according to the rules.
    
    Rules:
    - If the sentence contains a comma, split at each comma and mark each part as a fragment
    - If no comma, keep up to the first special character (., !, ?) and mark as not a fragment
    
    Args:
        sentence: The preprocessed sentence
        
    Returns:
        List of dictionaries with keys 'Sentence Fragment' and 'is_fragment'
    """
    fragments = []
    
    # Check if the sentence contains a comma
    if ',' in sentence:
        # Split at each comma
        parts = sentence.split(',')
        
        # Create a new row for each part and set is_fragment = True
        for part in parts:
            part = part.strip()
            if part:  # Only add if part is not empty
                fragments.append({'Sentence Fragment': part, 'is_fragment': True})
    else:
        # Find the first occurrence of special characters
        match = re.search(r'[.!?]', sentence)
        
        if match:
            # Get the position of the first special character
            pos = match.start()
            
            # Keep everything before and including the special character
            fragment = sentence[:pos+1].strip()
        else:
            # If no special character exists, keep the whole text
            fragment = sentence.strip()
            
        # Add to fragments list if not empty
        if fragment:
            fragments.append({'Sentence Fragment': fragment, 'is_fragment': False})
            
    return fragments


def balance_dataset(df: pd.DataFrame, strategy: str = "reduce") -> pd.DataFrame:
    """
    Balance the dataset based on the 'is_fragment' column using the specified strategy.
    
    Args:
        df: DataFrame containing the dataset to balance
        strategy: Strategy to use for balancing ('reduce' or 'expand')
        
    Returns:
        Balanced DataFrame
    """
    # Count the instances of each class
    true_count = df[df['is_fragment'] == True].shape[0]
    false_count = df[df['is_fragment'] == False].shape[0]
    
    print(f"Before balancing: {true_count} fragments, {false_count} non-fragments")
    
    # If already balanced, return the original DataFrame
    if true_count == false_count:
        print("Dataset is already balanced.")
        return df
    
    # Determine which class is the majority
    if strategy == "reduce":
        # Reduction strategy: reduce the number of majority class instances
        if true_count > false_count:
            # More fragments than non-fragments
            # Randomly drop rows where is_fragment = True
            drop_count = true_count - false_count
            drop_indices = df[df['is_fragment'] == True].sample(drop_count).index
            balanced_df = df.drop(drop_indices)
        else:
            # More non-fragments than fragments
            # Randomly drop rows where is_fragment = False
            drop_count = false_count - true_count
            drop_indices = df[df['is_fragment'] == False].sample(drop_count).index
            balanced_df = df.drop(drop_indices)
            
    elif strategy == "expand":
        # Expansion strategy: increase the number of minority class instances
        balanced_df = df.copy()
        
        if true_count < false_count:
            # More non-fragments than fragments, need to create more fragments
            # Target number of new fragments to create
            target_new = false_count - true_count
            new_fragments = []
            
            # Create a SmartExpander instance for intelligent splitting
            expander = SmartExpander()
            
            # Get non-fragment sentences to split
            non_fragments = df[df['is_fragment'] == False]['Sentence Fragment'].tolist()
            random.shuffle(non_fragments)  # Shuffle to randomize selection
            
            # Add progress bar for fragment expansion
            print(f"Creating new fragments to balance dataset...")
            for sentence in tqdm(non_fragments, total=None, desc="Expanding fragments", unit="sent", leave=True):
                if len(new_fragments) >= target_new:
                    break
                    
                # Only process sentences with at least 4 words
                if len(sentence.split()) >= 4:
                    # Use the SmartExpander to split the sentence
                    fragments = expander.split_sentence(sentence)
                    
                    # Add fragments to the new list
                    for fragment in fragments:
                        if fragment.strip():  # Only add if not empty
                            new_fragments.append({
                                'Sentence Fragment': fragment.strip(), 
                                'is_fragment': True
                            })
                            
                            # If we've reached our target, stop
                            if len(new_fragments) >= target_new:
                                break
            
            # Add new fragments to the dataset
            if new_fragments:
                new_df = pd.DataFrame(new_fragments)
                balanced_df = pd.concat([balanced_df, new_df], ignore_index=True)
        else:
            # More fragments than non-fragments, need to create more non-fragments
            # This case is more complex as creating valid non-fragments from fragments
            # is challenging. For now, we'll use the reduction strategy as a fallback.
            print("Warning: Expansion from fragments to non-fragments is not supported. Falling back to reduction strategy.")
            drop_count = true_count - false_count
            drop_indices = df[df['is_fragment'] == True].sample(drop_count).index
            balanced_df = df.drop(drop_indices)
    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}. Use 'reduce' or 'expand'.")
    
    # Count the instances of each class after balancing
    true_count_after = balanced_df[balanced_df['is_fragment'] == True].shape[0]
    false_count_after = balanced_df[balanced_df['is_fragment'] == False].shape[0]
    
    print(f"After balancing: {true_count_after} fragments, {false_count_after} non-fragments")
    
    return balanced_df


def process_dataset(input_file: str, output_file: str, balance_strategy: str = None) -> None:
    """
    Process the dataset file, apply transformations, and save the results.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the output CSV file
        balance_strategy: Strategy to use for balancing the dataset ('reduce', 'expand', or None for no balancing)
    """
    try:
        # Read the dataset
        print(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        
        if 'Sentence' not in df.columns:
            print(f"Error: Input file must contain a 'Sentence' column.")
            return
        
        # Process each sentence and collect results
        all_fragments = []
        
        print(f"Processing {len(df)} sentences...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sentences"):
            # Get the current sentence
            sentence = row['Sentence']
            
            # Preprocess the sentence
            processed_sentence = preprocess_sentence(sentence)
            
            # Extract fragments
            fragments = extract_fragments(processed_sentence)
            
            # Add each fragment to the results
            all_fragments.extend(fragments)
        
        # Create a new DataFrame with the results
        print("Creating output DataFrame...")
        result_df = pd.DataFrame(all_fragments)
        
        # Balance the dataset if a strategy is specified
        if balance_strategy:
            print(f"Balancing dataset using '{balance_strategy}' strategy...")
            result_df = balance_dataset(result_df, balance_strategy)
        
        # Save to CSV
        print(f"Saving results to: {output_file}")
        result_df.to_csv(output_file, index=False)
        
        print(f"Processing complete. Output saved to {output_file}")
        print(f"Processed {len(df)} original sentences into {len(result_df)} fragments.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
