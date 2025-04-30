import re
import pandas as pd
import random
from typing import Tuple, List, Dict, Optional, Union
from tqdm.auto import tqdm
from fd_linguistic_features import (
    AUXILIARY_VERBS, COMMON_EXPRESSIONS, CONJUNCTIONS, TEMPORAL_WORDS,
    ADVERBS_OPINION, SENTENCE_STARTERS, ADVERB_PATTERNS, NON_SPLITTING_ADVERBS,
    PAST_VERB_PATTERNS, GERUND_PATTERNS
)
from colab_print import Printer, header, list_, table, dfd, info, error, success, warning, title


class SmartExpander:
    def __init__(self):
        # Use imported word lists from fd_linguistic_features
        self.auxiliary_verbs = AUXILIARY_VERBS
        self.common_expressions = COMMON_EXPRESSIONS 
        self.conjunctions = CONJUNCTIONS
        self.temporal_words = TEMPORAL_WORDS
        self.adverbs_opinion = ADVERBS_OPINION
        self.sentence_starters = SENTENCE_STARTERS
        self.adverb_patterns = ADVERB_PATTERNS
        self.non_splitting_adverbs = NON_SPLITTING_ADVERBS
        self.past_verb_patterns = PAST_VERB_PATTERNS
        self.gerund_patterns = GERUND_PATTERNS
        
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
    
    def is_past_verb(self, word: str) -> bool:
        """
        Check if a word matches past verb patterns (ending with 'ed' or 'en').
        
        Args:
            word: The word to check
            
        Returns:
            True if the word is likely a past tense verb, False otherwise
        """
        # Clean the word from punctuation
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        
        # Check against all past verb patterns
        for pattern in self.past_verb_patterns:
            if re.match(pattern, clean_word):
                return True
                
        return False
    
    def is_gerund(self, word: str) -> bool:
        """
        Check if a word matches gerund patterns (ending with 'ing').
        
        Args:
            word: The word to check
            
        Returns:
            True if the word is likely a gerund, False otherwise
        """
        # Clean the word from punctuation
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        
        # Check against all gerund patterns
        for pattern in self.gerund_patterns:
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
        
        # If no keyword match, check for special patterns in this order:
        # 1. First check for auxiliary verbs followed by gerunds (e.g., "is playing")
        for i, word in enumerate(words[:-1]):  # Skip the last word
            clean_word = re.sub(r'[^\w\s]', '', word)
            next_word = re.sub(r'[^\w\s]', '', words[i+1])
            
            if clean_word in self.auxiliary_verbs and self.is_gerund(next_word):
                # Found auxiliary + gerund pattern (e.g., "is playing")
                # Split at the gerund to keep the auxiliary with the first part
                return i+1, original_words[i+1]
        
        # 2. Check for gerunds
        for i, word in enumerate(words):
            if self.is_gerund(word):
                return i, original_words[i]
        
        # 3. Check for past verbs
        for i, word in enumerate(words):
            if self.is_past_verb(word):
                return i, original_words[i]
        
        # 4. Finally check for adverbs
        for i, word in enumerate(words):
            if self.is_adverb_by_pattern(word):
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


def balance_dataset(df: pd.DataFrame, strategy: str = "reduce", 
                   keep_balancing: Optional[str] = None, tolerance: Optional[str] = None) -> pd.DataFrame:
    """
    Balance the dataset based on the 'is_fragment' column using the specified strategy.
    Optionally maintain a target balance percentage even after duplicate removal.
    
    Args:
        df: DataFrame containing the dataset to balance
        strategy: Strategy to use for balancing ('reduce' or 'expand')
        keep_balancing: Target balance percentage as string (e.g., "95%"). If provided, 
                       iteratively balance and remove duplicates until target is reached.
        tolerance: Acceptable imbalance as string (e.g., "1%"). If the improvement between
                  iterations is less than this value, stop iterating.
        
    Returns:
        Balanced DataFrame
    """
    # Extract target percentage if keep_balancing is provided
    target_percentage = None
    if keep_balancing:
        # Extract the percentage value (remove % symbol and convert to float)
        try:
            target_percentage = float(keep_balancing.strip('%'))
            if target_percentage <= 0 or target_percentage > 100:
                warning(f"Invalid target percentage {target_percentage}. Using 95% as default.")
                target_percentage = 95.0
        except ValueError:
            warning(f"Could not parse '{keep_balancing}' as percentage. Using 95% as default.")
            target_percentage = 95.0
        
        info(f"Targeting {target_percentage}% balance after duplicate removal.")
    
    # Extract tolerance percentage if provided
    tolerance_percentage = None
    if tolerance:
        try:
            tolerance_percentage = float(tolerance.strip('%'))
            if tolerance_percentage <= 0 or tolerance_percentage > 10:  # Cap at 10% for safety
                warning(f"Invalid tolerance {tolerance_percentage}. Using 1% as default.")
                tolerance_percentage = 1.0
        except ValueError:
            warning(f"Could not parse '{tolerance}' as percentage. Using 1% as default.")
            tolerance_percentage = 1.0
        
        info(f"Will stop when balance improvement is less than {tolerance_percentage}%.")
    
    # Function to calculate current balance percentage
    def calculate_balance_percentage(dataframe):
        true_count = dataframe[dataframe['is_fragment'] == True].shape[0]
        false_count = dataframe[dataframe['is_fragment'] == False].shape[0]
        
        if true_count == 0 or false_count == 0:
            return 0.0
        
        return (min(true_count, false_count) / max(true_count, false_count)) * 100
    
    # Function to display class distribution
    def print_distribution(dataframe, label=""):
        true_count = dataframe[dataframe['is_fragment'] == True].shape[0]
        false_count = dataframe[dataframe['is_fragment'] == False].shape[0]
        balance_pct = calculate_balance_percentage(dataframe)
        
        info(f"{label}: {true_count} fragments, {false_count} non-fragments " +
              f"(balance: {balance_pct:.2f}%)")
    
    # Initial distribution
    print_distribution(df, "Before balancing")
    
    # If target_percentage is None, just balance once without iterative process
    if target_percentage is None:
        # Perform simple balancing as before
        balanced_df = _balance_dataset_once(df, strategy)
        print_distribution(balanced_df, "After balancing")
        return balanced_df
    
    # Iterative balancing to maintain target percentage after deduplication
    current_df = df.copy()
    iteration = 1
    max_iterations = 10  # Prevent infinite loops
    previous_balance = calculate_balance_percentage(current_df)
    
    while iteration <= max_iterations:
        header(f"\nIteration {iteration}:")
        
        # Step 1: Balance the dataset
        current_df = _balance_dataset_once(current_df, strategy)
        print_distribution(current_df, "After balancing")
        
        # Step 2: Remove duplicates
        original_count = len(current_df)
        current_df = current_df.drop_duplicates(subset=['Sentence Fragment']).reset_index(drop=True)
        removed_count = original_count - len(current_df)
        
        info(f"Removed {removed_count} duplicates")
        print_distribution(current_df, "After removing duplicates")
        
        # Step 3: Check if we've reached the target balance
        current_balance = calculate_balance_percentage(current_df)
        if current_balance >= target_percentage:
            success(f"Target balance of {target_percentage}% achieved!")
            break
        
        # Check if the improvement is less than the tolerance
        if tolerance_percentage is not None:
            balance_improvement = current_balance - previous_balance
            info(f"Balance improvement: {balance_improvement:.2f}%")
            
            if balance_improvement < tolerance_percentage and balance_improvement >= 0:
                warning(f"Balance improvement ({balance_improvement:.2f}%) is less than tolerance ({tolerance_percentage}%).")
                info(f"Stopping iterations as we're close enough to optimal balance.")
                break
            
            # Update the previous balance for the next iteration
            previous_balance = current_balance
        
        # If we didn't reach the target but we're at the last iteration
        if iteration == max_iterations:
            warning(f"Could not achieve target balance of {target_percentage}% " +
                  f"after {max_iterations} iterations. Current balance: {current_balance:.2f}%")
        
        iteration += 1
    
    return current_df


def _balance_dataset_once(df: pd.DataFrame, strategy: str = "reduce") -> pd.DataFrame:
    """
    Helper function that performs one round of dataset balancing.
    
    Args:
        df: DataFrame containing the dataset to balance
        strategy: Strategy to use for balancing ('reduce' or 'expand')
        
    Returns:
        Balanced DataFrame
    """
    # Count the instances of each class
    true_count = df[df['is_fragment'] == True].shape[0]
    false_count = df[df['is_fragment'] == False].shape[0]
    
    # If already balanced, return the original DataFrame
    if true_count == false_count:
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
            info(f"Creating new fragments to balance dataset...")
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
            warning("Expansion from fragments to non-fragments is not supported. Falling back to reduction strategy.")
            drop_count = true_count - false_count
            drop_indices = df[df['is_fragment'] == True].sample(drop_count).index
            balanced_df = df.drop(drop_indices)
    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}. Use 'reduce' or 'expand'.")
    
    return balanced_df


def process_dataset(input_file: str, output_file: str, balance_strategy: str = None, 
                   keep_balancing: str = None, tolerance: str = None) -> None:
    """
    Process the dataset file, apply transformations, and save the results.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the output CSV file
        balance_strategy: Strategy to use for balancing the dataset ('reduce', 'expand', or None for no balancing)
        keep_balancing: Target balance percentage after duplicate removal (e.g., "95%")
        tolerance: Acceptable imbalance percentage (e.g., "1%"). If improvement between iterations
                  is less than this value, stop iterating.
    """
    try:
        # Read the dataset
        info(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        
        if 'Sentence' not in df.columns:
            error(f"Input file must contain a 'Sentence' column.")
            return
        
        # Process each sentence and collect results
        all_fragments = []
        
        info(f"Processing {len(df)} sentences...")
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
        info("Creating output DataFrame...")
        result_df = pd.DataFrame(all_fragments)
        
        # Balance the dataset if a strategy is specified
        if balance_strategy:
            info(f"Balancing dataset using '{balance_strategy}' strategy...")
            result_df = balance_dataset(result_df, balance_strategy, keep_balancing, tolerance)
        
        # Save to CSV
        info(f"Saving results to: {output_file}")
        result_df.to_csv(output_file, index=False)
        
        success(f"Processing complete. Output saved to {output_file}")
        info(f"Processed {len(df)} original sentences into {len(result_df)} fragments.")
        
    except Exception as e:
        error(f"An error occurred: {str(e)}")
