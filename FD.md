# Fragment Detector Dataset Creator

This tool processes sentences to create a dataset for fragment detection in an autocomplete application. It extracts meaningful fragments from input sentences and labels them as fragments or complete sentences.

## Transformation Rules

The script applies the following transformation rules to each input sentence:

### 1. Preprocessing

- Removes the starting part: `<name> commented`
- Removes the ending part: `on <date> on <platform>.`
- Preserves text within quotation marks
- Trims leading/trailing whitespaces

### 2. Fragment Extraction

- **When commas are present**:
  - Splits the sentence at each comma
  - Creates a new row for each split part
  - Sets `is_fragment = True` for all parts

- **When no commas are present**:
  - Identifies the first occurrence of a special character (`.`, `!`, or `?`)
  - Keeps everything up to and including that character
  - Sets `is_fragment = False`
  - If no special character exists, keeps the entire text and sets `is_fragment = False`

### 3. Dataset Balancing

The script can balance the dataset based on the `is_fragment` column using two strategies:

- **Reduction Strategy**:
  - Reduces the number of majority class instances (usually `is_fragment = False`)
  - Randomly drops rows from the majority class until both classes have equal counts
  - No new data is created, but some data is removed

- **Expansion Strategy with SmartExpander**:
  - Increases the number of minority class instances (usually `is_fragment = True`)
  - Uses intelligent linguistic analysis to find natural split points in sentences
  - Splits sentences at meaningful linguistic points rather than just the middle
  - Creates more natural and contextually relevant fragments
  - All generated fragments are marked with `is_fragment = True`

#### SmartExpander Functionality

The `SmartExpander` class analyzes sentences for optimal split points based on linguistic cues:

1. **Split Point Priority**:
   - Common Expressions/Exclamations: `wow`, `amazing`, `fantastic`, etc.
   - Sentence Starters/Fillers: `well`, `actually`, `in fact`, etc.
   - Conjunctions: `and`, `but`, `or`, `because`, etc.
   - Auxiliary Verbs: `is`, `are`, `was`, `were`, etc.
   - Temporal Words: `then`, `after`, `before`, `suddenly`, etc.
   - Explicitly listed Adverbs of Opinion/Degree: `clearly`, `obviously`, `certainly`, etc.
   - **Regex-detected Adverbs**: Any word matching adverb patterns (see below)

2. **Adverb Pattern Detection**:
   - Detects adverbs using regex patterns rather than just a fixed list
   - Matches common adverb suffixes: 
     - Words ending with `ly` (e.g., carefully, quickly)
     - Words ending with `ily` (e.g., happily, easily)
     - Words ending with `ally` (e.g., basically, naturally)
     - Words ending with `ically` (e.g., specifically, dramatically)
   - Excludes common non-splitting adverbs like "very", "quite", "just", etc.
   - This allows for detection of thousands of potential adverbs without explicit listing

3. **Split Behavior**:
   - When a keyword or pattern match is found, the sentence is split at that point
   - Fragment handling ensures both fragments remain meaningful
   - Each fragment maintains proper sentence structure (with punctuation)
   - Fallback to middle-word splitting if no linguistic cues are found
   - Both fragments are marked as `is_fragment = True`

...

### 4. Output

The final output contains two columns:
- `Sentence Fragment`: The extracted text portion
- `is_fragment`: Boolean flag indicating whether the sentence is a fragment (True) or a complete sentence (False)

## Features

- **Smart Sentence Splitting**: Intelligently splits sentences at natural linguistic breakpoints
- **Pattern-based Adverb Detection**: Uses regex to identify adverbs dynamically without requiring an exhaustive list
- **Progress Tracking**: Uses tqdm progress bars to provide visual feedback for long-running operations
- **Detailed Logging**: Provides detailed information about the processing steps and dataset statistics
- **Flexible Balancing**: Offers two strategies for balancing imbalanced datasets

## Examples

### Example 1: Sentence with no commas

**Input:**
```
Raymond Stewart commented "Why isn't everyone talking about Daybreak? it breaks new ground" on 2021-08-26 14:03:32 on facebook.
```

**Processing Steps:**
1. Remove patterns → `Why isn't everyone talking about Daybreak? it breaks new ground`
2. No commas found, first special character is `?` → `Why isn't everyone talking about Daybreak?`
3. Set `is_fragment = False`

**Output:**

| Sentence Fragment                          | is_fragment |
|--------------------------------------------|-------------|
| Why isn't everyone talking about Daybreak? | False       |

### Example 2: Sentence with commas

**Input:**
```
Jane Doe commented "Coffee, tea, or juice, that is the question" on 2022-03-15 09:45:22 on twitter.
```

**Processing Steps:**
1. Remove patterns → `Coffee, tea, or juice, that is the question`
2. Split at commas → `Coffee`, `tea`, `or juice`, `that is the question`
3. Set `is_fragment = True` for all parts

**Output:**

| Sentence Fragment    | is_fragment |
|----------------------|-------------|
| Coffee               | True        |
| tea                  | True        |
| or juice             | True        |
| that is the question | True        |

### Example 3: Balancing with SmartExpander

**Input Non-Fragment:**
```
The graphics are breathtaking but the plot could be better.
```

**Intelligent Splitting Process:**
1. `SmartExpander` identifies `are` (auxiliary verb) and `but` (conjunction)
2. Based on priority, it splits first at `but` (conjunction has higher priority)
3. Creates two fragments with proper punctuation

**Output (New Fragments):**

| Sentence Fragment              | is_fragment |
|--------------------------------|-------------|
| The graphics are breathtaking. | True        |
| But the plot could be better.  | True        |

### Example 4: Regex-based Adverb Detection

**Input Non-Fragment:**
```
The actor performed brilliantly despite the weak script.
```

**Intelligent Splitting Process:**
1. `SmartExpander` detects `brilliantly` as an adverb via regex pattern (`\b\w+ly\b`)
2. Splits at the adverb position
3. Creates two fragments with proper punctuation

**Output (New Fragments):**

| Sentence Fragment                    | is_fragment |
|--------------------------------------|-------------|
| The actor performed.                 | True        |
| Brilliantly despite the weak script. | True        |

## Usage

### Prerequisites

- Python 3.6 or higher
- Required packages: pandas, tqdm, argparse, re

Install dependencies:
```bash
pip install pandas tqdm
```

### Running the Script

```bash
python fd_dataset_creator_script.py input_file.csv output_file.csv [--balance {reduce,expand}]
```

Where:
- `input_file.csv`: Path to the input CSV file (must contain a 'Sentence' column)
- `output_file.csv`: Path to save the output CSV file
- `--balance`: Optional strategy to balance the dataset:
  - `reduce`: Reduce majority class instances
  - `expand`: Create new minority class instances using intelligent splitting

## Edge Cases Handled

- **Multiple commas**: Each comma-separated part becomes a new row with `is_fragment = True`
- **No special characters**: If no `.`, `!`, or `?` is found, the entire text is kept with `is_fragment = False`
- **Quotes preservation**: Quotes within the actual sentence content are preserved
- **Empty fragments**: Empty fragments after splitting are ignored
- **Already balanced dataset**: If the dataset is already balanced, no changes are made
- **Insufficient data for expansion**: Only sentences with at least 4 words are split to create new fragments
- **Short fragments**: SmartExpander ensures fragments are meaningful by avoiding very short splits
- **Punctuation handling**: Added punctuation to fragments as needed to maintain proper sentence structure
- **Dynamic adverb detection**: Captures adverbs through pattern matching rather than requiring an exhaustive list 

## Recent Updates and Enhancements

### System Components

The Fragment Detector system now consists of four main components:

1. **fd_linguistic_features.py** - Shared module containing word lists, regex patterns, and feature descriptions.
2. **preprocessor.py** - *New*: Standardizes and cleans raw text using classic NLP techniques before further processing.
3. **fd_dataset_creator_script.py** - Original script enhanced, now benefits from upstream preprocessing.
4. **fd_ds_expander.py** - Script to expand datasets with linguistic features, also benefits from preprocessing.

### New: NLP Preprocessing Pipeline (`preprocessor.py`)

Before any fragment extraction or feature analysis, raw input sentences are now processed through a dedicated NLP preprocessing pipeline defined in `preprocessor.py`. This ensures consistency and improves the quality of data fed into downstream components.

The pipeline includes the following configurable steps:

1.  **Strip Platform Noise**: Removes patterns like `"User123 commented"` and `"on <date> on <platform>."`. (Based on previous `preprocess_sentence` logic).
2.  **Fix Broken Unicode**: Corrects common encoding errors (e.g., `Ã©` → `é`).
3.  **Remove Invalid Characters**: Strips HTML entities (`&nbsp;`), control characters (`\x00`), and unnecessary escape sequences.
4.  **Remove Emojis and Emoticons**: Uses the `emoji` library and regex to remove visual emojis (😂) and text emoticons (e.g., `:D`, `;-)`).
5.  **Lowercase Text**: Converts all text to lowercase for consistency.
6.  **Expand Contractions**: Converts forms like `"I'm"` to `"I am"`, `"won't"` to `"will not"` (Requires optional `contractions` library).
7.  **Normalize Unusual Word Forms**: Fixes specific artifacts like `"foot_ball"` → `"football"` or `"e-mail"` → `"email"`.
8.  **Normalize Punctuation**: Converts curly quotes (`""''`) to standard ones (`"'`), collapses repeated punctuation (`...` → `.`, `!!` → `!`), and fixes spacing around punctuation.
9.  **Normalize Whitespace**: Collapses multiple spaces, tabs, or newlines into a single space and trims leading/trailing whitespace.

**Optional Steps (Configurable):**

10. **Remove Stopwords**: Removes common function words like "the", "is", "in" (Uses NLTK).
11. **Tokenize Text**: Splits the cleaned text into individual words or tokens (Uses NLTK).
12. **Lemmatize Tokens**: Reduces words to their base or dictionary form (e.g., "running" → "run", "ran" → "run") (Uses NLTK WordNet).
13. **Correct Spelling**: Attempts to fix spelling errors (Requires optional `textblob` library; can be slow).

The order of operations is designed to handle dependencies (e.g., lowercasing before stopword removal).

### Enhanced Linguistic Pattern Recognition

The SmartExpander functionality has been extended with additional linguistic patterns:

1. **Past Verb Patterns** - Regular and irregular past tense verbs
   - Words ending with `-ed` or `-en` with at least 3 letters
   - Example: In "He walked to the store", "walked" is detected as a past verb

2. **Gerund Patterns** - Present participles ending with `-ing`
   - Words ending with `-ing` with at least 3 letters
   - Example: In "Running daily is healthy", "running" is detected as a gerund

3. **Special Pattern: Auxiliary + Gerund**
   - Detects auxiliary verbs followed immediately by gerunds (continuous tenses)
   - Example: In "Amr is playing football", it can split after "is" or at "playing"
   - This improves fragmentation of sentences with continuous verbs

### Example 5: Enhanced Verb Pattern Recognition

**Input Non-Fragment:**
```
Amr is playing football with his friends.
```

**Intelligent Splitting Process with New Patterns:**
1. `SmartExpander` now identifies `is` (auxiliary verb) followed by `playing` (gerund)
2. Special auxiliary+gerund handling prioritizes splitting at the gerund
3. Creates two fragments with proper punctuation

**Output (New Fragments):**

| Sentence Fragment                  | is_fragment |
|------------------------------------|-------------|
| Amr is.                            | True        |
| Playing football with his friends. | True        |

### Feature Extraction Functionality (`fd_ds_expander.py`)

The `fd_ds_expander.py` script adds linguistic feature columns to datasets after they have been preprocessed:

```bash
# First, preprocess the raw data (example)
# python preprocessor_script.py raw_input.csv preprocessed_output.csv

# Then, expand the preprocessed data with features
python fd_ds_expander.py --input preprocessed_output.csv --output expanded_features.csv
```

Adds 18 linguistic feature columns including:
- **has_past_verb** - Contains past tense verbs (words ending with -ed or -en)
- **has_gerund** - Contains gerunds/present participles (words ending with -ing)
- **has_auxiliary**, **has_fullstop**, **has_conjunction**, and many more

### Code Organization Improvements

- Extracted common patterns and word lists to `fd_linguistic_features.py`.
- Introduced `preprocessor.py` for standardized text cleaning.
- Removed redundant `all_caps_word` feature.
- Standardized feature extraction and processing across scripts.
- Added proper type hints and docstrings. 