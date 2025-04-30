# Fragma: Fragment Detection for Autocomplete Optimization

Fragma is a specialized model designed to detect sentence fragments for optimizing autocomplete systems. By identifying and classifying text fragments, Fragma helps autocomplete models provide more contextually relevant suggestions.

## Project Status

This project is currently in the **Model Development Phase**. We have completed the data preparation steps and are now moving to feature extraction and model development.

- See [STEPS.md](STEPS.md) for detailed project progress and upcoming tasks.

## Project Components

### 1. Fragment Detector Dataset Creator

The `fd_dataset_creator_script.py` is a preprocessing tool that builds the training dataset for Fragma. It extracts meaningful fragments from input sentences and labels them as fragments or complete sentences.

This component:
- Processes raw conversational data
- Applies intelligent splitting rules
- Balances the dataset for model training
- Generates labeled fragment/non-fragment pairs

### 2. Text Preprocessing Pipeline

The `preprocessor.py` module provides a comprehensive NLP preprocessing pipeline with metrics tracking:

- Unicode normalization and character cleaning
- HTML entity removal and whitespace normalization  
- Emoji and emoticon detection and removal
- Case normalization and contraction expansion
- Advanced tokenization with special handling for hashtags, quotes, and punctuation
- Detailed metrics collection for preprocessing steps
- Platform-specific noise removal

### 3. Fragma Model (Core)

The Fragma model leverages the processed dataset to learn patterns and characteristics of sentence fragments, enabling:
- Real-time fragment detection in user input
- Context-aware suggestion filtering
- Improved autocomplete relevance
- Reduced suggestion latency

## Documentation

For detailed information about the project:

- [Fragment Detection Documentation](FD.md) - Details about dataset creation process, transformation rules, and examples
- [STEPS.md](STEPS.md) - Project roadmap with completed and upcoming tasks
- Comprehensive docstrings in code files

## Quick Start

### Prerequisites

- Python 3.6 or higher
- Required packages:
  ```
  pandas
  tqdm
  nltk
  ftfy
  emoji
  textblob
  contractions
  ```

### Installation

Install dependencies:
```bash
pip install pandas tqdm nltk ftfy emoji textblob contractions
```

Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Dataset Creation

To prepare training data for the Fragma model:

```bash
python fd_dataset_creator_script.py input_file.csv output_file.csv [--balance {reduce,expand}]
```

Where:
- `input_file.csv`: Path to the input CSV file (must contain a 'Sentence' column)
- `output_file.csv`: Path to save the processed output CSV file
- `--balance`: Optional strategy to balance the dataset:
  - `reduce`: Reduce majority class instances
  - `expand`: Create new minority class instances using intelligent splitting 

### Text Preprocessing

To preprocess text data for model training:

```python
from preprocessor import preprocess_df

# Load your dataframe with a 'Sentence Fragment' column
processed_df, overall_metrics, instance_metrics = preprocess_df(df)

# Access the preprocessed text
processed_text = processed_df["Processed Text"]

# View preprocessing metrics
print(overall_metrics)
```

## Contributing

Contributions are welcome! Please check the roadmap in [STEPS.md](STEPS.md) to see what areas need attention. 