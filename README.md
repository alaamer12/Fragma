# Fragma: Fragment Detection for Autocomplete Optimization

Fragma is a specialized model designed to detect sentence fragments for optimizing autocomplete systems. By identifying and classifying text fragments, Fragma helps autocomplete models provide more contextually relevant suggestions.

## Project Components

### 1. Fragment Detector Dataset Creator

The `fd_dataset_creator_script.py` is a preprocessing tool that builds the training dataset for Fragma. It extracts meaningful fragments from input sentences and labels them as fragments or complete sentences.

This component:
- Processes raw conversational data
- Applies intelligent splitting rules
- Balances the dataset for model training
- Generates labeled fragment/non-fragment pairs

### 2. Fragma Model (Core)

The Fragma model leverages the processed dataset to learn patterns and characteristics of sentence fragments, enabling:
- Real-time fragment detection in user input
- Context-aware suggestion filtering
- Improved autocomplete relevance
- Reduced suggestion latency

## Documentation

For detailed information about the dataset creation process, including:
- Transformation rules
- SmartExpander functionality 
- Pattern-based adverb detection
- Examples and usage
- Edge cases handled

Please refer to the [Fragment Detection Documentation](FD.md).

## Quick Start

### Prerequisites

- Python 3.6 or higher
- Required packages: pandas, tqdm, argparse, re

Install dependencies:
```bash
pip install pandas tqdm
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