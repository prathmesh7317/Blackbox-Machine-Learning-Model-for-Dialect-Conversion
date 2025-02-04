# Blackbox-Machine-Learning-Model-for-Dialect-Conversion

### Instructions to Run the Notebook :-

1. Open the Google Colab notebook.
2. Install the required dependencies (see below).
3. Load the dataset and preprocess the text.
4. Train the model or use OpenAI for inference.
5. Evaluate results using validation metrics.
6. Run the inference pipeline to test dialect conversion.
7. Dependencies and Installation

### Ensure the following libraries are installed before running the notebook:

!pip install openai transformers datasets pandas scikit-learn evaluate

### Import necessary modules:

import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration,TrainingArguments,Trainer
import re
from sklearn.model_selection import train_test_split
import evaluate
import openai
from openai import OpenAI
from google.colab import userdata
import warnings
warnings.filterwarnings('ignore')


### Known Limitations and Potential Improvements
1. Limited Dataset Size: Small training data may lead to incomplete learning and inaccurate outputs.
2. Colab GPU Time Constraints: Training for more epochs is restricted by session time limits.
3. Pretrained Model Limitations: T5 may still require fine-tuning for domain-specific text.
4. Handling of Contextual Variations: Some phrases may require manual review to ensure accuracy.

### Alternative Approaches in Case of Time Constraints
1. Use OpenAI’s GPT Model: If fine-tuning is not feasible, leverage OpenAI’s API for accurate conversions.
2. Rule-Based Approach: Implement a dictionary-based method for common UK-to-US spelling differences.
