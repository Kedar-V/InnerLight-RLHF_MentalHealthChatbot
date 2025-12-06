"""
Data preparation and processing utilities for InnerLight
"""

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def assign_risk_label(text):
    """
    Assign risk labels based on content analysis.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        str: Risk level ('safe', 'concerning', or 'high-risk')
    """
    text_lower = str(text).lower()

    # high-risk indicators
    high_risk_keywords = [
        'suicide', 'kill myself', 'end my life', 'want to die',
        'no reason to live', 'better off dead', 'suicidal',
        'overdose', 'plan to die'
    ]

    # concerning indicators
    concerning_keywords = [
        'hopeless', 'worthless', 'can\'t go on', 'give up',
        'nobody cares', 'alone', 'trapped', 'burden',
        'depressed', 'anxiety', 'panic', 'self-harm'
    ]

    # check for high-risk
    if any(keyword in text_lower for keyword in high_risk_keywords):
        return 'high-risk'

    # check for concerning
    if any(keyword in text_lower for keyword in concerning_keywords):
        return 'concerning'

    return 'safe'


def load_classification_data(csv_path):
    """
    Load and label classification dataset.
    
    Args:
        csv_path (str): Path to CSV file
        
    Returns:
        DataFrame: Loaded conversations with risk labels
    """
    df = pd.read_csv(csv_path)
    df['risk_label'] = df['Context'].apply(assign_risk_label)
    return df


def balance_classification_dataset(df, max_samples_per_class=5000):
    """
    Balance dataset by resampling classes.
    
    Args:
        df (DataFrame): Input dataframe with risk_label column
        max_samples_per_class (int): Maximum samples per class
        
    Returns:
        DataFrame: Balanced dataset
    """
    df_safe = df[df['risk_label'] == 'safe']
    df_concerning = df[df['risk_label'] == 'concerning']
    df_high_risk = df[df['risk_label'] == 'high-risk']

    n_samples = max(len(df_safe), len(df_concerning), len(df_high_risk))
    n_samples = min(n_samples, max_samples_per_class)

    df_safe_upsampled = resample(df_safe, n_samples=min(n_samples, len(df_safe)), random_state=42)
    df_concerning_upsampled = resample(df_concerning, n_samples=n_samples, random_state=42, replace=True)
    df_high_risk_upsampled = resample(df_high_risk, n_samples=n_samples, random_state=42, replace=True)

    df_balanced = pd.concat([df_safe_upsampled, df_concerning_upsampled, df_high_risk_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced


def prepare_classifier_dataset(df_balanced, test_size=0.3, val_size=0.5):
    """
    Prepare HuggingFace dataset for classification.
    
    Args:
        df_balanced (DataFrame): Balanced dataset with 'Context' and 'risk_label' columns
        test_size (float): Test set proportion
        val_size (float): Validation set proportion of remaining data
        
    Returns:
        tuple: (DatasetDict, label2id, id2label)
    """
    df_final = df_balanced[['Context', 'risk_label']].copy()
    df_final.columns = ['text', 'label']

    label2id = {'safe': 0, 'concerning': 1, 'high-risk': 2}
    id2label = {0: 'safe', 1: 'concerning', 2: 'high-risk'}

    df_final['label'] = df_final['label'].map(label2id)

    train_df, temp_df = train_test_split(
        df_final, test_size=test_size, random_state=42, 
        stratify=df_final['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=val_size, random_state=42, 
        stratify=temp_df['label']
    )

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'validation': Dataset.from_pandas(val_df, preserve_index=False),
        'test': Dataset.from_pandas(test_df, preserve_index=False)
    })

    return dataset, label2id, id2label


def prepare_generation_dataset(df_conversations, dataset_name="ShenLab/MentalChat16K"):
    """
    Prepare generation dataset combining conversations and mental health data.
    
    Args:
        df_conversations (DataFrame): Conversation data with 'LLM', 'Context', 'Response'
        dataset_name (str): HuggingFace dataset to load
        
    Returns:
        DataFrame: Combined and formatted generation dataset
    """
    SAFETY_INSTRUCTION = (
        "You are a compassionate crisis counselor. Provide empathetic, safe, and supportive responses. "
        "If the user expresses suicidal thoughts, gently encourage them to seek professional help "
        "and provide crisis resources like 988 Lifeline (US) or local emergency services."
    )

    # Prepare conversations dataset
    df_conv_gen = df_conversations.rename(columns={
        "LLM": "instruction",
        "Context": "input",
        "Response": "output"
    })
    df_conv_gen = df_conv_gen[["instruction", "input", "output"]]
    df_conv_gen["instruction"] = SAFETY_INSTRUCTION

    # Load and prepare MentalChat dataset
    try:
        ds_mental = load_dataset(dataset_name)
        df_mental = ds_mental["train"].to_pandas()
        
        df_mental_gen = df_mental.rename(columns={
            "Context": "input",
            "Response": "output"
        })
        df_mental_gen["instruction"] = SAFETY_INSTRUCTION
        df_mental_gen = df_mental_gen[["instruction", "input", "output"]]
        
        df_generation = pd.concat([df_conv_gen, df_mental_gen], ignore_index=True)
    except Exception as e:
        print(f"Could not load {dataset_name}: {e}")
        df_generation = df_conv_gen

    # Remove duplicates and shuffle
    df_generation = df_generation.drop_duplicates(subset=["input", "output"])
    df_generation = df_generation.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_generation
