#!/usr/bin/env python3
"""
Fake News Detection Project Setup Script
This script helps you set up the entire project environment
"""

import os
import subprocess
import sys
import urllib.request
import zipfile

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def create_project_structure():
    """Create the project directory structure"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'src/preprocessing',
        'src/models',
        'src/web',
        'notebooks',
        'tests',
        'reports',
        'deployment'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_datasets():
    """Download necessary datasets"""
    print("Downloading datasets...")
    
    # URLs for datasets
    datasets = {
        'fake_news': {
            'url': 'https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/download',
            'filename': 'data/raw/fake_real_news.zip'
        },
        'liar_dataset': {
            'url': 'https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/liar_dataset.json',
            'filename': 'data/raw/liar_dataset.json'
        }
    }
    
    print("Note: Please manually download datasets from Kaggle using:")
    print("kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset")
    print("kaggle datasets download -d doanquanvietnamca/liar-dataset")

def create_requirements_txt():
    """Create requirements.txt file"""
    requirements = """
# Core ML Libraries
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0

# Deep Learning
torch==2.0.1
transformers==4.30.2
tensorflow==2.13.0

# NLP Libraries
nltk==3.8.1
spacy==3.6.1
textblob==0.17.1

# Web Framework
flask==2.3.2
streamlit==1.25.0
gunicorn==21.2.0

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
wordcloud==1.9.2

# Explainable AI
lime==0.2.0.1
shap==0.42.1

# Development
jupyter==1.0.0
ipykernel==6.25.0
pytest==7.4.0
black==23.7.0

# Utilities
tqdm==4.65.0
joblib==1.3.2
requests==2.31.0
beautifulsoup4==4.12.2
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("Created requirements.txt")

def create_main_notebook():
    """Create the main Jupyter notebook"""
    notebook_content = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Fake News Detection Using NLP\\n",
    "\\n",
    "## Project Overview\\n",
    "This notebook contains the complete implementation of a fake news detection system using various NLP techniques."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import all necessary libraries\\n",
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "from sklearn.model_selection import train_test_split\\n",
    "from sklearn.metrics import classification_report, confusion_matrix\\n",
    "import warnings\\n",
    "warnings.filterwarnings('ignore')\\n",
    "\\n",
    "# Set up plotting style\\n",
    "plt.style.use('seaborn-v0_8')\\n",
    "sns.set_palette('husl')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading and Exploration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load datasets\\n",
    "fake_df = pd.read_csv('../data/raw/Fake.csv')\\n",
    "true_df = pd.read_csv('../data/raw/True.csv')\\n",
    "\\n",
    "# Add labels\\n",
    "fake_df['label'] = 1\\n",
    "true_df['label'] = 0\\n",
    "\\n",
    "# Combine datasets\\n",
    "df = pd.concat([fake_df, true_df], ignore_index=True)\\n",
    "print(f&quot;Total samples: {len(df)}&quot;)\\n",
    "print(df['label'].value_counts())"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
    
    with open('notebooks/main_analysis.ipynb', 'w') as f:
        f.write(notebook_content)
    print("Created main_analysis.ipynb")

def create_config_file():
    """Create configuration file"""
    config = '''
# Configuration for Fake News Detection Project

# Data paths
RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data/processed'
MODELS_PATH = 'models'

# Model parameters
MAX_FEATURES = 5000
MAX_LENGTH = 512
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Training parameters
BERT_EPOCHS = 3
LSTM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 2e-5

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000
'''
    
    with open('config.py', 'w') as f:
        f.write(config)
    print("Created config.py")

def main():
    """Main setup function"""
    print("🚀 Setting up Fake News Detection Project...")
    print("=" * 50)
    
    try:
        # Create project structure
        create_project_structure()
        
        # Install required packages
        print("\n📦 Installing packages...")
        packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']
        for package in packages:
            install_package(package)
        
        # Create configuration files
        create_requirements_txt()
        create_config_file()
        create_main_notebook()
        
        # Download datasets
        download_datasets()
        
        print("\n✅ Project setup complete!")
        print("\nNext steps:")
        print("1. Install Kaggle CLI: pip install kaggle")
        print("2. Set up Kaggle API credentials")
        print("3. Download datasets using provided commands")
        print("4. Start with notebooks/main_analysis.ipynb")
        
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()