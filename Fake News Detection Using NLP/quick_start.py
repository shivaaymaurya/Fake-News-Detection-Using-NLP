#!/usr/bin/env python3
"""
Quick Start Script for Fake News Detection Project
Run this script to get started immediately!
"""

import os
import subprocess
import sys

def print_banner():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                FAKE NEWS DETECTION PROJECT                   ║
    ║                    Quick Start Guide                         ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    print("✅ Python version check passed")

def install_dependencies():
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def create_sample_data():
    """Create sample data for immediate testing"""
    sample_data = [
        {"text": "Breaking: Scientists discover cure for cancer using household items!", "label": 1},
        {"text": "Local hospital reports successful treatment of rare disease", "label": 0},
        {"text": "You won't believe what this celebrity did last night!", "label": 1},
        {"text": "Government announces new education policy for 2024", "label": 0},
        {"text": "Shocking truth about vaccines revealed by anonymous source", "label": 1},
        {"text": "Weather forecast predicts heavy rainfall in coastal areas", "label": 0}
    ]
    
    import pandas as pd
    df = pd.DataFrame(sample_data)
    df.to_csv('data/sample_data.csv', index=False)
    print("✅ Sample data created")

def create_baseline_model():
    """Create a simple baseline model"""
    print("🤖 Creating baseline model...")
    
    # Simple implementation
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    
    # Create pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('classifier', MultinomialNB())
    ])
    
    # Save model
    import joblib
    joblib.dump(model, 'models/baseline_model.pkl')
    print("✅ Baseline model created")

def main():
    print_banner()
    check_python_version()
    
    print("\n🚀 Starting Fake News Detection Project Setup...")
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Create requirements.txt if it doesn't exist
    if not os.path.exists('requirements.txt'):
        requirements = """
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
nltk
"""
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
    
    # Install dependencies
    install_dependencies()
    
    # Create sample data
    create_sample_data()
    
    # Create baseline model
    create_baseline_model()
    
    print("\n✅ Quick setup complete!")
    print("\nNext steps:")
    print("1. Run: python project_setup.py")
    print("2. Open: notebooks/main_analysis.ipynb")
    print("3. Follow the daily tracker in daily_tracker.md")
    print("\nHappy coding! 🎯")

if __name__ == "__main__":
    main()