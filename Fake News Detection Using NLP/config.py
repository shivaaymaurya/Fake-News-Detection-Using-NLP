
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
