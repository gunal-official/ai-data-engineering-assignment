import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the AI/Data Engineering Assignment"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    
    # Database Settings
    VECTOR_DB_PATH = "./milvus_demo.db"
    SQL_DB_URL = "sqlite:///./documents.db"
    
    # Model Settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    SPACY_MODEL = "en_core_web_sm"
    
    # File Paths
    DATA_DIR = "./data"
    SAMPLE_IMAGES_DIR = "./data/sample_images"
    PROCESSED_DIR = "./data/processed"
    
    # API Settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    CHAINLIT_PORT = 8001
    
    # Processing Settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.pdf', '.tiff', '.bmp']
    
    # Experiment Settings
    WANDB_PROJECT = "ai-data-engineering-assignment"
    
    @classmethod
    def validate_config(cls):
        """Validate that required configurations are set"""
        required_keys = ['OPENAI_API_KEY']  # Minimum required
        missing_keys = []
        
        for key in required_keys:
            if not getattr(cls, key):
                missing_keys.append(key)
        
        if missing_keys:
            print(f"⚠️  Missing API keys: {', '.join(missing_keys)}")
            print("Please set them in your .env file for full functionality")
        else:
            print("✅ Configuration validated successfully")
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.SAMPLE_IMAGES_DIR,
            cls.PROCESSED_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("✅ Directories created/verified")

# Document types for classification
DOCUMENT_TYPES = [
    'Advertisement',
    'Email', 
    'Form',
    'Letter',
    'Memo',
    'News',
    'Note',
    'Report',
    'Resume',
    'Scientific Paper'
]

# OCR Methods
OCR_METHODS = ['tesseract', 'paddleocr', 'easyocr']

# LLM Models  
LLM_MODELS = ['gpt-4', 'claude', 'gemini']

# Evaluation Metrics
EVALUATION_METRICS = {
    'ocr': ['cer', 'wer', 'bleu'],
    'classification': ['precision', 'recall', 'f1'],
    'summarization': ['rouge', 'bleu'],
    'retrieval': ['precision_at_k', 'recall_at_k', 'ndcg_at_k']
}