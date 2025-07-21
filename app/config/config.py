import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    DB_CONFIG = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME')
    }
    BACKEND_URL = os.getenv('BACKEND_URL', 'localhost')
    BACKEND_PORT = int(os.getenv('BACKEND_PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 't')
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'DEV')
    MODEL_API_KEY = os.getenv('MODEL_API_KEY', '')
    