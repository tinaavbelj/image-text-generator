import os
from dotenv import load_dotenv
load_dotenv('.env')

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
GOOGLE_SEARCH_ENGINE_ID = os.environ.get('GOOGLE_SEARCH_ENGINE_ID')
