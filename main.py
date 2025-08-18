from src.ai_tutor_app import AITutorApp
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

demo = AITutorApp()
demo.launch()

