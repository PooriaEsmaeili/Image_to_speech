from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())
HUGGINGACEHUB_API_TOKEN = os.getenv('HUGGINGACEHUB_API_TOKEN')
print(HUGGINGACEHUB_API_TOKEN)