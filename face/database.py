from mongoengine import connect
from urllib.parse import quote_plus
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import os

load_dotenv()

encrypted_password = os.getenv("MONGO_PASSWORD")
encryption_key = os.getenv("ENCRYPTION_KEY")
cipher_suite = Fernet(encryption_key.encode())
password = quote_plus(cipher_suite.decrypt(encrypted_password.encode()).decode())

username = quote_plus(os.getenv("MONGO_USERNAME"))
connection_string = f"mongodb+srv://{username}:{password}@cluster0.top9b.mongodb.net/?retryWrites=true&w=majority"

def connect_to_db():
    connect(db='test_db', host=connection_string)