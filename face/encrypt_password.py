from cryptography.fernet import Fernet

key = Fernet.generate_key()
print(f"Save this key securely: {key.decode()}")

cipher_suite = Fernet(key)
encrypted_password = cipher_suite.encrypt(b"Ayush@310803")
print(f"Encrypted Password: {encrypted_password.decode()}")