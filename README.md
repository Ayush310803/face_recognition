### Face Recognition API
This project is a FastAPI-based application that provides endpoints for saving face images, recognizing faces from images, and processing video streams for face recognition. The application uses MongoDB for storing face data and face_recognition library for face detection and recognition.

## Table of Contents
1. Prerequisites
2. Setting Up the Virtual Environment
3. Setting Up MongoDB Connection
4. Installing Requirements
5. Setting Up Environment Variables
6. Running the Application

# Prerequisites
1. Python 3.7 or higher
2. MongoDB Atlas account or local MongoDB instance
3. pip package manager

# Setting Up the Virtual Environment
1. Navigate to the face folder:
  cd face_recognition/face

2. Create a virtual environment:
  python -m venv venv

3. Activate the virtual environment:
   on linux:
   source venv/bin/activate

   on windows:
   venv\Scripts\activate

# Setting Up MongoDB Connection
1. Create a MongoDB Atlas cluster or use a local MongoDB instance.

2. Get the connection string from MongoDB Atlas or use the local connection string.

3. Encrypt your MongoDB password using Fernet encryption. You can use the following Python code to generate an encrypted password:
   from cryptography.fernet import Fernet

   key = Fernet.generate_key()
   cipher_suite = Fernet(key)
   encrypted_password = cipher_suite.encrypt(b'your_mongodb_password')
   print(encrypted_password.decode())

4. Store the encryption key and encrypted password in a .env file.

# Installing Requirements
1. Install the required packages:
   pip install -r requirements.txt

   The requirements.txt file should include the following packages:
    annotated-types==0.7.0
    anyio==4.8.0
    cffi==1.17.1
    click==8.1.8
    cryptography==44.0.2
    dlib==19.24.6
    dnspython==2.7.0
    face-recognition==1.3.0
    fastapi==0.115.11
    h11==0.14.0
    idna==3.10
    mongoengine==0.29.1
    numpy==2.2.3
    opencv-python==4.11.0.86
    pillow==11.1.0
    pycparser==2.22
    pydantic==2.10.6
    pydantic_core==2.27.2
    pymongo==4.11.1
    python-dotenv==1.0.1
    sniffio==1.3.1
    SQLAlchemy==2.0.38
    starlette==0.46.0
    typing_extensions==4.12.2
    uvicorn==0.34.0

# Setting Up Environment Variables
1. Create a .env file in the face directory.

2. Add the following environment variables to the .env file:
   MONGO_USERNAME=your_mongodb_username
   MONGO_PASSWORD=your_encrypted_mongodb_password
   ENCRYPTION_KEY=your_encryption_key

   Replace your_mongodb_username, your_encrypted_mongodb_password, and your_encryption_key with the appropriate values.

# Running the Application
1. Start the FastAPI application:
   uvicorn main:app --reload

2. Access the API at http://127.0.0.1:8000

3. Interactive API documentation can be accessed at http://127.0.0.1:8000/docs