# Face Recognition API
This project is a FastAPI-based application that provides endpoints for saving face images, recognizing faces from images, and processing video streams for face recognition. The application uses MongoDB for storing face data and face_recognition library for face detection and recognition.

---

## Table of Contents
1. Prerequisites
2. Setting Up the Virtual Environment
3. Setting Up MongoDB Connection
4. Installing Requirements
5. Setting Up Environment Variables
6. Running the Application

---

### Prerequisites
1. Python 3.7 or less than equal to 3.10
2. MongoDB Atlas account or local MongoDB instance
3. pip package manager

---

### Setting Up the Virtual Environment
1. Navigate to the face folder:

   ```
   cd face_recognition/face
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   ```

3. Activate the virtual environment:

   on linux:
   ```
   source venv/bin/activate
   ```
   on windows:
   ```
   venv\Scripts\activate
   ```
---   

### Setting Up MongoDB Connection
1. Create a MongoDB Atlas cluster or use a local MongoDB instance.

2. Get the connection string from MongoDB Atlas or use the local connection string.

3. Encrypt your MongoDB password using Fernet encryption. You can use the following Python code to generate an encrypted password:
```
   from cryptography.fernet import Fernet

   key = Fernet.generate_key()

   cipher_suite = Fernet(key)

   encrypted_password = cipher_suite.encrypt(b'your_mongodb_password')

   print(encrypted_password.decode())
```

4. Store the encryption key and encrypted password in a .env file.

---

### Installing Requirements
1. Install the required packages:

   pip install -r requirements.txt

---

### Setting Up Environment Variables
1. Create a .env file in the face directory.

2. Add the following environment variables to the .env file:
```
   MONGO_USERNAME=your_mongodb_username

   MONGO_PASSWORD=your_encrypted_mongodb_password

   ENCRYPTION_KEY=your_encryption_key
```

   Replace your_mongodb_username, your_encrypted_mongodb_password, and your_encryption_key with the appropriate values.

---

### Running the Application
1. Start the FastAPI application:
   uvicorn main:app --reload

2. Access the API at http://127.0.0.1:8000

3. Interactive API documentation can be accessed at http://127.0.0.1:8000/docs

---

## API Endpoints

#### Face Recognition Endpoints

#### 1. Save Face Image
```http
POST /face/save_face/
```
**Description:** Captures an image from the webcam and saves it with the provided name.

**Parameters:**
- `name` *(str)*: The name associated with the face image.

---

#### 2. Recognize Faces from Image
```http
POST /face/recognize_faces/
```
**Description:** Captures an image from the webcam and recognizes faces in the image.

---

#### 3. Recognize Faces from Video
```http
POST /face/recognize_faces_video/
```
**Description:** Processes a video file and recognizes faces in the video.

**Parameters:**
- `file` *(UploadFile)*: The video file to process.

---

### Video Processing Endpoints

#### 1. Capture and Recognize Video
```http
POST /video/capture_and_recognize_video/
```
**Description:** Captures a video stream from the webcam, processes it, and recognizes faces in the video.

---

### Home Endpoint

#### 1. Home
```http
GET /
```
**Description:** Returns a simple greeting message.

