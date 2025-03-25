import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import datetime
from models.face_model import FaceImage
from fastapi import UploadFile, File
from fastapi import HTTPException

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

async def save_face(name: str):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open webcam")

    time.sleep(3)  
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Could not read frame from webcam")

    if not os.path.exists("static/images"):
        os.makedirs("static/images")

    image_filename = f"{name}{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    image_path = os.path.join("static/images", image_filename)
    cv2.imwrite(image_path, frame)

    try:
        face = FaceImage(name=name, image_path=image_path)
        face.save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving face to database: {e}")

    return {"message": "Face saved successfully"}

async def recognize_faces():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open webcam")

    time.sleep(3)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Could not read frame from webcam")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    captured_encodings = face_recognition.face_encodings(frame_rgb)
    if len(captured_encodings) == 0:
        return {"recognized_faces": ["No face detected"]}

    recognized_faces = []

    for captured_encoding in captured_encodings:
        best_match_name = "Unknown"
        best_similarity = 0.0

        for db_face in FaceImage.objects:
            if not os.path.exists(db_face.image_path):
                continue 

            stored_img = cv2.imread(db_face.image_path)
            if stored_img is None:
                continue  

            stored_img_rgb = cv2.cvtColor(stored_img, cv2.COLOR_BGR2RGB)

            stored_encodings = face_recognition.face_encodings(stored_img_rgb)
            if len(stored_encodings) == 0:
                continue  

            matches = face_recognition.compare_faces(stored_encodings, captured_encoding)
            similarity_scores = [cosine_similarity(stored_enc, captured_encoding) for stored_enc in stored_encodings]

            if True in matches:
                best_match_index = np.argmax(similarity_scores)  
                if similarity_scores[best_match_index] > best_similarity:
                    best_match_name = db_face.name
                    best_similarity = round(similarity_scores[best_match_index], 4)

        recognized_faces.append({"name": best_match_name, "similarity": best_similarity})

    return {"recognized_faces": recognized_faces}

async def capture_and_recognize_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open webcam")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    video_filename = f"video_{datetime.now().strftime('%Y%m%d%H%M%S')}.avi"
    original_video_path = os.path.join("static/video", video_filename)
    processed_video_path = os.path.join("static/video", f"processed_{video_filename}")

    os.makedirs("static/video", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 20.0
    out = cv2.VideoWriter(original_video_path, fourcc, fps, (frame_width, frame_height))
    processed_out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

    recognized_faces = []
    unknown_faces = []
    start_time = time.time()
    capture_duration = 10 

    while int(time.time() - start_time) < capture_duration:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(frame_rgb)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            best_match_name = None
            best_similarity = 0.0

            for db_face in FaceImage.objects:
                if not os.path.exists(db_face.image_path):
                    continue

                stored_img = cv2.imread(db_face.image_path)
                if stored_img is None:
                    continue

                stored_img_rgb = cv2.cvtColor(stored_img, cv2.COLOR_BGR2RGB)
                stored_encodings = face_recognition.face_encodings(stored_img_rgb)

                if len(stored_encodings) == 0:
                    continue

                matches = face_recognition.compare_faces(stored_encodings, face_encoding)
                similarity_scores = [cosine_similarity(stored_enc, face_encoding) for stored_enc in stored_encodings]

                if True in matches:
                    best_match_index = np.argmax(similarity_scores)
                    if similarity_scores[best_match_index] > best_similarity:
                        best_match_name = db_face.name
                        best_similarity = round(similarity_scores[best_match_index], 4)

            if best_match_name is None:
                match_found = False
                for idx, unknown_enc in enumerate(unknown_faces):
                    if face_recognition.compare_faces([unknown_enc], face_encoding, tolerance=0.6)[0]:
                        best_match_name = f"Unknown_{idx + 1}"
                        match_found = True
                        break
                
                if not match_found:
                    unknown_faces.append(face_encoding)
                    best_match_name = f"Unknown_{len(unknown_faces)}"

            recognized_faces.append({"name": best_match_name, "similarity": best_similarity})

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, best_match_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        processed_out.write(frame)

    cap.release()
    out.release()
    processed_out.release()

    return {
        "message": "Video captured and processed successfully",
        "original_video_path": original_video_path,
        "processed_video_path": processed_video_path,
        "recognized_faces": recognized_faces
    }

async def recognize_faces_video(file: UploadFile):
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as buffer:
        buffer.write(file.file.read())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video file")

    unknown_faces = [] 
    known_faces = {} 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        captured_encodings = face_recognition.face_encodings(frame_rgb)
        if len(captured_encodings) == 0:
            continue

        for captured_encoding in captured_encodings:
            best_match_name = None
            best_similarity = 0.0

            for db_face in FaceImage.objects:
                if not os.path.exists(db_face.image_path):
                    continue  

                stored_img = cv2.imread(db_face.image_path)
                if stored_img is None:
                    continue  

                stored_img_rgb = cv2.cvtColor(stored_img, cv2.COLOR_BGR2RGB)
                stored_encodings = face_recognition.face_encodings(stored_img_rgb)
                if len(stored_encodings) == 0:
                    continue  

                matches = face_recognition.compare_faces(stored_encodings, captured_encoding)
                similarity_scores = [np.dot(stored_enc, captured_encoding) / (np.linalg.norm(stored_enc) * np.linalg.norm(captured_encoding)) for stored_enc in stored_encodings]

                if True in matches:
                    best_match_index = np.argmax(similarity_scores)  
                    if similarity_scores[best_match_index] > best_similarity:
                        best_match_name = db_face.name
                        best_similarity = round(similarity_scores[best_match_index], 4)

            if best_match_name is None:
                match_found = False
                for idx, unknown_enc in enumerate(unknown_faces):
                    if face_recognition.compare_faces([unknown_enc], captured_encoding, tolerance=0.6)[0]:
                        best_match_name = f"Unknown_{idx + 1}"
                        match_found = True
                        break
                
                if not match_found:
                    unknown_faces.append(captured_encoding)
                    best_match_name = f"Unknown_{len(unknown_faces)}"

            if best_match_name not in known_faces or known_faces[best_match_name] < best_similarity:
                known_faces[best_match_name] = best_similarity

    cap.release()
    os.remove(video_path)  

    recognized_faces_list = [{"name": name, "similarity": similarity} for name, similarity in known_faces.items()]
    return {"recognized_faces": recognized_faces_list}

async def save_face_from_upload(name: str, file: UploadFile):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not os.path.exists("static/images"):
        os.makedirs("static/images")

    image_filename = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    image_path = os.path.join("static/images", image_filename)
    
    try:
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())
        
        img = cv2.imread(image_path)
        if img is None:
            os.remove(image_path)
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img_rgb)
        if len(face_encodings) == 0:
            os.remove(image_path)
            raise HTTPException(status_code=400, detail="No face detected in the image")
            
        face = FaceImage(name=name, image_path=image_path)
        face.save()
        
        return {"message": "Face saved successfully", "path": image_path}
        
    except Exception as e:
        if os.path.exists(image_path):
            os.remove(image_path)
        raise HTTPException(status_code=500, detail=f"Error saving face: {str(e)}")

async def recognize_faces_from_upload(file: UploadFile):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    temp_path = f"temp_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        frame = cv2.imread(temp_path)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not read the uploaded image file")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        captured_encodings = face_recognition.face_encodings(frame_rgb)
        if len(captured_encodings) == 0:
            return {"recognized_faces": ["No face detected"]}

        recognized_faces = []

        for captured_encoding in captured_encodings:
            best_match_name = "Unknown"
            best_similarity = 0.0

            for db_face in FaceImage.objects:
                if not os.path.exists(db_face.image_path):
                    continue 

                stored_img = cv2.imread(db_face.image_path)
                if stored_img is None:
                    continue  

                stored_img_rgb = cv2.cvtColor(stored_img, cv2.COLOR_BGR2RGB)
                stored_encodings = face_recognition.face_encodings(stored_img_rgb)
                if len(stored_encodings) == 0:
                    continue  

                matches = face_recognition.compare_faces(stored_encodings, captured_encoding)
                similarity_scores = [cosine_similarity(stored_enc, captured_encoding) 
                                  for stored_enc in stored_encodings]

                if True in matches:
                    best_match_index = np.argmax(similarity_scores)  
                    if similarity_scores[best_match_index] > best_similarity:
                        best_match_name = db_face.name
                        best_similarity = round(similarity_scores[best_match_index], 4)

            recognized_faces.append({"name": best_match_name, "similarity": best_similarity})

        return {"recognized_faces": recognized_faces}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)