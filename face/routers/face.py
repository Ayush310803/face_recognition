from fastapi import APIRouter, HTTPException, UploadFile, File
from models.face_model import FaceImage
from utils.face_recognition_utils import save_face, recognize_faces, recognize_faces_video, save_face_from_upload, recognize_faces_from_upload
import os

router = APIRouter(prefix="/face", tags=["Face Recognition"])

@router.post("/save_face/")
async def save_face_endpoint(name: str):
    return await save_face(name)

@router.post("/save_face_upload/")
async def save_face_upload_endpoint(name: str, file: UploadFile = File(...)):
    return await save_face_from_upload(name, file)

@router.post("/recognize_faces/")
async def recognize_faces_endpoint():
    return await recognize_faces()

@router.post("/recognize_faces_upload/")
async def recognize_faces_upload_endpoint(file: UploadFile = File(...)):
    return await recognize_faces_from_upload(file)

@router.post("/recognize_faces_video/")
async def recognize_faces_video_endpoint(file: UploadFile = File(...)):
    return await recognize_faces_video(file)