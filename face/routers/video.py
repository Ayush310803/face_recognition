from fastapi import APIRouter, HTTPException
from utils.face_recognition_utils import capture_and_recognize_video

router = APIRouter(prefix="/video", tags=["Video Processing"])

@router.post("/capture_and_recognize_video/")
async def capture_and_recognize_video_endpoint(is_live: str = "close"):
    return await capture_and_recognize_video(is_live)