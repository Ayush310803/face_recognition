from fastapi import FastAPI
from routers import face, video
from database import connect_to_db
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

connect_to_db()

app.include_router(face.router)
app.include_router(video.router)

@app.get("/")
async def home():
    return {"message": "Hello, World!"}