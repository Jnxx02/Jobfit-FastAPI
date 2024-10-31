from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

# Data yang dikirimkan oleh user
class UserInput(BaseModel):
    skills: str
    experience: int

# Inisialisasi FastAPI
app = FastAPI()

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua asal
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def homepage():
    return "Welcome to the Jobfit API!"


