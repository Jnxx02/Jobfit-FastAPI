from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Inisialisasi FastAPI
app = FastAPI()

@app.get("/")
async def welcome():
    return "Welcome to the Jobfit API!"

# Muat model TF-IDF dan data pre-processing yang sudah disimpan
tfidf_model = joblib.load('models/tfidf_model.pkl')
tfidf_matrix = joblib.load('models/tfidf_matrix.pkl')
df_sorted = joblib.load('models/df_sorted.pkl')

# Data yang dikirimkan oleh user
class UserInput(BaseModel):
    skills: str
    experience: int

# Endpoint untuk job matching

