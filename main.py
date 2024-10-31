from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

# Muat model yang sudah dilatih
model = joblib.load('models/tfidf_model.pkl')

class InputData(BaseModel):
    user_skill: str

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Transform input data
        user_skill_tfidf = model.transform([data.user_skill])
        # Lakukan prediksi atau operasi lain
        # Misalnya, hitung kesamaan atau prediksi lainnya
        # result = some_function(user_skill_tfidf)
        result = {"message": "Prediction result here"}  # Placeholder
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Jobfit API!"}

@app.get("/model_info")
async def get_model_info():
    # Contoh informasi yang bisa dikembalikan
    return {
        "model": "TF-IDF",
        "status": "loaded",
        "num_features": len(model.get_feature_names_out())
    }

# Jalankan server menggunakan Uvicorn (asynchronous server)
# Buka terminal dan jalankan:
# uvicorn main:app --reload
