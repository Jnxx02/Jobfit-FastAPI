from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Inisialisasi FastAPI
app = FastAPI()

# Muat model TF-IDF dan data pre-processing yang sudah disimpan
tfidf_model = joblib.load('models/tfidf_model.pkl')
tfidf_matrix = joblib.load('models/tfidf_matrix.pkl')
df_sorted = joblib.load('models/df_sorted.pkl')

# Data yang dikirimkan oleh user
class UserInput(BaseModel):
    skills: str
    experience: int

# Endpoint untuk job matching
@app.post("/match_job")
def match_job(user_input: UserInput):
    user_skills = user_input.skills.lower()
    user_experience = user_input.experience

    # Transform skill user menggunakan TF-IDF model
    user_skill_tfidf = tfidf_model.transform([user_skills])

    # Hitung similarity antara user skill dan deskripsi pekerjaan
    skill_similarity = cosine_similarity(user_skill_tfidf, tfidf_matrix)

    # Tambahkan kolom skill similarity dan hitung difference experience
    df_sorted['Skill_Similarity'] = skill_similarity.flatten()
    df_sorted['Experience_Diff'] = abs(df_sorted['Job Experience'] - user_experience)

    # Sort berdasarkan skill similarity dan experience difference
    top_jobs = df_sorted.sort_values(by=['Skill_Similarity', 'Experience_Diff'], ascending=[False, True])

    # Ambil hasil top 5 pekerjaan yang paling cocok
    recommended_jobs = top_jobs[['Job_Role', 'Company', 'Location', 'Skill_Similarity', 'Experience_Diff']].head(5)

    if recommended_jobs.empty:
        raise HTTPException(status_code=404, detail="No matching jobs found")

    # Kembalikan hasil rekomendasi
    return recommended_jobs.to_dict(orient="records")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Jobfit API!"}

@app.get("/model_info")
async def get_model_info():
    # Contoh informasi yang bisa dikembalikan
    return {
        "model": "TF-IDF",
        "status": "loaded",
        "num_features": len(tfidf_model.get_feature_names_out())
    }

# Jalankan server menggunakan Uvicorn (asynchronous server)
# Buka terminal dan jalankan:
# uvicorn main:app --reload
