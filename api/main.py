from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Data yang dikirimkan oleh user
class UserInput(BaseModel):
    skills: str
    experience: int

# Inisialisasi FastAPI dan Jinja2
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Ambil daftar skill dari model atau data yang relevan
    skills_list = ["Python", "Java", "Data Analysis", "Machine Learning"]  # Contoh daftar skill
    return templates.TemplateResponse("index.html", {"request": request, "skills": skills_list})

# Muat model TF-IDF dan data pre-processing yang sudah disimpan
tfidf_model = joblib.load('models/tfidf_model.pkl')
tfidf_matrix = joblib.load('models/tfidf_matrix.pkl')
df_sorted = joblib.load('models/df_sorted.pkl')

# Endpoint untuk job matching
@app.post("/match_job", response_class=HTMLResponse)
async def match_job(request: Request, skills: str = Form(...), experience: int = Form(...)):
    # Proses input dan hitung kecocokan
    user_input = f"{skills} {experience} years"
    user_vector = tfidf_model.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Dapatkan 5 kecocokan teratas
    top_indices = np.argsort(similarity_scores)[-5:][::-1]
    top_matches = df_sorted.iloc[top_indices]
    top_matches['Match_Percentage'] = similarity_scores[top_indices] * 100

    # Ambil kecocokan terbaik untuk pie chart
    best_match = top_matches.iloc[0]

    return templates.TemplateResponse("match_result.html", {
        "request": request,
        "company": best_match['Company'],
        "job_role": best_match['Job_Role'],
        "match_percentage": best_match['Match_Percentage'],
        "top_matches": top_matches.to_dict(orient="records")
    })

@app.get("/companies_jobs", response_class=HTMLResponse)
async def get_companies_jobs(request: Request):
    companies_jobs = df_sorted[['Company', 'Job_Role']].drop_duplicates()
    return templates.TemplateResponse("companies_jobs.html", {"request": request, "companies_jobs": companies_jobs.to_dict(orient="records")})

@app.get("/companies_jobs/json")
async def get_companies_jobs_json():
    companies_jobs = df_sorted[['Company', 'Job_Role']].drop_duplicates()
    return JSONResponse(content=companies_jobs.to_dict(orient="records"))

@app.post("/top_matches")
def top_matches(user_input: UserInput):
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

