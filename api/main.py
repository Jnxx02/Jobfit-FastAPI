from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

# Inisialisasi FastAPI dan Jinja2
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model dan data
tfidf_model = joblib.load("models/tfidf_model.pkl")
tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")
df_sorted = joblib.load("models/df_sorted.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Ambil daftar skill dari model atau data yang relevan
    skills_list = ["Python", "Java", "Data Analysis", "Machine Learning"]  # Contoh daftar skill
    return templates.TemplateResponse("index.html", {"request": request, "skills": skills_list})

@app.post("/match_job", response_class=HTMLResponse)
async def match_job(request: Request, skills: str = Form(...), experience: int = Form(...)):
    # Proses input pengguna dan hitung kecocokan
    user_input = f"{skills} {experience} years"
    user_vector = tfidf_model.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # Ambil hasil kecocokan tertinggi
    top_match_index = np.argmax(similarity_scores)
    top_match = df_sorted.iloc[top_match_index]
    match_percentage = similarity_scores[top_match_index] * 100

    return templates.TemplateResponse("match_result.html", {
        "request": request,
        "company": top_match['Company'],
        "job_role": top_match['Job_Role'],
        "match_percentage": match_percentage
    })

@app.get("/companies_jobs", response_class=HTMLResponse)
async def get_companies_jobs(request: Request):
    companies_jobs = df_sorted[['Company', 'Job_Role']].drop_duplicates()
    return templates.TemplateResponse("companies_jobs.html", {"request": request, "companies_jobs": companies_jobs.to_dict(orient="records")})
