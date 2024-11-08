from fastapi import FastAPI, HTTPException, Request, Form, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Data yang dikirimkan oleh user
class JobMatchInput(BaseModel):
    skills: str
    experience: int
    company: str
    job_role: str

class TopMatchesInput(BaseModel):
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
async def match_job(
    request: Request,
    skills: str = Form(...),
    experience: int = Form(...),
    company: str = Form(...),
    job_role: str = Form(...)
):
    # Proses input dan hitung kecocokan
    user_input = f"{skills} {experience} years"
    user_vector = tfidf_model.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Filter berdasarkan company dan job role
    filtered_jobs = df_sorted[
        (df_sorted['Company'].str.contains(company, case=False)) &
        (df_sorted['Job_Role'].str.contains(job_role, case=False))
    ]

    if filtered_jobs.empty:
        raise HTTPException(status_code=404, detail="No matching jobs found for the specified company and job role")

    # Calculate match percentage for filtered jobs
    valid_indices = filtered_jobs.index.intersection(range(len(similarity_scores)))
    filtered_jobs = filtered_jobs.loc[valid_indices]
    filtered_jobs['Match_Percentage'] = similarity_scores[valid_indices] * 100

    if filtered_jobs.empty or filtered_jobs['Match_Percentage'].isnull().all():
        raise HTTPException(status_code=404, detail="No valid matches found after filtering")

    # Return all matches
    return templates.TemplateResponse("match_result.html", {
        "request": request,
        "top_matches": filtered_jobs.to_dict(orient="records")
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
def top_matches(user_input: TopMatchesInput):
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

@app.post("/match_job/json", response_class=JSONResponse)
async def match_job_json(input_data: JobMatchInput):
    # Proses input dan hitung kecocokan
    user_input = f"{input_data.skills} {input_data.experience} years"
    user_vector = tfidf_model.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Filter berdasarkan company dan job role
    filtered_jobs = df_sorted[
        (df_sorted['Company'].str.contains(input_data.company, case=False)) &
        (df_sorted['Job_Role'].str.contains(input_data.job_role, case=False))
    ]

    if filtered_jobs.empty:
        raise HTTPException(status_code=404, detail="No matching jobs found for the specified company and job role")

    # Calculate match percentage for filtered jobs
    valid_indices = filtered_jobs.index.intersection(range(len(similarity_scores)))
    filtered_jobs = filtered_jobs.loc[valid_indices]
    filtered_jobs['Match_Percentage'] = similarity_scores[valid_indices] * 100

    if filtered_jobs.empty or filtered_jobs['Match_Percentage'].isnull().all():
        raise HTTPException(status_code=404, detail="No valid matches found after filtering")

    # Return all matches as JSON
    return JSONResponse(content=filtered_jobs.to_dict(orient="records"))

