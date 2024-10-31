from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def homepage():
    return "Welcome to the Jobfit API!"


