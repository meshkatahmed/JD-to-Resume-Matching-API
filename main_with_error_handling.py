import shutil
from pathlib import Path
from fastapi import FastAPI, status, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from services.matching_service import get_available_cvs, compute_similarity_scores

# Initialize the FastAPI application
app = FastAPI(
    title="Recruitment Automation System AI API",
    description="A production-ready FastAPI Endpoint",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JD_FOLDER = Path("job_descriptions")
CVS_FOLDER = Path("cvs")

# Create folders if they don't exist
JD_FOLDER.mkdir(exist_ok=True)
CVS_FOLDER.mkdir(exist_ok=True)

@app.get("/")
def read_root():
    return FileResponse('index.html')

@app.post("/upload_jd")
async def upload_jd(file: UploadFile = File(...)):
    # Save to job_descriptions folder
    file_path = JD_FOLDER / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": "JD saved"}

@app.post("/upload_cvs")
async def upload_cvs(files: list[UploadFile] = File(...)):
    for file in files:
        file_path = CVS_FOLDER / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    return {"message": f"Saved {len(files)} CVs"}

@app.get("/match_cv/{job_ref_id}")
def match_cv(job_ref_id: str):
    print("API Called",job_ref_id)
    jd_path = Path(f"job_descriptions/{job_ref_id}.docx")
    print(jd_path)
    if not jd_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Job description with reference ID '{job_ref_id}' not found."
        )

    cvs = get_available_cvs()
    if not cvs:
        return {}

    scores = compute_similarity_scores(jd_path, cvs)
    scores.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity score in descending order
    
    results = {}
    for i, (cv_path, similarity) in enumerate(scores):
        similarity_percentage = f"{round(similarity * 100, 0)}%"
        results[str(i)] = {
            "name": cv_path.name,
            "similarity_score": similarity_percentage
        }

    return results

@app.get("/cvs/{filename}")
async def get_cv(filename: str):
    file_path = Path(f"cvs/{filename}")
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(404, "CV not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
