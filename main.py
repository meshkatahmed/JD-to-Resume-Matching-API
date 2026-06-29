import re
import shutil
import math
from pathlib import Path

from fastapi import FastAPI, status, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from services.matching_service import get_available_cvs, compute_similarity_scores, compute_similarity_scores_tfidf

# Initialize the FastAPI application
app = FastAPI(
    title="Recruitment Automation System AI API",
    description="A production-ready FastAPI Endpoint",
    version="1.0.0"
)

# CORS configuration to allow requests from any origin
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
def read_ui():
    return FileResponse('uis/index_2.html')

@app.post("/upload_jd/{job_ref_id}")
async def upload_jd(file: UploadFile = File(...), job_ref_id: str = None):
    
    # Save to job_ref_id folder
    JD_JOB_REF_FOLDER = JD_FOLDER / job_ref_id
    JD_JOB_REF_FOLDER.mkdir(exist_ok=True)
    
    file_path = JD_JOB_REF_FOLDER / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": "JD saved"}

@app.post("/upload_cvs/{job_ref_id}")
async def upload_cvs(files: list[UploadFile] = File(...),job_ref_id: str = None):
    
    # Save to job_ref_id folder
    CVS_JOB_REF_FOLDER = CVS_FOLDER / job_ref_id
    CVS_JOB_REF_FOLDER.mkdir(exist_ok=True)

    for file in files:
        file_path = CVS_JOB_REF_FOLDER / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    return {"message": f"Saved {len(files)} CVs"}

@app.get("/match_cv/{job_ref_id}")
def match_cv(job_ref_id: str, vectorization_method: str = "cbow"):
    """
    Match CVs against a job description using the specified vectorization technique.

    Parameters
    ----------
    job_ref_id : str
        Filename of the job description (e.g. ``senior_dev.txt``).
    vectorization_method : str, optional
        ``"cbow"`` (default) — raw term-count vectors (Continuous Bag of Words).
        ``"tfidf"`` — TF-IDF weighted vectors; rare, discriminative terms get
        higher weight across the full document corpus.
    """
    job_ref_id_without_ext = re.sub(r'\.[^/.]+$', '', job_ref_id)
    jd_path = JD_FOLDER / job_ref_id_without_ext / job_ref_id
    
    if not jd_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Job description with reference ID '{job_ref_id}' not found."
        )

    cv_paths = get_available_cvs(CVS_FOLDER / job_ref_id_without_ext)
    
    if not cv_paths:
        return {}

    if vectorization_method == "tfidf":
        scores = compute_similarity_scores_tfidf(jd_path, cv_paths)
    else:
        # Default: CBOW (raw term-count vectors)
        scores = compute_similarity_scores(jd_path, cv_paths)

    scores.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity score in descending order
    
    results = {}
    for i, (cv_path, similarity) in enumerate(scores):
        similarity_percentage = f"{math.ceil(similarity * 100)} %"
        results[str(i)] = {
            "name": cv_path.name,
            "similarity_score": similarity_percentage
        }

    return results

@app.get("/cvs/{job_ref_id}/{filename}")
async def get_cv(job_ref_id: str, filename: str):
    file_path = Path(CVS_FOLDER / job_ref_id / filename)
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(404, "CV not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
