from pathlib import Path
from fastapi import FastAPI, status, Path as FastAPIPath, HTTPException
from services.matching_service import get_available_cvs, compute_similarity_scores

# Initialize the FastAPI application
app = FastAPI(
    title="Recruitment Automation System AI API",
    description="A production-ready FastAPI Endpoint",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Recruitment Automation System AI API!"}


@app.get("/match_cv/{job_ref_id}")
def match_cv(job_ref_id: str):
    jd_path = Path(f"job_descriptions/{job_ref_id}.txt")
    if not jd_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Job description with reference ID '{job_ref_id}' not found."
        )

    cvs = get_available_cvs()
    if not cvs:
        return {}

    scores = compute_similarity_scores(jd_path, cvs)
    
    results = {}
    for i, (cv_path, similarity) in enumerate(scores):
        similarity_percentage = f"{round(similarity * 100, 1)}%"
        results[str(i)] = {
            "name": cv_path.name,
            "similarity_score": similarity_percentage
        }

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
