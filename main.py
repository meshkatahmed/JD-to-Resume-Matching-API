import string
from typing import Set, Union
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from text_extraction_engine import TextExtractionEngine
from text_processing_engine import TextProcessingEngine

# Initialize the FastAPI application
app = FastAPI(
    title="JD-to-Resume-Matching API",
    description="A production-ready FastAPI Endpoint",
    version="1.0.0"
)

# Define a Pydantic schema for request body validation
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

def build_token_set(tokens: list[str]) -> Set[str]:
    """Step 3: Reduce a token list down to a set of unique words."""
    return set(tokens)

# 1. GET Root Endpoint
@app.get("/", status_code=status.HTTP_200_OK)
def read_root():
    return {"message": "Welcome to JD-to-Resume Matching API!"}

# 2. GET Endpoint with Path and Query Parameters
@app.get("/match_cv/{job_ref_id}", status_code=status.HTTP_200_OK)
def match_cv(job_reference_id: str):
    cvs = ["cvs/cv_1.pdf","cvs/cv_2.pdf"]

    # Step 1: Extraction
    extraction_engine = TextExtractionEngine()

    raw_jd = extraction_engine.extract("job_descriptions/{job_reference_id}.txt")
    cvs = ["cvs/cv_1.pdf", "cvs/cv_2.pdf"]
    raw_cvs = [extraction_engine.extract(cv) for cv in cvs]

    # Step 2: Processing
    processing_engine = TextProcessingEngine()
    jd_tokens = processing_engine.process(raw_jd)
    cv_tokens_list = [processing_engine.process(raw_cv) for raw_cv in raw_cvs]

    # Step 3: Token Set Creation
    jd_token_set = build_token_set(jd_tokens)
    cv_tokens_set = [build_token_set(cv_tokens) for cv_tokens in cv_tokens_list]

    # Step 4: Master Vector Creation
    master_token_sets = [jd_token_set.union(cv_tokens) for cv_tokens in cv_tokens_set]

    # Step 5: 
    



    return {"job_ref_id": job_reference_id}






















# 3. POST Endpoint with Request Body Validation
@app.post("/items/", status_code=status.HTTP_201_CREATED)
def create_item(item: Item):
    if item.price <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Price must be greater than zero."
        )
    
    # Calculate total price if tax exists
    total_price = item.price + (item.tax if item.tax else 0)
    
    return {
        "message": "Item successfully created",
        "data": item,
        "total_price": total_price
    }
