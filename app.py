from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from typing import List
from api.hackrx_pipeline import process_questions
from api.auth import validate_api_key

app = FastAPI()

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=AnswerResponse)
def run_hackrx(
    request: QuestionRequest,
    fastapi_request: Request,
    _: None = Depends(validate_api_key)
):
    # You can use fastapi_request here if needed
    answers = process_questions(request.documents, request.questions)
    return {"answers": answers}