from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import shutil
from lanchain_chotbot import RAGServiceModern

app = FastAPI(
    title="RAG ChatBot API",
    description="Modern API for PDF-based Question Answering using RAG with LCEL"
)


rag_service = RAGServiceModern(vector_db_path="vector_db")

UPLOAD_DIR = "uploaded_pdfs"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

current_pdf_path: Optional[str] = None


class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the PDF", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the main topic of this document?"
            }
        }


class QuestionResponse(BaseModel):
    status: str
    question: str
    answer: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "question": "What is the main topic of this document?",
                "answer": "The document discusses..."
            }
        }

class MessageResponse(BaseModel):
    status: str
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Operation completed successfully"
            }
        }

class ChatMessage(BaseModel):
    type: str
    content: str


class ChatHistoryResponse(BaseModel):
    status: str
    chat_history: List[ChatMessage]
    total_messages: int


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str


@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "RAG ChatBot API - Modern Version with LCEL",
        "version": "2.0.0",
        "documentation": "/docs",
        "endpoints": {
            "health": "GET /health",
            "upload_pdf": "POST /upload-pdf",
            "ask_question": "POST /ask",
            "chat_history": "GET /chat-history"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        service="RAG ChatBot API"
    )


@app.post("/upload-pdf", response_model=MessageResponse)
async def upload_pdf(file: UploadFile = File(..., description="PDF file to upload and process")):

    global current_pdf_path
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are allowed. Please upload a .pdf file."
        )

    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        current_pdf_path = file_path
        
        result = rag_service.process_pdf(file_path)
        
        if result["status"] == "error":
            if os.path.exists(file_path):
                os.remove(file_path)
            current_pdf_path = None
            raise HTTPException(status_code=500, detail=result["message"])
        
        return MessageResponse(
            status="success",
            message=f"✅ PDF '{file.filename}' uploaded and processed successfully. {result['message']}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        if current_pdf_path and os.path.exists(current_pdf_path):
            os.remove(current_pdf_path)
        current_pdf_path = None
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")
    finally:
        await file.close()


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):

    if not request.question or request.question.strip() == "":
        raise HTTPException(
            status_code=400, 
            detail="Question cannot be empty"
        )
    
    result = rag_service.ask_question(request.question)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return QuestionResponse(
        status=result["status"],
        question=result["question"],
        answer=result["answer"]
    )


@app.get("/chat-history", response_model=ChatHistoryResponse)
async def get_chat_history():


    try:
        history = rag_service.get_chat_history()
        
        chat_messages = [
            ChatMessage(type=msg["type"], content=msg["content"])
            for msg in history
        ]
        
        return ChatHistoryResponse(
            status="success",
            chat_history=chat_messages,
            total_messages=len(chat_messages)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving chat history: {str(e)}"
        )




