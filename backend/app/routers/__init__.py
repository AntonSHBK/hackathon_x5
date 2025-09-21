from fastapi import APIRouter
from app.routers import ner_model

api_router = APIRouter()
api_router.include_router(ner_model.router)