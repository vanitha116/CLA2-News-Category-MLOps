from typing import List, Optional
from pydantic import BaseModel, Field


class NewsArticleInput(BaseModel):
    headline: str = Field(..., min_length=3, description="Headline of the news article")
    short_description: Optional[str] = Field("", description="Optional short summary of the article")
    authors: Optional[str] = Field("", description="Author(s) credited for the article")
    date: Optional[str] = Field("", description="Publication date (YYYY-MM-DD)")
    link: Optional[str] = Field("", description="Original article link")

    class Config:
        json_schema_extra = {
            "example": {
                "headline": "James Webb Space Telescope Captures Deep Space Cosmic Dawn",
                "short_description": "Astronomers discover primordial galaxies formed right after the Big Bang.",
                "authors": "Jane Doe, Science Staff",
                "date": "2026-10-15",
                "link": "https://www.huffpost.com/entry/science-jwst-dawn"
            }
        }


class BatchNewsInput(BaseModel):
    articles: List[NewsArticleInput] = Field(..., min_items=1, max_items=1000, description="List of news articles to classify")


class PredictionResult(BaseModel):
    headline: str
    predicted_category: str
    confidence: Optional[float] = None
    model_version: Optional[str] = "1"
    latency_ms: float


class SinglePredictionResponse(BaseModel):
    status: str = "success"
    data: PredictionResult


class BatchPredictionResponse(BaseModel):
    status: str = "success"
    total_processed: int
    predictions: List[PredictionResult]


class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str


class StandardErrorResponse(BaseModel):
    status: str = "error"
    message: str
    details: Optional[List[ErrorDetail]] = None
