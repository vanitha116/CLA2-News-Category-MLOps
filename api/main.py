from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.model_loader import ModelLoader
from api.monitoring_logger import log_prediction_event
from api.schemas import (
    BatchNewsInput, BatchPredictionResponse, NewsArticleInput,
    SinglePredictionResponse, StandardErrorResponse
)

app = FastAPI(
    title="News Category Classification MLOps API",
    description="Production-grade inference and serving platform for News Category Classification (Lab 10)",
    version="1.0.0"
)

# Global model loader instance
loader = ModelLoader()


# Standardized 422 Validation Error Handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = []
    for error in exc.errors():
        loc = " -> ".join([str(x) for x in error.get("loc", [])])
        msg = error.get("msg", "Invalid value")
        details.append({"field": loc, "message": msg})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Request validation failed. Ensure all fields adhere to constraints.",
            "details": details
        }
    )


# Standardized 500 Internal Exception Handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": f"Internal server error encountered: {str(exc)}"
        }
    )


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "News Category Classification MLOps API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs"
    }


@app.get("/health", tags=["Health"])
def health():
    if not loader.is_ready():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "message": "Model or preprocessor is not loaded and ready for inference."
            }
        )

    return {
        "status": "healthy",
        "model_loaded": loader.model is not None,
        "preprocessor_loaded": loader.preprocessor is not None,
        "model_version": loader.model_version
    }


@app.get("/metadata", tags=["Metadata"])
def metadata():
    if not loader.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready to serve metadata."
        )

    return {
        "model_name": "News_Category_Production_Model",
        "version": loader.model_version,
        "metadata": loader.metadata
    }


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Inference"])
def predict_single(article: NewsArticleInput):
    if not loader.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized and ready for inference."
        )

    try:
        result = loader.predict_one(article.headline, article.short_description)
        log_prediction_event(
            headline=result["headline"],
            predicted_category=result["predicted_category"],
            latency_ms=result["latency_ms"],
            model_version=loader.model_version
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch_input: BatchNewsInput):
    if not loader.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized and ready for inference."
        )

    try:
        results = loader.predict_batch(batch_input.articles)
        for r in results:
            log_prediction_event(
                headline=r["headline"],
                predicted_category=r["predicted_category"],
                latency_ms=r["latency_ms"],
                model_version=loader.model_version
            )
        return {
            "status": "success",
            "total_processed": len(results),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
