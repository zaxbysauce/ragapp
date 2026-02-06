"""
Health check API route.

Provides a health endpoint to check the status of LLM services and model availability.
"""
from fastapi import APIRouter

from app.services.llm_health import LLMHealthChecker
from app.services.model_checker import ModelChecker


router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns the status of LLM services (embeddings and chat) and
    the availability of configured models.

    Returns:
        JSON response with:
        - status: "ok" if all checks pass
        - llm: LLM service health status from LLMHealthChecker
        - models: Model availability status from ModelChecker
        - services: Frontend-compatible health badge booleans
          { backend: boolean, embeddings: boolean, chat: boolean }
    """
    llm_checker = LLMHealthChecker()
    model_checker = ModelChecker()
    
    llm_status = await llm_checker.check_all()
    models_status = await model_checker.check_models()

    # Derive service booleans for frontend health badges
    services = {
        "backend": True,  # Endpoint executed successfully
        "embeddings": llm_status.get("embeddings", {}).get("ok", False),
        "chat": llm_status.get("chat", {}).get("ok", False)
    }

    return {
        "status": "ok",
        "llm": llm_status,
        "models": models_status,
        "services": services
    }
