from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", summary="Check API health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
