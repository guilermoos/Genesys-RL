"""
Inference routes.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_user
from app.schemas.inference import InferenceRequest, InferenceResponse, InferenceBatchRequest, InferenceBatchResponse
from app.services.inference_service import InferenceService
from app.models.user import User

router = APIRouter()


@router.post("/projects/{project_id}/predict", response_model=InferenceResponse)
def predict(
    project_id: str,
    request: InferenceRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Run inference on a state.
    
    Args:
        project_id: Project ID
        request: Inference request with state
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Inference result with action
    """
    try:
        result = InferenceService.predict(
            db,
            project_id,
            current_user.id,
            state=request.state,
            model_version_id=request.model_version_id,
            metadata=request.metadata,
        )
        
        return InferenceResponse(
            action=result["action"],
            confidence=result["confidence"],
            model_version_id=result["model_version_id"],
            model_version=result["model_version"],
            inference_time_ms=result["inference_time_ms"],
            timestamp=result["timestamp"],
            metadata={"q_values": result.get("q_values")},
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


@router.post("/projects/{project_id}/predict/batch", response_model=InferenceBatchResponse)
def predict_batch(
    project_id: str,
    request: InferenceBatchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Run batch inference on multiple states.
    
    Args:
        project_id: Project ID
        request: Batch inference request with states
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Batch inference results
    """
    try:
        result = InferenceService.predict_batch(
            db,
            project_id,
            current_user.id,
            states=request.states,
            model_version_id=request.model_version_id,
        )
        
        return result
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}",
        )


@router.get("/projects/{project_id}/stats")
def get_inference_stats(
    project_id: str,
    hours: int = 24,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get inference statistics for a project.
    
    Args:
        project_id: Project ID
        hours: Time window in hours
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Inference statistics
    """
    stats = InferenceService.get_inference_stats(
        db,
        project_id,
        current_user.id,
        hours=hours,
    )
    
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    
    return stats
