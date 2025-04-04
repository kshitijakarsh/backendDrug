from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from app.utils.molecule_utils import predict_admet

router = APIRouter()

class AdmetRequest(BaseModel):
    smiles: str

    @validator('smiles')
    def validate_smiles(cls, v):
        if not v or len(v) < 1:
            raise ValueError('SMILES string cannot be empty')
        return v

@router.post("/admet/")
async def predict_admet_properties(request: AdmetRequest):
    """Predict ADMET properties of a molecule"""
    try:
        result = predict_admet(request.smiles)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
