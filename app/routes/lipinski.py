from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from app.utils.molecule_utils import check_drug_likeness

router = APIRouter()

class LipinskiRequest(BaseModel):
    smiles: str

    @validator('smiles')
    def validate_smiles(cls, v):
        if not v or len(v) < 1:
            raise ValueError('SMILES string cannot be empty')
        return v

@router.post("/lipinski/")
async def check_lipinski_rules(request: LipinskiRequest):
    """Check if a molecule follows Lipinski's Rule of Five"""
    try:
        result = check_drug_likeness(request.smiles)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
