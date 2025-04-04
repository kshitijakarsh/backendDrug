from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.utils.molecule_utils import generate_molecule

router = APIRouter()

@router.get("/generate/")
async def generate_drug_molecule(
    num_samples: int = Query(1, ge=1, le=10, description="Number of molecules to generate"),
    seed_smiles: Optional[str] = Query(None, description="Optional SMILES string to use as a template")
):
    """Generate novel drug-like molecules"""
    try:
        result = generate_molecule(num_samples=num_samples, seed_smiles=seed_smiles)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
