from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..utils.molecule_utils import check_drug_likeness, predict_admet
from DeepPurpose import DTI as models
from DeepPurpose.utils import data_process_repurpose_virtual_screening

router = APIRouter()

class AgentRequest(BaseModel):
    drug: str
    target: str
    model_type: str = "CNN"

@router.post("/agent/")
async def full_analysis(request: AgentRequest):
    """Perform complete drug analysis including binding, Lipinski rules, and ADMET"""
    try:
        # Check drug-likeness
        drug_likeness = check_drug_likeness(request.drug)
        if "error" in drug_likeness:
            raise HTTPException(status_code=400, detail=drug_likeness["error"])

        # Predict binding
        model = models.model_pretrained(model='CNN_CNN_DAVIS')
        processed_data = data_process_repurpose_virtual_screening(
            request.drug,
            request.target,
            drug_encoding=model.drug_encoding,
            target_encoding=model.target_encoding,
            mode='repurposing'
        )
        predictions = model.predict(processed_data)

        # Get ADMET properties
        admet_results = predict_admet(request.drug)

        return {
            "drug_smiles": request.drug,
            "target_sequence": request.target,
            "drug_likeness": drug_likeness,
            "binding_score": float(predictions[0]),
            "admet": admet_results,
            "message": "Higher scores indicate stronger predicted binding"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
