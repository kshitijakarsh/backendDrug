from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from DeepPurpose import DTI as models
from DeepPurpose.utils import data_process_repurpose_virtual_screening

router = APIRouter()

class BindingRequest(BaseModel):
    drug: str
    target: str
    model_type: str = "CNN"

    @validator('drug')
    def validate_smiles(cls, v):
        if not v or len(v) < 1:
            raise ValueError('SMILES string cannot be empty')
        return v

    @validator('target')
    def validate_sequence(cls, v):
        if not v or len(v) < 1:
            raise ValueError('Protein sequence cannot be empty')
        return v

@router.post("/binding/")
async def predict_binding(request: BindingRequest):
    """Predict drug-target binding affinity"""
    try:
        # Load pretrained model
        model = models.model_pretrained(model='CNN_CNN_DAVIS')

        # Process data
        processed_data = data_process_repurpose_virtual_screening(
            request.drug,
            request.target,
            drug_encoding=model.drug_encoding,
            target_encoding=model.target_encoding,
            mode='repurposing'
        )

        # Make prediction
        predictions = model.predict(processed_data)
        
        return {
            "drug_smiles": request.drug,
            "target_sequence": request.target,
            "binding_score": float(predictions[0]),
            "message": "Higher scores indicate stronger predicted binding"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
