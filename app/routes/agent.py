from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, root_validator
from typing import Optional
from ..utils.molecule_utils import check_drug_likeness, predict_admet
from DeepPurpose import DTI as models
from DeepPurpose.utils import data_process_repurpose_virtual_screening

router = APIRouter()

class AgentRequest(BaseModel):
    smiles: str = Field(None, description="SMILES string of the drug molecule")
    drug: Optional[str] = None
    target: str
    model_type: str = "CNN"
    question: Optional[str] = None

    @root_validator(pre=True)
    def check_smiles_or_drug(cls, values):
        """Ensure either smiles or drug is provided and set smiles if drug is provided"""
        smiles = values.get('smiles')
        drug = values.get('drug')
        
        if drug and not smiles:
            values['smiles'] = drug
        elif not drug and not smiles:
            raise ValueError("Either 'smiles' or 'drug' must be provided")
        
        return values

    class Config:
        schema_extra = {
            "example": {
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "target": "MRGPGAGVLVVGVGVGVGVGVGVGV",
                "model_type": "CNN",
                "question": "Analyze this drug's properties"
            }
        }

@router.post("/agent/")
async def full_analysis(request: AgentRequest):
    try:
        # Use smiles from either source
        drug_smiles = request.smiles
        
        # Step 1: Check drug-likeness
        drug_likeness = check_drug_likeness(drug_smiles)
        if "error" in drug_likeness:
            raise HTTPException(status_code=400, detail=drug_likeness["error"])

        # Step 2: Predict binding
        model = models.model_pretrained(model='CNN_CNN_DAVIS')
        processed_data = data_process_repurpose_virtual_screening(
            drug_smiles,
            request.target,
            drug_encoding=model.drug_encoding,
            target_encoding=model.target_encoding,
            mode='repurposing'
        )
        predictions = model.predict(processed_data)
        binding_score = float(predictions[0])

        # Step 3: Get ADMET properties
        admet_results = predict_admet(drug_smiles)

        # Prepare base response
        response_data = {
            "drug_smiles": drug_smiles,
            "target_sequence": request.target,
            "drug_likeness": drug_likeness,
            "binding_score": binding_score,
            "admet": admet_results,
            "message": "Higher scores indicate stronger predicted binding"
        }

        # Step 4: If question is provided, get AI analysis
        if request.question:
            try:
                from .agent_ai import ask_agent_ai, AgentAIRequest
                
                ai_request = AgentAIRequest(
                    instructions=request.question,
                    drug_data={
                        "drug_smiles": drug_smiles,
                        "target_sequence": request.target,
                        "drug_likeness": drug_likeness,
                        "binding_score": binding_score,
                        "admet": admet_results
                    },
                    llm_engine="gpt4o"
                )

                ai_response = await ask_agent_ai(ai_request)
                if ai_response and "analysis" in ai_response:
                    response_data["ai_analysis"] = ai_response["analysis"]
                else:
                    response_data["ai_analysis"] = {
                        "error": "No analysis received from AI service"
                    }
            except Exception as e:
                response_data["ai_analysis"] = {
                    "error": f"AI analysis failed: {str(e)}"
                }

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
