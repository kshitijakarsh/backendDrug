from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
from typing import Optional, Dict, Any

router = APIRouter()

class AgentAIRequest(BaseModel):
    instructions: str  # Changed from query to instructions to match API
    drug_data: Dict[str, Any]
    llm_engine: str = "gpt4o"  # Default to GPT-4

    class Config:
        schema_extra = {
            "example": {
                "instructions": "Analyze this drug's properties and suggest improvements ",
                "drug_data": {
                    "drug_smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC(=O)C4=CN=CC=C4",
                    "target_sequence": "MRGPGAGVLVVGVGVGVGVGVGVGV",
                    "drug_likeness": {
                        "molecular_weight": 443.551,
                        "logP": 3.642,
                        "HBD": 2,
                        "HBA": 5,
                        "drug_likeness": "Pass"
                    },
                    "binding_score": 5.042839050292969,
                    "admet": {
                        "absorption": {"intestinal_absorption": "High"},
                        "metabolism": {"risk_level": "Low"},
                        "toxicity": {"risk_level": "Low"}
                    }
                },
                "llm_engine": "gpt4o"
            }
        }

@router.post("/agentai/")
async def ask_agent_ai(request: AgentAIRequest):
    """
    Query the AgentAI with drug analysis data for insights and recommendations.
    """
    try:
        # API configuration
        API_KEY = "WqapEOrxL4wmfpCLC5BrR7auoBQWrekqKTLVy7ffCf785BOfpa6WsrpMvM0oGF5A"
        API_URL = "https://api-lr.agent.ai/v1/action/invoke_llm"

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        # Construct the instruction with drug data context
        full_instructions = f"""
        Context: Analyzing drug with the following properties:
        - Drug SMILES: {request.drug_data.get('drug_smiles')}
        - Target Sequence: {request.drug_data.get('target_sequence')}
        - Drug-likeness: {request.drug_data.get('drug_likeness')}
        - Binding Score: {request.drug_data.get('binding_score')}
        - ADMET Properties: {request.drug_data.get('admet')}

        User Question: {request.instructions}
        """

        payload = {
            "instructions": full_instructions,
            "llm_engine": request.llm_engine
        }

        response = requests.post(
            API_URL, 
            headers=headers, 
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"AgentAI API error: {response.text}"
            )

        return {
            "query": request.instructions,
            "context": request.drug_data,
            "analysis": response.json(),
            "message": "AI analysis completed successfully"
        }

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="AgentAI request timed out"
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error connecting to AgentAI: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) 