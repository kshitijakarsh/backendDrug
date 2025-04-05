from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import (
    generate,
    lipinski,
    binding,
    admet,
    agent,
    agent_ai
)

app = FastAPI(
    title="Drug Analysis API",
    description="Comprehensive API for drug analysis and prediction with AI insights",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(generate.router, tags=["Molecule Generation"])
app.include_router(lipinski.router, tags=["Drug-likeness"])
app.include_router(binding.router, tags=["Binding Prediction"])
app.include_router(admet.router, tags=["ADMET Properties"])
app.include_router(agent.router, tags=["Full Analysis"])
app.include_router(agent_ai.router, tags=["AI Analysis"])

@app.get("/")
async def root():
    """Root endpoint that provides API information"""
    return {
        "name": "Drug Analysis API",
        "version": "1.0.0",
        "description": "API for drug analysis and prediction with AI insights",
        "endpoints": {
            "/generate": "Generate novel drug-like molecules",
            "/lipinski": "Check Lipinski's Rule of Five",
            "/binding": "Predict drug-target binding",
            "/admet": "Predict ADMET properties",
            "/agent": "Perform complete drug analysis",
            "/agentai": "Get AI-powered analysis and recommendations"
        }
    }