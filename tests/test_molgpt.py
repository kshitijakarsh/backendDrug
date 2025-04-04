# tests/test_molgpt.py

from app.services.molgpt_service import generate_molecule

def test_molgpt_generation():
    result = generate_molecule("C")
    assert isinstance(result, str)
    assert len(result) > 0
    print("Generated Molecule:", result)
