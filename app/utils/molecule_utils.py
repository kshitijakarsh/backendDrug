from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np
from typing import Dict, List, Union
import random

# Move all the utility functions here
def check_drug_likeness(smiles: str) -> Dict[str, Union[float, str]]:
    """
    Checks if a given molecule (SMILES format) follows Lipinski's Rule of Five.
    
    Args:
        smiles (str): SMILES representation of the molecule
        
    Returns:
        Dict with molecular properties and Lipinski's rule check results
    """
    try:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES string"}

        # Calculate properties
        mol_weight = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        # Check Lipinski's rules
        passes_lipinski = (
            mol_weight <= 500 and 
            logp <= 5 and 
            hbd <= 5 and 
            hba <= 10
        )

        return {
            "molecular_weight": round(mol_weight, 3),
            "logP": round(logp, 3),
            "HBD": hbd,
            "HBA": hba,
            "drug_likeness": "Pass" if passes_lipinski else "Fail",
            "message": "Molecule satisfies Lipinski's Rule of Five" if passes_lipinski else "Molecule does NOT satisfy Lipinski's Rule"
        }

    except Exception as e:
        return {"error": f"Drug-likeness calculation failed: {str(e)}"}

def predict_admet(smiles: str) -> Dict[str, Union[Dict, str]]:
    """
    Predict ADMET properties using RDKit descriptors.
    
    Args:
        smiles (str): SMILES representation of the molecule
        
    Returns:
        Dict with ADMET properties and predictions
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES string"}

        # Calculate molecular properties
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        # Predict absorption
        absorption_prob = "High" if (tpsa < 140 and rotatable_bonds < 10) else "Low"
        bbb_prob = "High" if (mw < 400 and logp < 5 and tpsa < 90) else "Low"

        # Predict metabolism (based on Lipinski's rules)
        metabolism_risk = "Low" if (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10) else "High"

        # Predict toxicity (basic rules)
        toxicity_risk = "Low" if (logp < 5 and tpsa > 75) else "High"

        return {
            "absorption": {
                "intestinal_absorption": absorption_prob,
                "blood_brain_barrier": bbb_prob,
                "TPSA": round(tpsa, 2),
                "rotatable_bonds": rotatable_bonds
            },
            "metabolism": {
                "risk_level": metabolism_risk,
                "molecular_weight": round(mw, 2),
                "logP": round(logp, 2)
            },
            "toxicity": {
                "risk_level": toxicity_risk,
                "hbd": hbd,
                "hba": hba
            },
            "message": "ADMET properties predicted using RDKit descriptors"
        }

    except Exception as e:
        return {"error": f"ADMET prediction failed: {str(e)}"}

def generate_molecule(num_samples: int = 1, seed_smiles: str = None) -> Dict[str, Union[List[str], str]]:
    """
    Generate novel drug-like molecules using RDKit's structure generation.
    """
    try:
        generated_molecules = []
        fragments = [
            "CC", "c1ccccc1", "C1CCCCC1", "c1ccncc1",
            "CC(=O)N", "CCO", "CCN", "CC(=O)O",
            "CN", "CF", "CCl", "CBr",
            "c1cccnc1", "c1ccco1", "c1ccs1"
        ]
        
        if seed_smiles:
            # Use the seed molecule as a template
            seed_mol = Chem.MolFromSmiles(seed_smiles)
            if not seed_mol:
                return {"error": "Invalid seed SMILES string"}
                
            # Generate similar molecules by modifying the seed
            for _ in range(num_samples):
                modified_mol = modify_molecule(seed_mol)
                if modified_mol:
                    try:
                        smiles = Chem.MolToSmiles(modified_mol)
                        if check_drug_likeness(smiles).get("drug_likeness") == "Pass":
                            generated_molecules.append(smiles)
                    except:
                        continue
        else:
            # Generate molecules from scratch using common fragments
            attempts = 0
            max_attempts = num_samples * 10  # Allow multiple attempts per requested sample
            
            while len(generated_molecules) < num_samples and attempts < max_attempts:
                attempts += 1
                try:
                    # Randomly combine 2-4 fragments
                    num_fragments = random.randint(2, 4)
                    selected_fragments = random.sample(fragments, num_fragments)
                    
                    # Create base molecule from first fragment
                    mol = Chem.MolFromSmiles(selected_fragments[0])
                    
                    # Add remaining fragments
                    for fragment in selected_fragments[1:]:
                        fragment_mol = Chem.MolFromSmiles(fragment)
                        if fragment_mol:
                            mol = Chem.CombineMols(mol, fragment_mol)
                    
                    # Convert to SMILES
                    if mol:
                        smiles = Chem.MolToSmiles(mol)
                        # Check if it's drug-like
                        if check_drug_likeness(smiles).get("drug_likeness") == "Pass":
                            if smiles not in generated_molecules:
                                generated_molecules.append(smiles)
                except:
                    continue
        
        if not generated_molecules:
            return {
                "error": "Failed to generate valid drug-like molecules. Try different parameters or seed SMILES."
            }
            
        return {
            "generated_molecules": generated_molecules,
            "num_molecules": len(generated_molecules),
            "message": "Generated drug-like molecules that pass Lipinski's Rule of Five"
        }
        
    except Exception as e:
        return {"error": f"Molecule generation failed: {str(e)}"}

def modify_molecule(mol):
    """Helper function to modify a molecule"""
    try:
        # Make a copy of the molecule
        new_mol = Chem.RWMol(mol)
        
        # Randomly choose a modification
        modification = random.choice([
            'add_atom',
            'remove_atom',
            'modify_bond',
            'add_fragment'
        ])
        
        if modification == 'add_atom':
            # Add a random atom (C, N, O) to a random position
            atom_types = [6, 7, 8]  # C, N, O
            atom_idx = random.randint(0, new_mol.GetNumAtoms()-1)
            new_atom_idx = new_mol.AddAtom(Chem.Atom(random.choice(atom_types)))
            new_mol.AddBond(atom_idx, new_atom_idx, Chem.BondType.SINGLE)
            
        elif modification == 'remove_atom':
            if new_mol.GetNumAtoms() > 5:  # Keep at least 5 atoms
                atom_idx = random.randint(0, new_mol.GetNumAtoms()-1)
                new_mol.RemoveAtom(atom_idx)
                
        elif modification == 'modify_bond':
            if new_mol.GetNumBonds() > 0:
                bond_idx = random.randint(0, new_mol.GetNumBonds()-1)
                bond = new_mol.GetBondWithIdx(bond_idx)
                new_order = random.choice([
                    Chem.BondType.SINGLE,
                    Chem.BondType.DOUBLE
                ])
                bond.SetBondType(new_order)
                
        elif modification == 'add_fragment':
            fragments = ["CC", "CN", "CO", "CF", "CCl", "c1ccccc1"]
            fragment = Chem.MolFromSmiles(random.choice(fragments))
            if fragment:
                new_mol = Chem.CombineMols(new_mol, fragment)
        
        # Try to sanitize the molecule
        Chem.SanitizeMol(new_mol)
        return new_mol
    except:
        return None

def create_molecule_from_fragments(fragments):
    """Helper function to create molecules from fragments"""
    # ... (previous implementation)
