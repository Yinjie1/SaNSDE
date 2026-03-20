
import os
import shutil
import numpy as np
from ase.io import read
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from SaNSDE_new import Population, randomCluster, localOptimize

def test_workflow():
    print("--- Step 1: Read POSCAR and add to Population ---")
    
    # 1. Read POSCAR
    try:
        atoms = read('POSCAR')
        if isinstance(atoms, list):
            atoms = atoms[0]
        print(f"Read POSCAR: {atoms}")
    except FileNotFoundError:
        print("Error: POSCAR not found.")
        return

    # Initialize Population
    pop = Population()

    # Set Calculator (LJ)
    # Using sigma=2.88 to match Au approx distance, epsilon=0.1
    atoms.calc = LennardJones(sigma=2.88, epsilon=0.1)

    # Calculate energy and forces
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    print(f"Initial Energy (LJ): {e:.4f} eV")

    # Add to population
    pop.addCluster(atoms)
    
    # Manually update energy/forces for first cluster
    # Note: access via pop.energies list directly as per class definition
    pop.energies[0] = e
    pop.forces[0] = f

    print(f"Population Size after Step 1: {pop.size}")

    print("\n--- Step 2: Add 5 randomly generated clusters ---")
    
    for i in range(5):
        # Create fresh calculator for each
        calc = LennardJones(sigma=2.88, epsilon=0.1)
        
        # Add random cluster
        # Using a=2.88 (approx Au-Au distance), delta=0.5
        randomCluster(pop, 
                      a=2.88, 
                      delta=0.5, 
                      cell=[15, 15, 15], 
                      potential=calc, 
                      optimizer=BFGS, 
                      fmax=0.05, 
                      steps=50) # Reduced steps for speed in test
        
        print(f"Added Cluster {i+1}. Energy: {pop.energies[-1]:.4f}")

    print(f"Final Population Size: {pop.size}")

    print("\n--- Step 4: Import from JSON ---")
    pop_new = Population()
    pop_new.import_population(name="TestPop_CIF")
    print(f"Imported Population Size: {pop_new.size}")
    if pop_new.size > 0:
        print(f"First Cluster Energy: {pop_new.energies[0]}")
        print(f"Last Cluster Energy: {pop_new.energies[-1]}")
    
    print("Test completed.")

if __name__ == "__main__":
    test_workflow()
