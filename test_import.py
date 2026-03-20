
import os
import shutil
from ase import Atoms
from SaNSDE_new import Population

def test_import_autodetect():
    print("--- Testing Import Auto-detection ---")

    # 1. Setup: Create dummy population and export in different formats
    pop = Population(symbol="H2")
    
    # Add a simple H2 molecule
    h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    h2.set_cell([10, 10, 10])
    h2.set_pbc(True)
    pop.addCluster(h2)
    
    # Export as XYZ
    xyz_dir_name = "TestAuto_XYZ"
    print(f"Exporting to {xyz_dir_name} (format='xyz')...")
    pop.export(name=xyz_dir_name, format="xyz")

    # Export as VASP
    vasp_dir_name = "TestAuto_VASP"
    print(f"Exporting to {vasp_dir_name} (format='vasp')...")
    pop.export(name=vasp_dir_name, format="vasp")
    
    # 2. Test Import (XYZ)
    print(f"\nTesting import from {xyz_dir_name} (no format specified)...")
    pop_new = Population()
    pop_new.import_population(name=xyz_dir_name)
    
    if pop_new.size == 1:
        print("SUCCESS: Imported 1 cluster from XYZ.")
        print(f"Cluster 0 symbol: {pop_new.clusters[0].get_chemical_formula()}")
    else:
        print(f"FAILURE: Expected 1 cluster, got {pop_new.size}")

    # 3. Test Import (VASP)
    print(f"\nTesting import from {vasp_dir_name} (no format specified)...")
    # Must use confirm=True to overwrite
    pop_new.import_population(name=vasp_dir_name, confirm=True)
    
    if pop_new.size == 1:
        print("SUCCESS: Imported 1 cluster from VASP.")
        print(f"Cluster 0 symbol: {pop_new.clusters[0].get_chemical_formula()}")
    else:
        print(f"FAILURE: Expected 1 cluster, got {pop_new.size}")

    # Cleanup
    # shutil.rmtree(os.path.join('./Populations', xyz_dir_name), ignore_errors=True)
    # shutil.rmtree(os.path.join('./Populations', vasp_dir_name), ignore_errors=True)

if __name__ == "__main__":
    test_import_autodetect()
