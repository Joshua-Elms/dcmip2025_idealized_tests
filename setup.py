from pathlib import Path
import os

cwd = Path().cwd().resolve()

if cwd.name != "DCMIP2025-ML":
    raise ValueError(f"Please run this script from the root of the DCMIP2025-ML repository. Current directory: {cwd}")

msg = \
f"""
Welcome to the ML working group at DCMIP 2025!

This repository contains all necessary code for the workshop. 

These tests produce a lot of data -- possibly on the order of hundreds of gigabytes of netcdf files, depending on your configuration.

Your $HOME directory on NCAR's HPC systems has limited storage space, so you will probably run out of space if you try to store the data there. 

This script will create data directories throughout the repository and symlink them to a high-capacity storage system of your choice, such as $WORK or $SCRATCH.

$WORK has 2TB of storage space per user and no purge policy (your data will not be deleted unless you delete it yourself). 

$SCRATCH has 30TB of storage space per user, but it is purged if you do not access your data for 180 days. 

For more details, see https://ncar-hpc-docs.readthedocs.io/en/latest/storage-systems/.

Where would you like to store the data? 
"""

print(msg)

storage_choice = input("Enter 1 for $WORK, 2 for $SCRATCH, or a custom path: ").strip()

dirname = "DCMIP_2025_ML_workshop_data"

if storage_choice == "1":
    storage_path = Path(os.environ["WORK"]).resolve() / dirname
elif storage_choice == "2":
    storage_path = Path(os.environ["SCRATCH"]).resolve() / dirname
else:
    storage_path = Path(storage_choice).resolve()
    if not storage_path.exists():
        raise ValueError(f"The specified storage path {storage_path} does not exist. Please check your input.")
    
print(f"\nStoring data here: {storage_path}")

approval = input(f"\nIs this path correct? (y/n): ").strip().lower()
if approval != "y":
    raise ValueError("Setup aborted. Please run the script again with the correct path.")

print(f"\nCreating data directories at {storage_path}...")

experiments = ["A.mass_conservation", "B.hakim_and_masanam", "C.bouvier_baroclinic_wave"]

for experiment in experiments:
    experiment_path = storage_path / experiment
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    symlink_path = cwd / "experiments" / experiment / "data"
    
    # Check if the symlink already exists and remove it if necessary
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink(missing_ok=True)
    
    # Check if the target experiment directory exists
    if not experiment_path.exists():
        print(f"Warning: Could not create {experiment_path}. Check permissions.")
        continue
        
    # Create the parent directory of the symlink if it doesn't exist
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the symlink
    os.symlink(experiment_path, symlink_path)
    print(f"Making symlink:\n{symlink_path}\n\tpoints to \n{experiment_path}\n")
    
print("Setup complete! You can now run the experiments, providing the link ending in /data in the 0.config.yaml files for the 'experiment_dir' parameter.")