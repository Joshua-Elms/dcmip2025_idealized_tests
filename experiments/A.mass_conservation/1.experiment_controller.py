from utils_E2S import general
from experiment import run_experiment
from pathlib import Path
import subprocess

# load config file
config_path = Path(__file__).parent / "0.config.yaml"
config = general.read_config(config_path)
print(f"Running experiment '{config['experiment_name']}' with models: {config['models']}")

# get ready to output data to disk
exp_dir = general.prepare_output_directory(config)

# see whether debug run or full run
if config["debug_run"]:
    print("Running in debug mode. The experiment function will be invoked directly instead of via subprocess.")
    if len(config["models"]) != 1:
        raise ValueError("In debug mode, only one model can be run at a time. Please set 'debug_run' to False or choose a single model in the config. Exiting.")
    status = run_experiment(config["models"][0], str(config_path.resolve()))
    
else:
    print("Running in full mode. The 'run_experiment' function will be invoked for each model via subprocess.")
    # loop over models and run the experiment for each
    for model_name in config["models"]:
        subprocess.run(["python", "-c", f"from experiment import run_experiment; run_experiment('{model_name}', '{str(config_path.resolve())}')"], check=True)
        
print("Experiment completed. Results written to ", exp_dir)
