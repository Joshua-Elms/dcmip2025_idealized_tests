from utils import *
import numpy as np
from itertools import product
import yaml
from pathlib import Path

# read configuration
config_path = Path('/glade/u/home/jmelms/projects/dcmip2025_idealized_tests/initial_conditions/bouvier_et_al_2024/configs/template.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
