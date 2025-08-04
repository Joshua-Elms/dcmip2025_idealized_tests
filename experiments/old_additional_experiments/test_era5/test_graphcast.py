import datetime
import dotenv
import datetime as dt
import torch
from earth2mip.networks import get_model
from earth2mip.initial_conditions import cds, get_data_source
from earth2mip.inference_ensemble import run_basic_inference

dotenv.load_dotenv()

model = get_model("e2mip://graphcast_operational", device="cuda:0")
output_path = "graphcast_operational_0.nc"
print(f"output_path: {output_path}")
data_source = cds.DataSource(model.in_channel_names)
print(model.in_channel_names)
ds = run_basic_inference(model, n=1, data_source=data_source, time=datetime.datetime(2018, 1, 1))
ds.to_netcdf(output_path)
print(f"output: {ds}")

"""
Graphcast channels:
[
    'z1', 'z2', 'z3', 'z5', 'z7', 'z10', 'z20', 
    'z30', 'z50', 'z70', 'z100', 'z125', 'z150', 
    'z175', 'z200', 'z225', 'z250', 'z300', 'z350',
    'z400', 'z450', 'z500', 'z550', 'z600', 'z650',
    'z700', 'z750', 'z775', 'z800', 'z825', 'z850',
    'z875', 'z900', 'z925', 'z950', 'z975', 'z1000',
    'q1', 'q2', 'q3', 'q5', 'q7', 'q10', 'q20', 
    'q30', 'q50', 'q70', 'q100', 'q125', 'q150', 
    'q175', 'q200', 'q225', 'q250', 'q300', 'q350', 
    'q400', 'q450', 'q500', 'q550', 'q600', 'q650', 
    'q700', 'q750', 'q775', 'q800', 'q825', 'q850', 
    'q875', 'q900', 'q925', 'q950', 'q975', 'q1000', 
    't1', 't2', 't3', 't5', 't7', 't10', 't20', 
    't30', 't50', 't70', 't100', 't125', 't150', 
    't175', 't200', 't225', 't250', 't300', 't350', 
    't400', 't450', 't500', 't550', 't600', 't650', 
    't700', 't750', 't775', 't800', 't825', 't850', 
    't875', 't900', 't925', 't950', 't975', 't1000', 
    'u1', 'u2', 'u3', 'u5', 'u7', 'u10', 'u20', 
    'u30', 'u50', 'u70', 'u100', 'u125', 'u150', 
    'u175', 'u200', 'u225', 'u250', 'u300', 'u350', 
    'u400', 'u450', 'u500', 'u550', 'u600', 'u650', 
    'u700', 'u750', 'u775', 'u800', 'u825', 'u850', 
    'u875', 'u900', 'u925', 'u950', 'u975', 'u1000', 
    'v1', 'v2', 'v3', 'v5', 'v7', 'v10', 'v20', 
    'v30', 'v50', 'v70', 'v100', 'v125', 'v150', 
    'v175', 'v200', 'v225', 'v250', 'v300', 'v350', 
    'v400', 'v450', 'v500', 'v550', 'v600', 'v650', 
    'v700', 'v750', 'v775', 'v800', 'v825', 'v850', 
    'v875', 'v900', 'v925', 'v950', 'v975', 'v1000', 
    'w1', 'w2', 'w3', 'w5', 'w7', 'w10', 'w20', 
    'w30', 'w50', 'w70', 'w100', 'w125', 'w150', 
    'w175', 'w200', 'w225', 'w250', 'w300', 'w350', 
    'w400', 'w450', 'w500', 'w550', 'w600', 'w650', 
    'w700', 'w750', 'w775', 'w800', 'w825', 'w850', 
    'w875', 'w900', 'w925', 'w950', 'w975', 'w1000', 
    'u10m', 'v10m', 't2m', 'msl'
]
"""