The directory contains the codes necessary to generate the stable initial condition from Bouvier et al. 2024. 

It is organized as follows: 
- `fortran_metadata.py` - contains helper code to modify the `gen_IC_FCN.F90` Fortran file. Only needed if producing IC for a model other than FCNv2_sm
- `gen_IC_FCN.F90` - Fortran program adapted from Bouvier et al. 2024 to produce hydrostatically balanced initial condition for FCNv2_sm
- `utils.py` - contains helper code to run and process data from `gen_IC_FCN.out` into a usable format
- `gen_range_data.py` - example script which uses functions from `utils.py` to drive `gen_IC_FCN.out` (Fortran executable) to produce a range of initial conditions which are written to disk for use with FCNv2_sm. 

### Use Case 1: Generating FourCastNet-Formatted Initial Conditions

Populate with instructions


### Use Case 2: Generating Custom-Formatted Initial Conditions

Populate with instructions


Details on the CLI for gen_IC_FCN.out

Compile with `ifort gen_IC_FCN.F90 -o gen_IC_FCN.out`

gen_IC_FCN.F90 has been hardcoded (values for VETAF, GELAT_DEG, and GELAT) for FourCastNet, particularly the 73 channel SFNO version

Args in order:
    NLAT     - number of latitude ticks, must agree with GELAT_DEG and GELAT
    NLEV     - number of levels, must agree with VETAF
    ZN       - Jet width
    ZB       - Jet height
    ZRH0     - Surface level relative humidity (%)
    ZT0      - Average surface virtual temperature (K)
    ZU0      - works with ZB to affect amplitude of zonal mean wind speed (m/s)
    ZGAMMA   - Lapse rate (K/m)
    MOISTURE - 41 for dry run, 42 for moist
    FILENAME - Output location for csv file containing NLAT x NLEV rows and all fields needed to run FCN

Defaults: 
    NLAT     - 721
    NLEV     - 15
    ZN       - 3
    ZB       - 2.0
    ZRH0     - 0.8
    ZT0      - 288.0
    ZU0      - 35.0
    ZGAMMA   - 0.005
    MOISTURE - 42
    FILENAME - "fields.csv"

Running downloaded version: 
    ./gen_IC.out 320 137 3 2.0 0.8 288.0 35.0 0.005 42 fields.csv

Running FCN (this) version:
    ./gen_IC_FCN.out 721 102 3 2.0 0.8 288.0 35.0 0.005 42 fields.csv


### Citations

