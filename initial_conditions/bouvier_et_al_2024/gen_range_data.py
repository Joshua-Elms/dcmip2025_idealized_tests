from utils import *
import numpy as np

fort_kwargs = dict(
    executable_path="/glade/u/home/jmelms/projects/dcmip2025_idealized_tests/initial_conditions/bouvier_et_al_2024/gen_IC_FCN.out", 
    nlat=721, 
    nlev=102, 
    zn=3, 
    zb=2.0, 
    zrh0=0.8, 
    zu0=35.0, 
    zgamma=0.005, 
    moisture=42,
)

data_dir = Path("/glade/work/jmelms/data/dcmip2025_idealized_tests/initial_conditions")
metadata_dir = Path("/glade/work/jmelms/data/dcmip2025_idealized_tests/initial_conditions/metadata")

processor_kwargs = dict(
    metadata_dir = metadata_dir,
    lat_fname="latitude.npy",
    lon_fname="longitude.npy",
    lev_fname="p_eta_levels_full.txt",
    write_data = True,
    output_to_dir=data_dir / "steady-state",
    nlat=721,
    keep_plevs=[1000, 925, 850, 700, 600, 500,
                400, 300, 250, 200, 150, 100, 50],  # 13 levels used for 73 ch SFNO
    standardize=False, 
    include_dewpt=False, # must use 74 ch hens_channel_order.txt for dewpt and q instead of r
)

mid = 288 # K
half_range = 30 # K
step = 2 # K
zt0_range = np.arange(mid-half_range, mid+half_range+step, step)
for zt0 in zt0_range:
    fort_data_path = Path(f"/glade/work/jmelms/data/dcmip2025_idealized_tests/initial_conditions/steady-state/raw_fort_output/output_{zt0}.csv")
    if not fort_data_path.exists(): # fortran refuses to write over extant file ... fine.
        out, err = run_fortran_executable(zt0=zt0, filename=fort_data_path, **fort_kwargs)
    
    ds = process_individual_fort_file(
        fort_path=Path(fort_data_path),
        f_out_name=f"steady_state_{zt0}.nc",
        **processor_kwargs
    )
    print(ds.dims)
    
    print(f"saved ds for {zt0}")
    

print("program should end now, feel free to end via CTRL+C")