import xarray as xr
from datetime import datetime
import numpy as np
from earth2studio.utils.type import TimeArray, VariableArray

class DataSet:
    """An in-memory xarray dataset data source.

    Parameters
    ----------
    dataset : xr.Dataset
        The xarray dataset to use as data source.
    """

    def __init__(self, dataset: xr.Dataset):
        self.ds = dataset

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            Loaded data array
        """
        # loop over variables and concatenate the data arrays
        da_list = [self.ds[v].sel(time=np.atleast_1d(time)) for v in variable]
        da = xr.concat(da_list, dim="variable")
        da = da.assign_coords(variable=variable)
        # reorder to time variable lat lon
        da = da.transpose("time", "variable", "lat", "lon")
        return da