import xarray as xr


def nearest(source_grid: xr.Dataset, target_grid: xr.Dataset) -> xr.DataArray:
    """Nearest-neighbour interpolation weights"""
    pass


def gaussian_weights(source_grid: xr.Dataset, target_grid: xr.Dataset) -> xr.DataArray:
    """Weights based on a distance-dependent gaussian function

    The algorithm first finds the two closest healpix cells on both the northern
    and southern ring closest to the source point, and assigns the source
    point's weights for these cells as the gaussian function evaluated at the
    distance between both cell centers.
    """
    # steps:
    # - find the 4 closest cells in a vectorized way
    #   - the cdshealpix crate does the following (to be implemented in healpix-geo):
    #     - find the enclosing cell
    #     - find all immediate neighbours
    #     - figure out the quadrant within the enclosing cell
    #     - choose neighbours depending on the quadrant
    # - for each, compute the (squared) distance in radians
    # - transform the distance into weights
    # - arrange into a sparse matrix
    pass
