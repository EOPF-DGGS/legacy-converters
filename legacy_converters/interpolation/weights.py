import numpy as np
import pyproj
import sparse
import xarray as xr
import xdggs  # noqa: F401


def nearest_affine(source_grid: xr.Dataset, target_grid: xr.Dataset) -> xr.DataArray:
    """Nearest-neighbour interpolation weights based on the affine transform

    Parameters
    ----------
    source_grid : xarray.Dataset
        The source grid. Must contain at least one variable with a ``proj:transform`` attribute.
    target_grid : xarray.Dataset
        The target grid. Must have a healpix index.

    Returns
    -------
    weights : xarray.DataArray
        The interpolation weights as a sparse matrix.
    """
    source_crs = source_grid.grid4earth.crs
    target_crs = pyproj.CRS.from_epsg(4326)
    crs_transformer = pyproj.Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    )

    transform = source_grid.grid4earth.affine_transform

    lon = target_grid["longitude"].data
    lat = target_grid["latitude"].data

    nx, ny = source_grid.sizes["x"], source_grid.sizes["y"]

    x, y = crs_transformer.transform(lon, lat, direction="INVERSE")

    pixel_x, pixel_y = ~transform * (x, y)

    indices_x = np.round(np.clip(pixel_x, min=0, max=nx - 1)).astype(int)
    indices_y = np.round(np.clip(pixel_y, min=0, max=ny - 1)).astype(int)

    n_cells = target_grid.dggs.coord.size
    target_shape = (n_cells,)
    source_shape = (nx, ny)
    rows = np.arange(n_cells)

    weights = sparse.COO(
        coords=[rows, indices_x, indices_y],
        data=np.ones_like(rows, dtype="float16"),
        shape=target_shape + source_shape,
        fill_value=0,
    )

    return xr.DataArray(
        weights, dims=["cells", "x", "y"], coords=source_grid[["x", "y"]].coords
    ).assign_coords(target_grid.dggs.coord.coords)


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
