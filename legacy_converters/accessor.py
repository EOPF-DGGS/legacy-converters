from __future__ import annotations

from functools import cached_property

import numpy as np
import pyproj
import xarray as xr
from affine import Affine

from legacy_converters.crs import CRSLike, create_transformer, maybe_convert


def _maybe_create_raster_index(ds):
    import rasterix

    transform = ds.grid4earth.affine_transform

    if transform is None:
        return ds

    x_dim = "x"
    y_dim = "y"
    width = ds.sizes[x_dim]
    height = ds.sizes[y_dim]

    raster_index = rasterix.RasterIndex.from_transform(
        transform,
        width=width,
        height=height,
        x_dim=x_dim,
        y_dim=y_dim,
    )

    coords = xr.Coordinates.from_xindex(raster_index)

    return ds.assign_coords(coords)


@xr.register_datatree_accessor("grid4earth")
class DataTreeConverterAccessor:
    def __init__(self, dt: xr.DataTree):
        self._dt = dt

    def _infer_crs_code(self) -> str:
        return self._dt.attrs["other_metadata"]["horizontal_CRS_code"]

    @cached_property
    def crs(self) -> pyproj.CRS:
        """The EOPF tree's spatial CRS"""
        return pyproj.CRS.from_user_input(self._infer_crs_code())

    def create_raster_indexes(self):
        return self._dt.map_over_datasets(_maybe_create_raster_index)

    def convert_to(self, target_crs: CRSLike) -> xr.DataTree:
        """Attach spatial coordinates in the target CRS

        Parameters
        ----------
        target_crs : int or str or pyproj.CRS
            The target CRS, as interpreted by :py:func:`pyproj.CRS.from_user_input`.

        Returns
        -------
        xarray.DataTree
            The current object with new coordinates.
        """
        transformer = create_transformer(self.crs, target_crs)

        return xr.DataTree.from_dict(
            {
                path: maybe_convert(node.ds, transformer)
                for path, node in self._dt.subtree_with_keys
            }
        )


def _search_attribute(ds: xr.Dataset, name: str) -> int | str | list:
    values = {
        var_name: var.attrs[name]
        for var_name, var in ds.data_vars.items()
        if name in var.attrs
    }
    unique_values = {tuple(v) if isinstance(v, list) else v for v in values.values()}
    if len(unique_values) > 1:
        raise ValueError(f"disagreement in {name}")

    return next(iter(values.values()), None)


@xr.register_dataset_accessor("grid4earth")
class DatasetConverterAccessor:
    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    def _infer_crs_code(self) -> str | None:
        return _search_attribute(self._ds, "proj:epsg")

    def _infer_affine_transform(self) -> Affine | None:
        return _search_attribute(self._ds, "proj:transform")

    def _infer_bounding_box(self) -> str | None:
        return _search_attribute(self._ds, "proj:bbox")

    @cached_property
    def crs(self) -> pyproj.CRS | None:
        crs_code = self._infer_crs_code()
        if crs_code is None:
            return crs_code

        return pyproj.CRS.from_user_input(crs_code)

    @property
    def bbox(self) -> tuple[float, ...] | None:
        index = self._ds.xindexes.get("x")
        if index is not None and hasattr(index, "xy_dims"):
            return index.bbox

        return self._infer_bounding_box()

    def minimum_bounding_rectangle(self) -> np.ndarray:
        transform = self.affine_transform

        nx = self._ds.sizes["x"]
        ny = self._ds.sizes["y"]

        x = np.array([0, nx, nx, 0], dtype="uint64")
        y = np.array([0, 0, ny, ny], dtype="uint64")

        coords_x, coords_y = transform * (x, y)

        return np.stack([coords_x, coords_y], axis=-1)

    @property
    def affine_transform(self) -> Affine | None:
        index = self._ds.xindexes.get("x")
        if index is not None and hasattr(index, "transform"):
            return index.transform()

        values = self._infer_affine_transform()
        if values is None:
            return None

        return Affine(*values)
