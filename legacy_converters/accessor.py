from __future__ import annotations

from functools import cached_property

import pyproj
import xarray as xr

from legacy_converters.crs import CRSLike, create_transformer, maybe_convert


@xr.register_datatree_accessor("grid4earth")
class ConverterAccessor:
    def __init__(self, dt: xr.DataTree):
        self._dt = dt

    def _infer_crs_code(self) -> str:
        return self._dt.attrs["other_metadata"]["horizontal_CRS_code"]

    @cached_property
    def crs(self) -> pyproj.CRS:
        return pyproj.CRS.from_string(self._infer_crs_code())

    def convert_to(self, target_crs: CRSLike) -> xr.DataTree:
        transformer = create_transformer(self.crs, target_crs)

        return xr.DataTree.from_dict(
            {
                path: maybe_convert(path, node.ds, transformer)
                for path, node in self._dt.subtree_with_keys
            }
        )
