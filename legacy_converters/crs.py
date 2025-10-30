import pyproj

CRSLike = int | str | pyproj.CRS


def ensure_crs(crs_like: CRSLike) -> pyproj.CRS:
    if isinstance(crs_like, int):
        crs = pyproj.CRS.from_epsg(crs_like)
    elif isinstance(crs_like, str):
        crs = pyproj.CRS.from_string(crs_like)
    else:
        crs = crs_like

    return crs


def create_transformer(source_crs: CRSLike, target_crs: CRSLike) -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(
        ensure_crs(source_crs), ensure_crs(target_crs), always_xy=True
    )
