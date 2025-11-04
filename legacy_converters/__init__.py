import importlib.metadata

from legacy_converters.accessor import (  # noqa: F401
    DatasetConverterAccessor,
    DataTreeConverterAccessor,
)

__version__ = importlib.metadata.version("legacy-converters")
