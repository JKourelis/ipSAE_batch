"""
Data readers for different protein structure prediction backends.

Supported backends:
- alphafold3: AlphaFold3 (CIF + JSON)
- colabfold: ColabFold/AlphaFold2-multimer (PDB + JSON)
- boltz2: Boltz2 (CIF + NPZ + JSON)
- intellifold: IntelliFold (CIF + JSON, AF3-like format)

Auto-detection:
The backend can be automatically detected from file patterns using detect_backend().
"""

from .base import BaseReader, FoldingResult, ReaderError
from .auto_detect import detect_backend, detect_backend_with_confidence, detect_backend_for_input_folder

__all__ = [
    'BaseReader', 'FoldingResult', 'ReaderError', 'get_reader',
    'detect_backend', 'detect_backend_with_confidence', 'detect_backend_for_input_folder'
]

# Registry of available readers (populated on import)
_READERS = {}


def register_reader(backend_name: str):
    """Decorator to register a reader class for a backend."""
    def decorator(cls):
        _READERS[backend_name] = cls
        return cls
    return decorator


def get_reader(backend: str) -> BaseReader:
    """
    Get a reader instance for the specified backend.

    Args:
        backend: One of 'alphafold3', 'colabfold', 'boltz2', 'intellifold'

    Returns:
        Initialized reader instance

    Raises:
        ReaderError: If backend is not supported
    """
    # Lazy import readers to avoid circular imports
    if not _READERS:
        _import_readers()

    backend = backend.lower()
    if backend not in _READERS:
        available = ', '.join(sorted(_READERS.keys()))
        raise ReaderError(f"Unknown backend '{backend}'. Available: {available}")

    return _READERS[backend](backend)


def _import_readers():
    """Import all reader modules to register them."""
    try:
        from . import alphafold3
    except ImportError:
        pass

    try:
        from . import colabfold
    except ImportError:
        pass

    try:
        from . import boltz2
    except ImportError:
        pass

    try:
        from . import intellifold
    except ImportError:
        pass


def list_backends() -> list:
    """List all available backends."""
    if not _READERS:
        _import_readers()
    return sorted(_READERS.keys())
