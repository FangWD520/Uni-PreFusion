import inspect
import os
import sys


_UNILION_MAMBA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'ops', 'mamba'))
_MAMBAFUSION_MAMBA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'mamba_diffv', 'mamba'))

# Select which simple Mamba block UniLION modules use.
# Default is external to keep the previous accidental one-way structure.
#   external/mambafusion/unidirectional/legacy: use MambaFusion's Block2.
#   auto/unilion/bidirectional: prefer UniLION's renamed bidirectional library.
_BACKEND_ALIASES = {
    'auto': 'auto',
    'unilion': 'auto',
    'bidirectional': 'auto',
    'bi': 'auto',
    'external': 'external',
    'mambafusion': 'external',
    'unidirectional': 'external',
    'uni': 'external',
    'legacy': 'external',
}
_REQUESTED_BACKEND = os.environ.get('UNILION_MAMBA_BACKEND', 'external').strip().lower()
if _REQUESTED_BACKEND not in _BACKEND_ALIASES:
    raise ValueError(
        "Unsupported UNILION_MAMBA_BACKEND={!r}; choose one of {}.".format(
            _REQUESTED_BACKEND, sorted(_BACKEND_ALIASES)))
MAMBA_BACKEND = _BACKEND_ALIASES[_REQUESTED_BACKEND]


def _supports_simple_block_api(block_cls):
    """Return True if block_cls matches UniLION/MambaFusion simple block API."""
    try:
        params = inspect.signature(block_cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "d_model" in params and "d_state" in params


def _has_bidirectional_state(block_cls):
    try:
        block = block_cls(d_model=8, d_state=2, d_conv=2, expand=1)
    except Exception:
        return False
    return hasattr(block.mamba, "A_b_log")


def _module_file(cls):
    module = sys.modules.get(cls.__module__)
    return os.path.abspath(getattr(module, '__file__', '') or '')


def _prepend_path(path):
    if not path or not os.path.isdir(path):
        return
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _import_unilion_original_block():
    """Import UniLION's original bidirectional Mamba Block under an isolated name."""
    _prepend_path(_UNILION_MAMBA_ROOT)
    from unilion_mamba_ssm import Block as block_cls
    return block_cls


def _import_mambafusion_external_block():
    """Import MambaFusion's simple one-way Block2."""
    _prepend_path(_MAMBAFUSION_MAMBA_ROOT)
    from mamba_ssm import Block2 as block_cls
    return block_cls


try:
    _OrigBlock = _import_unilion_original_block()
except ImportError:
    _OrigBlock = None

if _OrigBlock is not None and _supports_simple_block_api(_OrigBlock) and _has_bidirectional_state(_OrigBlock):
    UniLIONMambaBlock = _OrigBlock
else:
    UniLIONMambaBlock = None

try:
    _Block2 = _import_mambafusion_external_block()
except ImportError:
    _Block2 = None

if _Block2 is not None and _supports_simple_block_api(_Block2):
    ExternalMambaBlock = _Block2
else:
    ExternalMambaBlock = None

if MAMBA_BACKEND != 'external' and UniLIONMambaBlock is None:
    try:
        from mamba_ssm import Block as _Block
    except ImportError as exc:
        raise ImportError(
            "Failed to import a UniLION-compatible mamba_ssm.Block."
        ) from exc
    UniLIONMambaBlock = _Block

if MAMBA_BACKEND == 'external':
    if ExternalMambaBlock is None:
        raise ImportError(
            "UNILION_MAMBA_BACKEND=external requested, but MambaFusion "
            "mamba_ssm.Block2 could not be imported.")
    MambaBlock = ExternalMambaBlock
elif UniLIONMambaBlock is not None and _supports_simple_block_api(UniLIONMambaBlock):
    MambaBlock = UniLIONMambaBlock
elif ExternalMambaBlock is not None:
    MambaBlock = ExternalMambaBlock
else:
    raise ImportError("Incompatible mamba_ssm Block API detected.")

MAMBA_BACKEND_INFO = dict(
    requested=_REQUESTED_BACKEND,
    selected='external' if MambaBlock is ExternalMambaBlock else 'unilion',
    block_module=MambaBlock.__module__,
    block_file=_module_file(MambaBlock),
    bidirectional=_has_bidirectional_state(MambaBlock),
)

__all__ = [
    "MambaBlock", "UniLIONMambaBlock", "ExternalMambaBlock",
    "MAMBA_BACKEND", "MAMBA_BACKEND_INFO"]

