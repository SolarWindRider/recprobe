"""Compatibility shim: stub xgrammar.structural_tag symbols that sglang
0.5.10's function-call parsers import but the installed aarch64 xgrammar
wheel (0.1.32) does not provide.

The stubs are only referenced by structured-output / function-call parsing
(KimiK3 detector et al.), which plain-text RTL generation never touches.
Import this module BEFORE any `import sglang ...`:

    import _sglang_patch  # noqa: F401
    from sglang.srt.entrypoints.engine import Engine
"""
import xgrammar.structural_tag as _st

_STUB_NAMES = [
    "AnyTokensFormat",
    "ExcludeTokenFormat",
    "OptionalFormat",
    "StarFormat",
    "TokenFormat",
]


class _StubFormat:
    """Inert stand-in: construction records nothing, matching raises."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "xgrammar format stub hit -- structured output is not "
            "supported by this patched environment")


def patch() -> None:
    for name in _STUB_NAMES:
        if not hasattr(_st, name):
            setattr(_st, name, type(name, (_StubFormat,), {}))


patch()
