from .base.pbx import pbx_probe
from .base.source_compare import compare_sources, SourceComparisonResult

bibliometrix = pbx_probe

__all__ = [
    "pbx_probe",
    "web_app",
    "web_stop",
    "compare_sources",
    "SourceComparisonResult",
]


def web_app(*args, **kwargs):
    from .base.app import web_app as _web_app
    return _web_app(*args, **kwargs)


def web_stop(*args, **kwargs):
    from .base.app import web_stop as _web_stop
    return _web_stop(*args, **kwargs)
