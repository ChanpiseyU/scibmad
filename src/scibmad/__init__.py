from . import core

load = core.load
define = core.define
seval = core.seval
using = core.using

__all__ = ["core", "load", "define", "seval", "using"]


def __getattr__(name):
    return getattr(core, name)


def __dir__():
    return sorted(set(globals()) | set(dir(core)))
