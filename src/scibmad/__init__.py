from . import core

"""
Aliases for four core.py _JuliaProxy methods at the top level, so
callers can write scibmad.load(...) instead of scibmad.core.load(...).
"""

load = core.load
define = core.define
seval = core.seval
using = core.using

__all__ = ["core", "load", "define", "seval", "using"] # Public API names for `from scibmad import *` + doc tools, doesn't effect __getattr__/__dir__ below (still expose all of core.py)

def __getattr__(name):
    """
    Module-level fallback for scibmad.<name> when name isn't defined in this
    file.

    Forwards the lookup to core, so anything available there (including
    everything reachable through core's own __getattr__) is transparently available on scibmad.

    Arguments:
    - name: the attribute name being looked up

    Returns:
    - value: the corresponding attribute from core

    Raises:
    - AttributeError: if core doesn't have that attribute either (ultimately raised by core.py's own __getattr__)
    """
    return getattr(core, name)


def __dir__():
    """
    Module-level override for dir(scibmad)

    Combines the names defined directly in this file with everything dir(core) reports, so autocomplete reflects the full merged API rather
    than just the four aliased names.

    Arguments:
    - (none)

    Returns:
    - names: a sorted list of all attribute names visible on scibmad, combining this module's own globals with core's
    """
    return sorted(set(globals()) | set(dir(core)))
