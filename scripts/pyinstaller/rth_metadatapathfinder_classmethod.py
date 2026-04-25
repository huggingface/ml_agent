"""PyInstaller runtime hook: fix ``MetadataPathFinder.invalidate_caches``.

Symptom in the frozen build::

    TypeError: MetadataPathFinder.invalidate_caches() missing 1 required
    positional argument: 'cls'

Root cause: ``importlib.metadata.MetadataPathFinder.invalidate_caches``
is defined with a ``cls`` parameter but is *not* decorated as
``@classmethod`` (see CPython ``Lib/importlib/metadata/__init__.py``).
On regular Python this rarely matters, but in a PyInstaller frozen
build the class itself ends up on ``sys.meta_path``, so a call to
``importlib.invalidate_caches()`` walks the meta path and invokes the
unbound method with no arguments — boom.

This wasn't a problem until ``fastmcp``'s transitive ``key_value`` dep
started calling ``beartype.claw.beartype_this_package`` at import time,
which calls ``importlib.invalidate_caches()`` during package init.

Fix: at process startup (before any third-party import), wrap the
original function in a real ``@classmethod`` so it tolerates being
called on the class. No-op on non-frozen Python and side-effect-free.
"""

from __future__ import annotations

import importlib.metadata as _meta


_finder = getattr(_meta, "MetadataPathFinder", None)
if _finder is not None:
    _orig = _finder.__dict__.get("invalidate_caches")
    if _orig is not None and not isinstance(_orig, (classmethod, staticmethod)):
        _finder.invalidate_caches = classmethod(_orig)
