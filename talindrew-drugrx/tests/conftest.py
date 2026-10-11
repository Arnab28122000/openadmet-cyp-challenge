"""Import order, before pytest collects anything.

LightGBM and torch each bundle their own libomp, and on macOS arm64 whichever
loads second aborts the process — `Fatal Python error: Aborted` inside LightGBM's
`__init_from_np2d`, with no Python traceback and no failing test to point at. It
only shows up when the suite runs as a whole, because one file alone imports one
of the two; so the symptom was "every file passes, the suite dies", which reads
like a flaky runner rather than a load-order problem.

Measured: `import lightgbm, torch` is fine; `import torch, lightgbm` aborts.

`drugrx/serve.py` fixes the same order for the serving path, where the hazard is
real rather than merely annoying. This file does it for the tests, which import
the two in whatever order collection happens to reach them.
"""

from __future__ import annotations

try:  # pragma: no cover - the import itself is the test
    import lightgbm  # noqa: F401
except ImportError:  # LightGBM absent: nothing to order against
    pass
