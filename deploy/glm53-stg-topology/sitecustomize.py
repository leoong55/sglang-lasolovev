"""Propagate the opt-in observer to Python multiprocessing spawn workers."""

import os

if any(
    os.environ.get(key) == "1" for key in ("GLM53_PP_OBSERVER", "GLM53_DPA_OBSERVER")
):
    from observe import install

    install()
