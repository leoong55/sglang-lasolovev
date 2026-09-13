"""Propagate the opt-in observer to Python multiprocessing spawn workers."""

import os

if os.environ.get("GLM53_PP_OBSERVER") == "1":
    from observe import install

    install()
