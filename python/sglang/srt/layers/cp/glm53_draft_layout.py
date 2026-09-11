"""Bounded GLM53 draft geometry shared by the solver and allocation."""

import os


def bounded_draft_enabled():
    return os.environ.get("SGLANG_GLM53_DRAFT_CACHE_WINDOW", "0") == "2048"


def bounded_draft_geometry(max_requests, page_size=256):
    if max_requests is None or max_requests < 1 or page_size != 256:
        raise ValueError("Bounded GLM53 draft requires explicit running capacity and page256")
    # Window + left page alignment + a separate page for the eight verify rows.
    stride = 2048 + 2 * page_size
    rows = (int(max_requests) + 1) * stride
    # MHAPool includes a sentinel page. Tags: int64 virtual ID + int32 version.
    # The logical-version table costs another 4 bytes per logical target token;
    # only its sentinel padding is fixed, the solver charges the rest per token.
    fixed_bytes = (rows + page_size) * (3072 + 8 + 4) + page_size * 4
    return rows, stride, fixed_bytes
