"""Check the pinned image's DeepEP legacy API; no GPU allocation or downloads."""


def check():
    from deep_ep import Buffer, Config
    from sgl_kernel import cutlass_w4a8_moe_mm, get_cutlass_w4a8_moe_mm_data

    for method in (
        "capture",
        "dispatch",
        "combine",
        "get_dispatch_layout",
        "get_dispatch_config",
        "get_combine_config",
    ):
        if not hasattr(Buffer, method):
            raise RuntimeError(
                f"The base image's DeepEP Buffer lacks {method}; legacy normal API required"
            )
    if (
        not callable(Config)
        or not callable(cutlass_w4a8_moe_mm)
        or not callable(get_cutlass_w4a8_moe_mm_data)
    ):
        raise RuntimeError("Missing DeepEP/CUTLASS W4AFP8 entrypoints")
    print(
        "glm53-cp8-dcp4-deepep-v4: DeepEP legacy Buffer and CUTLASS W4AFP8 imports verified",
        flush=True,
    )


if __name__ == "__main__":
    check()
