"""Check the pinned, already upstream-required dependency without a GPU."""

from importlib.metadata import version


def main():
    found = version("humming-kernels")
    if found != "0.1.12":
        raise RuntimeError(
            f"Expected humming-kernels==0.1.12, found {found}. Use the pinned base image."
        )
    from humming.layer import HummingMethod
    from humming.schema import HummingInputSchema, HummingWeightSchema

    for name in ("prepare_layer_meta", "transform_humming_layer", "forward_layer"):
        if not callable(getattr(HummingMethod, name, None)):
            raise RuntimeError(f"Humming API missing {name}")
    assert HummingInputSchema is not None and HummingWeightSchema is not None
    print(
        "Humming dependency/API check passed: 0.1.12; GPU execution not tested.",
        flush=True,
    )


if __name__ == "__main__":
    main()
