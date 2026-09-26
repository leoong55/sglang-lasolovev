"""Check the pinned, already upstream-required dependency without a GPU."""

from importlib.metadata import version, distribution
import ast


def main():
    if version("sgl-deep-gemm") != "0.2.0":
        raise RuntimeError("Expected sgl-deep-gemm==0.2.0 from the pinned 0.5.20 image")
    source = distribution("sgl-deep-gemm").locate_file("deep_gemm/__init__.py").read_text()
    if not any(isinstance(n,ast.FunctionDef) and n.name == "fp8_mqa_logits"
               and "max_seqlen_k" in [a.arg for a in n.args.args] for n in ast.walk(ast.parse(source))):
        raise RuntimeError("DeepGEMM compact logits API missing")
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
