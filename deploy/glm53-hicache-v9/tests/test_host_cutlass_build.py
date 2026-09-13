"""Build orchestration and split C++ wrapper checks; no CUDA compiler is used."""
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

KIT = Path(__file__).resolve().parents[1]
SOURCE = KIT / "cutlass"
spec = importlib.util.spec_from_file_location("cutlass_build_under_test", SOURCE / "build.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class CutlassBuildTest(unittest.TestCase):
    def test_download_is_atomic_and_validates_content_before_reuse(self):
        content = b"source archive" * 1000
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as tmp, patch.object(builder, "SHA256", digest):
            root = Path(tmp)
            with patch.object(builder.urllib.request, "urlopen", return_value=io.BytesIO(b"truncated")):
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    builder.fetch_archive(root)
            self.assertFalse((root / "cutlass.tar.gz").exists())
            self.assertFalse((root / "cutlass.tar.gz.part").exists())
            with patch.object(builder.urllib.request, "urlopen", side_effect=TimeoutError):
                with self.assertRaises(TimeoutError):
                    builder.fetch_archive(root)
            with patch.object(builder.urllib.request, "urlopen", return_value=io.BytesIO(content)):
                archive = builder.fetch_archive(root)
            with patch.object(builder.urllib.request, "urlopen") as fetch:
                self.assertEqual(builder.fetch_archive(root).read_bytes(), content)
                fetch.assert_not_called()
                archive.write_bytes(b"changed cached data")
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    builder.fetch_archive(root)

    def test_humming_only_never_imports_torch_or_downloads_and_clears_stale_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "glm53_cutlass.so").write_bytes(b"old binary")
            with patch.object(builder, "__file__", str(root / "build.py")), \
                 patch.dict(os.environ, {"GLM53_BUILD_CUTLASS": "0"}), \
                 patch.dict(sys.modules, {"torch": None}), \
                 patch.object(builder, "fetch_archive") as fetch:
                builder.main()
                fetch.assert_not_called()
            info = json.loads((root / "BUILD.json").read_text())
            self.assertIs(info["enabled"], False)
            self.assertEqual(info["variants"], 0)
            self.assertFalse((root / "glm53_cutlass.so").exists())
            shutil.copy2(SOURCE / "build.py", root / "build.py")
            shutil.copy2(SOURCE / "check_build.py", root / "check_build.py")
            subprocess.run([sys.executable, str(root / "check_build.py"), "--expected", "0"],
                           check=True, capture_output=True, text=True)
            bad = subprocess.run([sys.executable, str(root / "check_build.py"), "--expected", "1"],
                                 capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)

    def test_sequential_compile_includes_all_units_and_failure_has_no_success_receipt(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                archive = root / "cutlass.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    item = tarfile.TarInfo(f"cutlass-{builder.PIN}/LICENSE.txt")
                    item.size = 7
                    tar.addfile(item, io.BytesIO(b"license"))
                (root / "cuda/bin").mkdir(parents=True)
                (root / "cuda/bin/nvcc").touch()
                (root / "csrc").mkdir()
                (root / "glm53_cutlass.so").write_bytes(b"stale binary")
                (root / "BUILD.json").write_text('{"enabled": true}')
                fake = ModuleType("torch")
                fake.__version__ = "test"
                fake.version = NS(cuda="test")
                fake.ops = NS(glm53_cutlass=NS(w4a8_mm=object()))
                ext = ModuleType("torch.utils.cpp_extension")
                ext.CUDA_HOME = str(root / "cuda")

                def load(**kwargs):
                    self.assertEqual(os.environ["MAX_JOBS"], "1")
                    self.assertEqual([Path(p).name for p in kwargs["sources"]],
                                     ["glm53_w4a8.cpp"] + [f"variant_{i}.cu" for i in range(1, 8)])
                    self.assertIn("-gencode=arch=compute_90a,code=sm_90a", kwargs["extra_cuda_cflags"])
                    self.assertIn("-O3", kwargs["extra_cuda_cflags"])
                    self.assertIs(kwargs["is_python_module"], False)
                    if fail:
                        raise RuntimeError("compiler failed")
                    path = Path(kwargs["build_directory"]) / "glm53_cutlass.so"
                    path.write_bytes(b"new linked binary")
                    return path

                ext.load = load
                with patch.object(builder, "__file__", str(root / "build.py")), \
                     patch.dict(os.environ, {"GLM53_BUILD_CUTLASS": "1", "MAX_JOBS": "32"}), \
                     patch.dict(sys.modules, {"torch": fake, "torch.utils": ModuleType("torch.utils"),
                                             "torch.utils.cpp_extension": ext}), \
                     patch.object(builder, "fetch_archive", return_value=archive):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                            builder.main()
                        self.assertFalse((root / "glm53_cutlass.so").exists())
                        self.assertFalse((root / "BUILD.json").exists())
                    else:
                        builder.main()
                        info = json.loads((root / "BUILD.json").read_text())
                        self.assertEqual(info["variants"], 7)
                        self.assertEqual(info["max_jobs"], 1)
                        self.assertEqual(info["library_sha256"], hashlib.sha256(b"new linked binary").hexdigest())
                        self.assertFalse((root / "build").exists())

    def test_build_script_uses_distinct_tags_and_verifies_before_push(self):
        for humming, fail_check in ((False, False), (True, False), (False, True)):
            with self.subTest(humming=humming, fail_check=fail_check), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                shutil.copy2(KIT / "build.sh", root / "build.sh")
                (root / "payload").write_bytes(b"build kit")
                (root / "SHA256SUMS").write_text(hashlib.sha256(b"build kit").hexdigest() + "  payload\n")
                docker = root / "docker"
                docker.write_text(
                    f"#!{sys.executable}\nimport json,os,sys\n"
                    "with open(os.environ['DOCKER_TEST_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                    "if os.environ['DOCKER_TEST_FAIL']=='1' and any(x.endswith('/check_build.py') for x in sys.argv): sys.exit(1)\n")
                docker.chmod(0o755)
                log = root / "commands.jsonl"
                env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}", DOCKER_TEST_LOG=str(log),
                           DOCKER_TEST_FAIL=str(int(fail_check)))
                args = ["bash", str(root / "build.sh"), "--push"] + (["--humming-only"] if humming else [])
                result = subprocess.run(args, env=env, capture_output=True, text=True)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(result.returncode == 0, not fail_check, result.stderr)
                self.assertIn(f"GLM53_BUILD_CUTLASS={0 if humming else 1}", calls[0])
                tag = calls[0][calls[0].index("-t") + 1]
                self.assertEqual("-humming-only-" in tag, humming)
                self.assertIn("v9.11.1", tag)
                self.assertEqual([call[0] for call in calls],
                                 ["build", "run", "run"] + ([] if fail_check else ["push"]))
                for call in calls[1:]:
                    self.assertIn(tag, call)

    @unittest.skipUnless(shutil.which("g++"), "C++ compiler required for wrapper linkage test")
    def test_split_wrappers_link_and_preserve_original_variant_and_argument_mapping(self):
        # Real wrapper source/header, stubbed CUTLASS and Tensor types. This
        # checks cross-unit linkage and argument forwarding, NOT CUDA kernels.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "torch").mkdir()
            (root / "torch/types.h").write_text(
                "#pragma once\n#include <cstdint>\nnamespace torch { struct Tensor { int value; }; }\n")
            heavy = root / "moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh"
            heavy.parent.mkdir(parents=True)
            heavy.write_text(r'''
#include <type_traits>
namespace cute { template<int V> struct Int {}; template<class...> struct Shape {};
using _512=Int<512>; using _1=Int<1>; }
namespace cutlass { namespace gemm {
struct KernelPtrArrayTmaWarpSpecializedPingpong {};
struct KernelPtrArrayTmaWarpSpecializedCooperative {};
} namespace epilogue {
struct PtrArrayTmaWarpSpecializedPingpong {};
struct PtrArrayTmaWarpSpecializedCooperative {};
} }
template<class T, class C, class KS, class ES> struct cutlass_3x_w4a8_group_gemm;
template<int M, int N, int K, int C, class KS, class ES>
struct cutlass_3x_w4a8_group_gemm<cute::Shape<cute::Int<M>,cute::Int<N>,cute::Int<K>>,
    cute::Shape<cute::Int<C>,cute::_1,cute::_1>,KS,ES> {
  static constexpr bool pp=std::is_same_v<KS,cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>;
  static_assert(pp==std::is_same_v<ES,cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong>);
  static constexpr int shape[5]={M,N,K,C,pp};
};
extern int captured[17];
template<class G> void cutlass_w4a8_group_gemm_caller(
    torch::Tensor out, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& as, const torch::Tensor& bs,
    const torch::Tensor& offsets, const torch::Tensor& problems,
    const torch::Tensor& astr, const torch::Tensor& bstr,
    const torch::Tensor& dstr, const torch::Tensor& sstr, int64_t group_size) {
  for(int i=0;i<5;++i) captured[i]=G::shape[i];
  int values[]={out.value,a.value,b.value,as.value,bs.value,offsets.value,
                problems.value,astr.value,bstr.value,dstr.value,sstr.value};
  for(int i=0;i<11;++i) captured[5+i]=values[i];
  captured[16]=group_size;
}
''')
            main = root / "main.cpp"
            main.write_text(r'''
#include "glm53_w4a8.h"
#include <cassert>
int captured[17];
int main() {
  using namespace glm53_cutlass;
  Launcher* launchers[]={launch_variant_1,launch_variant_2,launch_variant_3,
                        launch_variant_4,launch_variant_5,launch_variant_6,launch_variant_7};
  // Regression oracle: the seven configurations shipped in v9.11.
  int expected[7][5]={{128,16,512,1,0},{128,32,512,1,0},{128,64,512,1,0},
                     {64,16,512,1,1},{64,32,512,1,1},{128,32,512,2,0},{128,16,512,2,0}};
  for(int v=0;v<7;++v) {
    launchers[v]({1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},128);
    for(int j=0;j<5;++j) assert(captured[j]==expected[v][j]);
    for(int j=0;j<11;++j) assert(captured[5+j]==j+1);
    assert(captured[16]==128);
  }
}
''')
            objects = []
            for variant in range(1, 8):
                obj = root / f"variant_{variant}.o"
                subprocess.run(["g++", "-std=c++17", "-I", str(root), "-I", str(SOURCE),
                                "-x", "c++", "-c", str(SOURCE / f"variant_{variant}.cu"),
                                "-o", str(obj)], check=True, capture_output=True, text=True)
                objects.append(str(obj))
            binary = root / "check"
            subprocess.run(["g++", "-std=c++17", "-I", str(root), "-I", str(SOURCE),
                            str(main), *objects, "-o", str(binary)], check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
