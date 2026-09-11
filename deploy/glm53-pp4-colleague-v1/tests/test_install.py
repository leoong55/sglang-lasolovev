import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE.parents[1]
spec = importlib.util.spec_from_file_location('pp_install', BUNDLE / 'install.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PackagingTests(unittest.TestCase):
    def fixture(self, root):
        manifest = json.loads((BUNDLE / 'manifest.json').read_text())
        # Remote tree contains the full pinned ancestor. The local audit
        # snapshot also has the exact baseline as its first commit.
        base = manifest['base_commit']
        if subprocess.run(['git', 'cat-file', '-e', base], cwd=ROOT, capture_output=True).returncode:
            base = subprocess.check_output(['git', 'rev-list', '--max-parents=0', 'HEAD'], cwd=ROOT, text=True).strip()
        for entry in manifest['files']:
            if entry['base_sha256'] is None:
                continue
            path = root / entry['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            data = subprocess.check_output(['git', 'show', base + ':python/sglang/' + entry['path']], cwd=ROOT)
            path.write_bytes(data)
        return manifest

    def test_exact_install_and_idempotent_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            m.install(root, BUNDLE)
            m.install(root, BUNDLE, verify_only=True)
            m.install(root, BUNDLE)
            for entry in manifest['files']:
                self.assertEqual(m.digest((root / entry['path']).read_bytes()), entry['patched_sha256'])

    def test_version_mismatch_fails_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            wrong = root / 'srt/server_args.py'
            wrong.write_text('# different upstream\n')
            before = {str(p): p.read_bytes() for p in root.rglob('*.py')}
            with self.assertRaisesRegex(RuntimeError, 'Unsupported SGLang source'):
                m.install(root, BUNDLE)
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*.py')})

    def test_bundle_overlay_matches_reviewed_source(self):
        for p in (BUNDLE / 'overlay').rglob('*.py'):
            self.assertEqual(p.read_bytes(), (ROOT / 'python/sglang' / p.relative_to(BUNDLE / 'overlay')).read_bytes())

    def test_launch_profile_has_valid_flag_names(self):
        spec = importlib.util.spec_from_file_location('pp_launch', BUNDLE / 'launch.py')
        launch = importlib.util.module_from_spec(spec)
        import sys
        sys.modules['install'] = m
        spec.loader.exec_module(launch)
        from types import SimpleNamespace
        args = SimpleNamespace(model_path='/model', port=8080, mem_fraction_static=.9, hicache_size=128, l3_path=None)
        argv = launch.command(args)
        server = (ROOT / 'python/sglang/srt/server_args.py').read_text()
        for flag in [t for t in argv if t.startswith('--')]:
            self.assertIn(flag[2:].replace('-', '_') + ':', server, flag)
        self.assertNotIn('--enable-prefill-cp', argv)
        self.assertEqual(argv[argv.index('--max-running-requests') + 1], '40')


if __name__ == '__main__':
    unittest.main()
