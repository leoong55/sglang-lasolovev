import importlib.util
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import render


class RenderContract(unittest.TestCase):
    def test_serving_owns_only_dedicated_resources(self):
        objects = render.serving('pp2', 'ghcr.io/leoong55/sglang-lasolovev@sha256:' + 'a' * 64, 'b' * 40, 'node1')
        for obj in objects:
            self.assertEqual(obj['metadata']['namespace'], 'inf-glm53')
            self.assertEqual(obj['metadata']['labels']['app.kubernetes.io/part-of'], render.PART)
        spec = objects[0]['spec']['template']['spec']
        self.assertEqual(spec['schedulerName'], 'hami-scheduler')
        self.assertNotIn('nodeName', spec)
        self.assertFalse(spec['automountServiceAccountToken'])
        c = spec['containers'][0]
        self.assertEqual(c['resources']['limits']['nvidia.com/gpu'], '8')
        self.assertEqual(c['resources']['limits']['nvidia.com/gpucores'], '100')
        self.assertTrue(next(m for m in c['volumeMounts'] if m['name'] == 'model')['readOnly'])
        self.assertEqual(objects[1]['spec']['type'], 'ClusterIP')

    def test_unpinned_image_rejected(self):
        with self.assertRaises(ValueError):
            render.serving('dpa8', 'ghcr.io/leoong55/sglang-lasolovev:latest', 'b' * 40, 'node1')

    def test_benchmark_is_cpu_only_and_uses_same_weights(self):
        obj = render.benchmark('trial-dpa2', 'dpa2', 'digest', 'commit', 'node1')
        spec = obj['spec']['template']['spec']
        c = spec['containers'][0]
        self.assertNotIn('nvidia.com/gpu', c['resources']['limits'])
        self.assertEqual(c['image'], render.BENCH_IMAGE)
        self.assertIn('/model', c['command'])
        self.assertFalse(spec['automountServiceAccountToken'])

    def test_export_name_validation(self):
        with self.assertRaises(ValueError):
            render.job('../other', 'node1', [])


if __name__ == '__main__':
    unittest.main()
