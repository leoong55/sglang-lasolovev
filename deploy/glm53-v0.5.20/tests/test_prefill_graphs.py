import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[3]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
bcg=load('bcg','python/sglang/srt/layers/cp/glm53_bcg.py')
graph=load('graph','python/sglang/srt/layers/dcp/glm53_qstream_graph.py')

class PrefillGraphTest(unittest.TestCase):
    def setUp(self):
        diagnostics=load('diagnostics','python/sglang/srt/observability/glm53_prefill.py')
        self.enterContext(patch.dict('sys.modules',{'sglang.srt.observability.glm53_prefill':diagnostics}))
    def test_buckets_and_25_percent_padding(self):
        sizes=[4096,8192,12288,16384]
        self.assertEqual(bcg.validate_capture_sizes(sizes),sizes)
        self.assertEqual(bcg.replay_bucket(10000,[9000,1000],sizes),12288)
        self.assertIsNone(bcg.replay_bucket(9000,[9000],sizes))
        self.assertEqual(bcg.replay_bucket(3300,[1000,2300],sizes),4096)
        self.assertIsNone(bcg.replay_bucket(3200,[3200],sizes))
        self.assertIsNone(bcg.replay_bucket(10000,[9999],sizes))
    def test_graph_signature_tracks_storage_capacity_and_scale(self):
        kv=torch.empty(4096,1,576)
        group=object()
        key=lambda value=kv,scale=.1:graph.signature(value,(256,64,576),2048,scale,group)
        first=key()
        kv.fill_(1)
        self.assertEqual(first,key()) # data changes must reuse graph
        self.assertNotEqual(first,key(kv.clone()))
        self.assertNotEqual(first,key(kv[:2048]))
        self.assertNotEqual(first,key(scale=.2))
