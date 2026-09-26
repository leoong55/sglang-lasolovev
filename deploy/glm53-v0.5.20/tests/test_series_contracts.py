import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

class SeriesContractTests(unittest.TestCase):
    def test_real_cli_schema_accepts_switches_and_rollback(self):
        utils=load('glm_arg_utils','python/sglang/srt/arg_groups/arg_utils.py')
        with patch.dict(sys.modules,{'sglang.srt.arg_groups.arg_utils':utils,
                                   'sglang.srt.utils.common':N(human_readable_int=int)}):
            fields=load('glm_schedule','python/sglang/srt/arg_groups/fields/schedule.py')
        sys.modules[fields.__name__]=fields
        parser=argparse.ArgumentParser()
        names=['prefill_interleaving','prefill_interleaving_mode','glm53_dsa_indexer_mode',
               'glm53_dsa_logits_workspace_mib','glm53_dcp_prefill_mode',
               'glm53_hicache_event_sync','glm53_prefill_attention_graph']
        utils.add_cli_args_from_dataclass(parser,fields.Schedule,fields=names)
        args=parser.parse_args(['--prefill-interleaving','--prefill-interleaving-mode','adaptive',
                               '--glm53-dsa-indexer-mode','compact','--glm53-dsa-logits-workspace-mib','1024',
                               '--glm53-dcp-prefill-mode','q-stream','--glm53-hicache-event-sync','pipelined',
                               '--glm53-prefill-attention-graph','on'])
        self.assertTrue(args.prefill_interleaving)
        self.assertEqual(args.glm53_dsa_logits_workspace_mib,1024)
        rollback=parser.parse_args(['--no-prefill-interleaving'])
        self.assertFalse(rollback.prefill_interleaving)
        self.assertEqual(rollback.glm53_dcp_prefill_mode,'kv-gather')
        self.assertEqual(rollback.glm53_prefill_attention_graph,'off')

    def test_runtime_profile_rejects_silent_graph_fallback(self):
        mod=load('runtime_profile','python/sglang/srt/observability/glm53_runtime_profile.py')
        cfg=N(chunked_prefill_size=16384,min_free_slots_delay=1,
              glm53_dsa_indexer_mode='legacy',glm53_dcp_prefill_mode='kv-gather',
              glm53_prefill_attention_graph='off',glm53_hicache_event_sync='sync')
        runner=N(is_draft_worker=False,max_running_requests=40,max_total_num_tokens=100000,
                 attn_backend=N(dsa_prefill_impl='flashmla_sparse_q8'),
                 prefill_cuda_graph_runner=N(backend=type('BreakableCudaGraphBackend',(),{})(),
                                            capture_num_tokens=[8192,16384]))
        expected=dict(chunk=16384,slots=40,delay=1,prefill_backend='flashmla_sparse_q8',
                      graph='breakable',buckets=[8192,16384])
        with patch.dict('os.environ',{'SGLANG_GLM53_EXPECTED_RUNTIME':json.dumps(expected)}), \
             patch.dict(sys.modules,{'sglang.srt.runtime_context':N(get_schedule=lambda:cfg)}):
            mod.validate(runner)
            runner.prefill_cuda_graph_runner=None
            with self.assertRaisesRegex(RuntimeError,'graph mismatch'):mod.validate(runner)
