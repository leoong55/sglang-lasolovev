import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

KIT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(KIT))
import launch
import launch_profile

class ProfilesTest(unittest.TestCase):
    def test_all_stages_validate_and_share_weights_and_core_topology(self):
        reference=json.loads((KIT/'profile.json').read_text())
        keys=['--tp-size','--ep-size','--dcp-size','--chunked-prefill-size',
              '--speculative-draft-model-path','--glm53-draft-cache-window',
              '--moe-runner-backend','--max-running-requests','--min-free-slots-delay']
        with tempfile.TemporaryDirectory() as d:
            Path(d,'config.json').write_text(json.dumps(dict(num_hidden_layers=78,kv_lora_rank=512,qk_rope_head_dim=64)))
            for file in sorted((KIT/'profiles').glob('*.json')):
                with self.subTest(profile=file.name):
                    profile=json.loads(file.read_text())
                    for key in keys:
                        if key in reference['argv']:
                            self.assertEqual(profile['argv'][profile['argv'].index(key)+1],reference['argv'][reference['argv'].index(key)+1])
                    argv=launch_profile.argv_for(profile,model_path=d,kv_tokens=150000,mem_fraction=.76)
                    result=launch.check_profile(launch.configure(argv))
                    self.assertEqual(result.chunked_prefill_size,16384)
                    self.assertEqual(profile['env']['SGLANG_DSA_FUSE_TOPK'],'0')
                    forwarded=launch.runtime_argv(argv)
                    self.assertEqual(forwarded[forwarded.index('--max-total-tokens')+1],'150000')
                    for flag in ('--glm53-dcp-prefill-mode','--glm53-prefill-attention-graph'):
                        if flag in argv:self.assertIn(flag,forwarded)
