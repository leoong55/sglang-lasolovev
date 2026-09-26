import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[3]
s=importlib.util.spec_from_file_location('consensus',ROOT/'python/sglang/srt/mem_cache/glm53_consensus.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class ConsensusTest(unittest.TestCase):
    def test_snapshot_is_not_published_until_wait_and_digest_is_snapshot(self):
        events=[]
        class Work:
            def __init__(self,t): self.t=t
            def wait(self):
                events.append('wait')
                self.t[0]=1 # globally agreed MIN, not local 3
        def launch(t): events.append('submit');return Work(t)
        state=m.ReadyConsensus(launch)
        state.submit(3,0,19)
        self.assertEqual(events,['submit'])
        with self.assertRaises(RuntimeError):state.submit(0,0,25)
        self.assertEqual(state.finish(),(1,0))
        self.assertEqual(events,['submit','wait'])
        # A new tree digest does not affect the completed snapshot.
        state.submit(0,0,25)
        self.assertEqual(state.pending[2],25)
        state.finish()
        self.assertIsNone(state.pending)
        self.assertEqual(state.finish(),(0,0))

    def test_zero_local_events_still_submit_and_digest_mismatch_fails(self):
        calls=[]
        class Work:
            def __init__(self,t):self.t=t
            def wait(self):self.t[-1]=-7
        def launch(t):calls.append(t);return Work(t)
        state=m.ReadyConsensus(launch)
        state.submit(0,0,6)
        self.assertEqual(len(calls),1)
        with self.assertRaisesRegex(RuntimeError,'digest'):state.finish()
