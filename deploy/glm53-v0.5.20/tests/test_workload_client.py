import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import validate_prefill_workload as client

class WorkloadClientTests(unittest.TestCase):
    def test_sse_skips_headers_and_stops_at_done(self):
        self.assertEqual(list(client.parse_sse(io.BytesIO(b': ping\n\ndata: {"text":"x"}\n\ndata: [DONE]\n'))),[{'text':'x'}])
        with self.assertRaises(RuntimeError):list(client.parse_sse([b'data: {"error":"failed"}']))
    def test_mixed_arrivals_and_constant_stream_are_bounded(self):
        mixed=client.workload('mixed')
        self.assertEqual(mixed[0],(0,131072))
        self.assertEqual([n//1024 for _,n in mixed[1:]],[4,5,6,7,10,15])
        self.assertEqual(len(client.workload('burst')),40)
        self.assertGreater(client.workload('constant')[-1][0],30)
