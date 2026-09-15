import copy
import importlib.util
import unittest
from pathlib import Path

KIT=Path(__file__).resolve().parents[1]

def module(name):
    spec=importlib.util.spec_from_file_location(name,KIT/(name+'.py'))
    obj=importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj

render=module('render').render
compare=module('compare')
bench=module('bench_decode')
IMAGE='ghcr.io/leoong55/sglang-lasolovev:test@sha256:'+'a'*64


def arm(profile,**kwargs):
    doc=render(image=IMAGE,profile=profile,**kwargs)[0]
    pod=doc['spec']['template']['spec']
    return dict(pod=pod,container=pod['containers'][0],
                summary=dict(valid=True,node='same-node',image=IMAGE,dataset_sha256='same',concurrency=80,input_tokens=8192,output_tokens=2048))


class Profiles(unittest.TestCase):
    def test_topology_comparison_changes_only_declared_fields(self):
        compare.check(arm('tp8-dcp4-decode'),arm('pp4-decode'),'topology')

    def test_scheduler_comparison_changes_one_switch(self):
        compare.check(arm('pp4-decode'),arm('pp4-decode',skip=True),'skip')

    def test_other_scheduler_switch_is_rejected(self):
        with self.assertRaises(ValueError): compare.check(arm('pp4-decode'),arm('pp4-decode',park=True,skip=True),'skip')

    def test_archive_recipe_is_not_a_clean_decode_arm(self):
        with self.assertRaises(ValueError): compare.check(arm('pp4-decode'),arm('pp4-archive'),'topology')

    def test_fp8_requires_real_pvc_name(self):
        with self.assertRaises(ValueError): render(image=IMAGE,profile='pp4-decode',weights='fp8')

    def test_model_quantization_cannot_change_inside_topology_ab(self):
        with self.assertRaises(ValueError): compare.check(arm('tp8-dcp4-decode'),arm('pp4-decode',weights='fp8',model_pvc='actual'),'topology')

    def test_local_decode_graph_covers_microbatch(self):
        for c in (40,80,120,160):
            a=arm('pp4-decode',concurrency=c)
            args=compare.argv_map(a['container']['args'])
            self.assertEqual(args['--pp-max-micro-batch-size'],args['--cuda-graph-max-bs-decode'])
            self.assertEqual(args['--cuda-graph-bs-decode'][-1],str(c//4))

    def test_no_accidental_production_selector(self):
        a=arm('pp4-decode')
        self.assertNotIn('sglang-glm53-dcp4',str(a['pod'].get('affinity',{})))
        docs=render(image=IMAGE,profile='pp4-decode')
        self.assertEqual(docs[0]['spec']['selector']['matchLabels'],docs[1]['spec']['selector'])
        self.assertEqual(docs[0]['metadata']['name'],'sglang-glm53-h200-ab')


class Measurement(unittest.TestCase):
    def rows(self, offsets=(0,1)):
        return [{'sent':0,'wanted':300,'events':[[offset+n*.1,n] for n in range(1,301)]} for offset in offsets]

    def test_common_window_uses_all_requests_and_excludes_prefill(self):
        result=bench.summarize(self.rows())
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(result['window_start'],7.4)
        self.assertAlmostEqual(result['window_end'],23.6)
        self.assertAlmostEqual(result['decode_output_tokens_per_second'],20,delta=.2)

    def test_queueing_without_residency_invalidates_wave(self):
        self.assertFalse(bench.summarize(self.rows((0,40)))['valid'])

    def test_failed_request_invalidates_wave(self):
        rows=self.rows(); rows[1]['error']='HTTP 503'
        self.assertFalse(bench.summarize(rows)['valid'])

    def test_usage_only_frame_is_not_a_token(self):
        rows=self.rows(); rows[0]['events']=[[10,300]]
        self.assertFalse(bench.summarize(rows)['valid'])

    def test_declared_lazy_counter_starts_at_zero(self):
        self.assertEqual(bench.retract_counter('# TYPE sglang:num_retracted_requests_total counter'),0)
        self.assertIsNone(bench.retract_counter(''))

    def test_retraction_counter_is_not_a_gauge_or_creation_timestamp(self):
        self.assertEqual(bench.retract_counter('sglang:num_retracted_reqs 9\nsglang:num_retracted_requests_total{rank="0"} 2\nsglang:num_retracted_requests_created 999'),2)


if __name__=='__main__': unittest.main()
