"""Real host-pool methods and DMA ordering with CPU transfer substitutes.

The substitutes copy bytes and record events; they do not validate CUDA DMA.
"""
import ast
import logging
import os
import threading
import types
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(os.environ["SGLANG_SOURCE_ROOT"]) / "python/sglang/srt"


def extract(path, names, namespace, cls=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body if cls is None else next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    selected = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(selected) == len(names), names
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])), path, "exec"), namespace)
    return NS(**{name: namespace[name] for name in names})


def direct_copy(*, src_layers, dst_layers, src_indices, dst_indices, page_size):
    # Actual direct backend consumes token rows (MLA) or packed page rows (DSA).
    for src, dst in zip(src_layers, dst_layers, strict=True):
        dst[dst_indices.long()] = src[src_indices.long()]


class PoolTests(unittest.TestCase):
    def base(self):
        ns = {"torch": torch}
        funcs = extract("mem_cache/pool_host/base.py", {"maybe_dcp_kernel_indices", "logical_size", "logical_page_size"}, ns, "HostKVCache")
        return type("Base", (), {
            **vars(funcs), "dcp_size": 1, "dcp_rank": 0,
            "_effective_host_layer_num": lambda s: s.device_pool.layer_num,
            "_is_device_layer_sharded": lambda s,p: False,
            "_is_device_layer_owned": lambda s,p,l: True,
            "_host_layer_index": lambda s,l: l,
            "clear": lambda s: None,
        })

    def test_dsa_tail_capacity_and_relocated_round_trip(self):
        base = self.base()
        ns = dict(torch=torch, HostKVCache=base, threading=threading,
                  logger=logging.getLogger(__name__), _is_cuda=False, _is_hip=False,
                  _is_npu=False, _is_xpu=False, _is_mps=False,
                  get_allocator_from_storage=lambda _: None,
                  host_memory_budget_bytes=lambda: 10**9,
                  DSATokenToKVPool=NS(index_k_with_scale_buffer_dtype=torch.uint8),
                  ALLOC_MEMORY_FUNCS={"cpu":lambda shape,**kw:torch.empty(shape,dtype=kw["dtype"])},
                  transfer_kv_direct=direct_copy)
        cls = extract("mem_cache/pool_host/dsa.py", {"DSAIndexerPoolHost"}, ns).DSAIndexerPoolHost
        device = NS(page_size=64, store_dtype=torch.uint8, start_layer=0, end_layer=2,
                    layer_num=2, index_head_dim=128, quant_block_size=128, device="cpu")
        stride = 64 * (128 + 4)
        device.index_k_with_scale_buffer = [torch.randint(0,256,(40,stride),dtype=torch.uint8) for _ in range(2)]
        anchor = NS(page_size=64, size=256, logical_size=1024, page_num=4, mtp_draft_device_pools=())
        host = cls(device, anchor, "layer_first", pin_memory=False)
        self.assertEqual((host.size, host.page_size, host.logical_size), (1024,64,1024))
        old = torch.arange(512,768)
        host_ids = torch.arange(768,1024)  # Beyond the old rank-local host capacity.
        new = torch.arange(1536,1792)
        expected = [x[8:12].clone() for x in device.index_k_with_scale_buffer]
        host.backup_from_device_all_layer(device,host_ids,old,"direct")
        for x in device.index_k_with_scale_buffer: x.zero_()
        for layer in range(2): host.load_to_device_per_layer(device,host_ids,new,layer,"direct")
        for x,want in zip(device.index_k_with_scale_buffer,expected):
            torch.testing.assert_close(x[24:28],want)
        with self.assertRaisesRegex(ValueError,"page-aligned"):
            host._get_indexer_page_indices(host_ids[:-1],new[:-1])

    def test_mla_packed_bytes_round_trip_all_dcp_owners(self):
        names={"backup_from_device_all_layer","load_to_device_per_layer","_resolve_device_transfer_buffers"}
        f=extract("mem_cache/pool_host/mla.py",names,dict(torch=torch,transfer_kv_direct=direct_copy),"MLATokenToKVPoolHost")
        cls=type("MLA",(self.base(),),vars(f))
        old=torch.cat([torch.arange(256,512),torch.arange(768,1024)])
        new=torch.cat([torch.arange(1280,1536),torch.arange(512,768)])
        host_ids=torch.cat([torch.arange(0,256),torch.arange(768,1024)])
        for rank in range(4):
            host=cls();host.dcp_size=4;host.dcp_rank=rank
            host.layout="layer_first";host.page_size=64;host.mtp_draft_device_pools=()
            host.kv_buffer=torch.zeros((2,256,1,80),dtype=torch.uint8)
            host.data_refs=[host.kv_buffer[i] for i in range(2)];host.data_ptrs=None
            device=NS(kv_buffer=[torch.randint(0,256,(512,1,80),dtype=torch.uint8) for _ in range(2)],data_ptrs=None)
            host.device_pool=device
            want=[x[old[rank::4]//4].clone() for x in device.kv_buffer]
            host.backup_from_device_all_layer(device,host_ids,old,"direct")
            for x in device.kv_buffer:x.zero_()
            for layer in range(2):host.load_to_device_per_layer(device,host_ids,new,layer,"direct")
            for x,expected in zip(device.kv_buffer,want):
                torch.testing.assert_close(x[new[rank::4]//4],expected)

    def test_draft_uses_full_logical_capacity_and_shared_indices(self):
        fake_dsa=type("DSA",(),{});fake_hybrid=type("Hybrid",(),{})
        pool=NS(size=2048,layer_num=2)
        captured={}
        def build(**kw):
            captured.update(kw)
            return NS(size=int(kw["pool"].size*kw["host_to_device_ratio"]),layer_num=2)
        ns=dict(_build_mha_mla_host_pool=build,get_memory=lambda:NS(hicache_mem_layout="layer_first"),
                _get_allocator_type=lambda:"default",SidecarPoolSpec=lambda **kw:NS(**kw),
                PoolName=NS(KV="kv",DRAFT="draft"),build_pool_entry=lambda **kw:NS(**kw))
        f=extract("mem_cache/hybrid_cache/hybrid_pool_assembler.py",{"build_full_draft_pools"},ns)
        modules={"sglang.srt.mem_cache.memory_pool":NS(DSATokenToKVPool=fake_dsa,HybridLinearKVPool=fake_hybrid)}
        with patch.dict("sys.modules",modules):
            specs,entries=f.build_full_draft_pools(draft_kv_pool=pool,tree_cache=NS(cache_controller=NS(page_size=256,mem_pool_host=NS(logical_size=4096,size=1024))))
        self.assertGreaterEqual(entries[0].host_pool.size,4096)
        self.assertEqual(specs[0].indices_from_pool,"kv")
        self.assertEqual(entries[0].layer_mapping,{0:0,1:1})
        # Derived pools share the primary indices without allocating or freeing
        # them again, and without collapsing virtual IDs by DCP.
        g=extract("mem_cache/pool_host/group.py",{"resolve_host_transfers","release_transfers"},dict(replace=replace),"HostPoolGroup")
        from dataclasses import make_dataclass
        Transfer=make_dataclass("Transfer",[("name",str),("indices_from_pool",str),("host_indices",object,None),("device_indices",object,None)])
        ids=torch.arange(3072,3328);dev=torch.arange(1280,1536)
        owner=NS(anchor_entry=NS(name="kv"),free=lambda *_a,**_kw:self.fail("derived pool freed twice"))
        transfers=g.resolve_host_transfers(owner,[Transfer("draft","kv"),Transfer("indexer","kv")],primary_device_indices=dev,primary_host_indices=ids)
        for t in transfers:
            self.assertIs(t.host_indices,ids);self.assertIs(t.device_indices,dev)
        self.assertEqual(g.release_transfers(owner,transfers),0)


class CompletionTests(unittest.TestCase):
    def test_transfer_completion_follows_every_pool_and_layer(self):
        events=[]
        class Event:
            def __init__(self,label):self.label=label
            def record(self):events.append(self.label)
            def wait(self,*args):events.append("wait-producer")
        device=NS(Event=lambda:Event("producer"),stream=lambda _:nullcontext())
        funcs=extract("mem_cache/l2_transfer.py",{"L2TransferEngine"},dict(
            device_module=device,make_timing_event_pair=lambda:(Event("start"),Event("finish"),False),
            TransferCompletion=lambda *a:NS(start_event=a[0],finish_event=a[1],timing_enabled=a[2])))
        engine=object.__new__(funcs.L2TransferEngine);engine.io_backend="direct"
        engine.device_to_host_stream=engine.host_to_device_stream=None
        transfers=[]
        for name,layers in (("kv",3),("indexer",3),("draft",2)):
            host=NS(layer_num=layers,
                    backup_from_device_all_layer=lambda *a,n=name:events.append("backup-"+n),
                    load_to_device_per_layer=lambda *a,n=name,**kw:events.append(f"load-{n}-{a[3]}"))
            transfers.append(NS(host_pool=host,device_pool=None,host_indices=torch.arange(4),device_indices=torch.arange(4),layer_mapper=lambda l,n=layers:l if l<n else None,is_draft=False))
        engine.submit_device_to_host(transfers)
        self.assertEqual(events,["producer","wait-producer","start","backup-kv","backup-indexer","backup-draft","finish"])
        events.clear()
        engine.submit_host_to_device(transfers,layer_num=3,on_layer_done=lambda l:events.append(f"ready-{l}"))
        for layer in range(3):
            for name in ("kv","indexer",*( ["draft"] if layer<2 else [])):
                self.assertLess(events.index(f"load-{name}-{layer}"),events.index(f"ready-{layer}"))
        self.assertEqual(events[-1],"finish")

    def test_write_submission_fences_draft_producer_before_dma(self):
        events=[]
        op=NS(host_indices=torch.arange(4),device_indices=torch.arange(4),node_ids=[7])
        engine=NS(device_to_host_stream=NS(wait_stream=lambda s:events.append(("fence",s))),
                  submit_device_to_host=lambda x:(events.append("dma") or NS(start_event=None,finish_event=None,timing_enabled=False)))
        f=extract("managers/cache_controller.py",{"start_writing"},dict(CacheOperation=NS(merge_ops=lambda q:op),HiCacheAck=lambda **kw:NS(**kw)),"HiCacheController")
        owner=NS(write_queue=[op],write_fence_stream="forward",l2_transfer_engine=engine,ack_write_queue=[],
                 _move_write_operation=lambda op:(op.host_indices,op.device_indices,None),
                 _l2_transfers=lambda *a:[],_num_tokens_by_pool=lambda op:{},_transfer_num_bytes=lambda op:0)
        f.start_writing(owner)
        self.assertEqual(events,[("fence","forward"),"dma"])
        self.assertEqual(len(owner.ack_write_queue),1)

    def test_write_through_split_and_empty_rank_completion(self):
        ns={"torch":torch,"_OngoingWriteThrough":lambda *a:a}
        names={"_track_write_through_node","_replace_pending_write_through_node","_finish_write_through_ack","writing_check"}
        f=extract("mem_cache/unified_radix_cache.py",names,ns,"UnifiedRadixCache")
        cls=type("Cache",(),vars(f));cache=cls();events=[]
        cache.buffer_pipeline=None;cache.enable_storage=False;cache.pp_rank=0
        cache.ongoing_write_through={};cache.cache_controller=NS(ack_write_queue=[])
        cache.tree_core=NS(mark_write_through_pending=lambda n:events.append("pending"),
                           finish_write_through=lambda nodes,ack:events.append(("publish",nodes,ack)))
        cache.dec_lock_ref=lambda n,params:events.append(("unlock",n))
        cache._all_reduce=lambda *a:events.append("collective")
        cache._count_ready_acks=lambda q:len(q)
        cache._log_write_ack_metrics=lambda a:None
        cache.writing_check()
        self.assertEqual(events,["collective"]) # Empty local queue still participates.
        events.clear();cache._track_write_through_node(9,object())
        cache._replace_pending_write_through_node(9,9,[8,9])
        self.assertNotIn(("unlock",9),events)
        cache.cache_controller.ack_write_queue.append(NS(node_ids=[9],finish_event=NS(synchronize=lambda:events.append("dma-complete"))))
        cache.writing_check()
        self.assertEqual(events,["pending","collective","dma-complete",("publish",[8,9],9),("unlock",9)])
        self.assertFalse(cache.ongoing_write_through)


if __name__ == "__main__":
    unittest.main()
