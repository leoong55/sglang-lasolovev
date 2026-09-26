"""Diagnostic of Humming graph/eager drift; not a replacement acceptance gate.

Uses unchanged gates from check_w4_humming_gpu.py: graph/eager rtol=.001,
atol=.002; Humming vs independent signed-INT4 FP32 reference relative L2<=.08.
Also measures stricter cross-layout parity without silently relaxing it.
No attention/KV/full-model correctness or performance claim.
"""
import gc,json,os,pathlib,time
import torch
import torch.distributed as dist
from types import SimpleNamespace
from check_w4_humming_gpu import load_layer,reference_local,relative_error

rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
torch.backends.cuda.matmul.allow_tf32=False
model_path=pathlib.Path(os.environ['MODEL_PATH']);out=pathlib.Path(os.environ['LAYOUT_GATE_OUT']);out.mkdir(parents=True,exist_ok=True)
config=json.loads((model_path/'config.json').read_text());mapping=json.loads(pathlib.Path('/scripts/expert-layout.json').read_text())['physical_to_logical_map']
assert config['hidden_size']==6144 and config['moe_intermediate_size']==2048 and config['n_routed_experts']==256
from sglang.srt.server_args import ServerArgs
from sglang.srt.runtime_context import publish
from sglang.srt.distributed import init_distributed_environment,initialize_model_parallel
server=ServerArgs(model_path=str(model_path),trust_remote_code=True,tp_size=8,ep_size=8,moe_a2a_backend='none',moe_runner_backend='humming',quantization='w4afp8',dtype='bfloat16',disable_shared_experts_fusion=True,ep_dispatch_algorithm='dynamic')
publish(server,role='test');init_distributed_environment(world_size=8,rank=rank,local_rank=rank,timeout=600);initialize_model_parallel(tensor_model_parallel_size=8,expert_model_parallel_size=8)
from sglang.srt.eplb.expert_location import ExpertLocationMetadata,set_global_expert_location_metadata,_compute_logical_to_all_physical_map
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe.topk import StandardTopKOutput,_biased_grouped_topk_postprocess
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config
from sglang.srt.layers.quantization.w4afp8_humming import W4AFp8HummingMoEMethod
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
quant=W4AFp8Config.from_config(config['quantization_config']);factor=float(config.get('routed_scaling_factor',1));rows=[];started=time.time()
def metadata(physical):
 m=torch.tensor(physical,device='cuda',dtype=torch.int64)
 inv=_compute_logical_to_all_physical_map(m,256,8,rank)
 return ExpertLocationMetadata._init_raw(8,m,inv,rank)
metas={'identity':metadata([list(range(256)) for _ in mapping]),'fixed':metadata(mapping)}

import copy
from humming.config import GemmType

def comparison(a,b):
    af=a.float();bf=b.float();finite=bool(torch.isfinite(af).all() and torch.isfinite(bf).all())
    if not finite:return {'finite':False,'strict_close':False}
    mismatch=(af-bf).abs()>(.002+.001*bf.abs())
    return {'finite':True,'strict_close':not bool(mismatch.any()),'mismatch_fraction':float(mismatch.float().mean()),'max_abs':float((af-bf).abs().max()),'relative_l2':relative_error(af,bf)}

for layer_id in [3]:

 layers={};packed={};scales={}
 for name in ['identity','fixed']:
  meta=metas[name];set_global_expert_location_metadata(meta,allow_overwrite=True)
  with torch.device('cuda'):
   layer=FusedMoE(num_experts=256,hidden_size=6144,intermediate_size=2048,layer_id=layer_id,top_k=8,params_dtype=torch.bfloat16,quant_config=quant,quant_method=W4AFp8HummingMoEMethod(quant),inplace=False,gate_up_interleaved=False,routed_scaling_factor=factor,prefix=f'model.layers.{layer_id}.mlp.experts')
  layer._probe_logical_experts=meta.physical_to_logical_map_cpu[layer_id,rank*32:(rank+1)*32].tolist()
  load_layer(layer,model_path,layer_id,rank,False)
  packed[name]=tuple(getattr(layer,k+'_weight').detach().clone() for k in ['w13','w2'])
  scales[name]=tuple(getattr(layer,k+'_weight_scale_inv').detach().clone() for k in ['w13','w2'])
  layer.quant_method.process_weights_after_loading(layer);layers[name]=layer
 if rank==0:print(json.dumps({'stage':'layer_loaded','layer':layer_id}),flush=True)
 for n in [2048,16384]:
  x=torch.empty((n,6144),device='cuda',dtype=torch.bfloat16);ids=torch.empty((n,8),device='cuda',dtype=torch.int32);weights=torch.empty((n,8),device='cuda',dtype=torch.float32);valid=torch.tensor(n,device='cuda',dtype=torch.int32)
  def fill(seed,concentrated=False):
   torch.manual_seed(seed);x.normal_();chosen=torch.randn((n,256),device='cuda').topk(8,dim=-1).indices.int()
   if concentrated:chosen=torch.arange(8,device='cuda',dtype=torch.int32).expand(n,8)
   ids.copy_(chosen);weights.copy_(torch.softmax(torch.randn((n,8),device='cuda'),-1))
  infos={}
  for name,meta in metas.items():
   set_global_expert_location_metadata(meta,allow_overwrite=True);infos[name]=ExpertLocationDispatchInfo.init_new(layer_id)
  def forward(name):
   physical=_biased_grouped_topk_postprocess(ids,infos[name],valid);layer=layers[name]
   dispatch=layer.dispatcher.dispatch(x,StandardTopKOutput(weights,physical,None))
   return layer.dispatcher.combine(layer.quant_method.apply(layer,dispatch))
  for concentrated in [False,True]:
   seed=41 if concentrated else 19
   for name,layer in layers.items():
    fill(seed,concentrated);forward(name);torch.cuda.synchronize()
    core=layer.quant_method.runner.runner_core
    original_prepare=core._prepare_indexed_gemm_kwargs
    original_configs=copy.deepcopy(core.get_humming_gemm_configs(GemmType.INDEXED))
    logicals=metas[name].physical_to_logical_map_cpu[layer_id].tolist();inverse=torch.tensor([logicals.index(i) for i in range(256)],device='cuda',dtype=torch.int32)
    physical=inverse[ids.long()];local=torch.where((physical>=rank*32)&(physical<(rank+1)*32),physical-rank*32,-1)
    ref=reference_local(x,local,weights,packed[name],scales[name],factor)
    for mode in ['native_changed_routes','native_same_routes','frozen_sort','streamk_off']:
     core._prepare_indexed_gemm_kwargs=original_prepare
     configs=copy.deepcopy(original_configs)
     if mode=='streamk_off':
      for key in ['w13_tuning_config','w2_tuning_config']:
       for low,high,cfg in configs[key]:cfg['use_stream_k']=False
       configs[key+'_str']=json.dumps(configs[key])
     core.humming_gemm_configs[GemTypeKey:=GemmType.INDEXED.value]=configs
     fill(seed,concentrated)
     kw1=original_prepare(local)[0];kw2=original_prepare(local)[0]
     torch.cuda.synchronize();padded=int(kw1['num_tokens_padded']);sort_equal=bool(torch.equal(kw1['sorted_ids'][:padded],kw2['sorted_ids'][:padded]))
     if mode=='frozen_sort':
      frozen=original_prepare(local)
      core._prepare_indexed_gemm_kwargs=lambda _,frozen=frozen:frozen
     if mode in ['native_changed_routes','streamk_off']:fill(7,False)
     for _ in range(3):forward(name)
     torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
     with model_capture_mode(),torch.cuda.graph(g):graph_out=forward(name)
     fill(seed,concentrated)
     eager1=forward(name).clone();eager2=forward(name).clone();g.replay();graph1=graph_out.clone();g.replay();graph2=graph_out.clone();torch.cuda.synchronize()
     event_start=torch.cuda.Event(enable_timing=True);event_end=torch.cuda.Event(enable_timing=True)
     event_start.record()
     for _ in range(10):g.replay()
     event_end.record();event_end.synchronize()
     row={'rank':rank,'layer':layer_id,'rows':n,'seed':seed,'concentrated':concentrated,'layout':name,'mode':mode,'local_assignments':int((local>=0).sum()),'sort_prefix_equal_two_calls':sort_equal,'num_tokens_padded':padded,'eager_eager':comparison(eager2,eager1),'graph_graph':comparison(graph2,graph1),'graph_eager':comparison(graph1,eager1),'eager_reference_l2':relative_error(eager1,ref),'graph_reference_l2':relative_error(graph1,ref),'graph_mean_ms':event_start.elapsed_time(event_end)/10,'configs':configs}
     rows.append(row)
     (out/f'rank{rank}.json').write_text(json.dumps(rows,indent=2))
     if rank==0:print(json.dumps({k:v for k,v in row.items() if k!='configs'}),flush=True)
     del g,graph_out,eager1,eager2,graph1,graph2,kw1,kw2
    core._prepare_indexed_gemm_kwargs=original_prepare;core.humming_gemm_configs[GemTypeKey]=original_configs
    del ref
all_rows=[None]*8;dist.all_gather_object(all_rows,rows)
if rank==0:
 assert all(len(rr)==32 for rr in all_rows)
 summary={'status':'diagnostic_completed','runtime_source':'0b3735e42','ranks_count':8,'cases_per_rank':32,'ranks':all_rows,'elapsed_s':time.time()-started,'full_model_numerical_parity':False,'acceptance_gate_passed':False,'thresholds_unchanged':{'rtol':.001,'atol':.002,'fp32_reference_relative_l2':.08},'scope':'One real W4 MoE layer, EP8; changed/same capture routes, frozen sort and streamK-off only in diagnostic. Not serving or acceptance.'}
 (out/'complete.json').write_text(json.dumps(summary,indent=2));print('HUMMING_DRIFT_DIAGNOSTIC_COMPLETE',flush=True)
dist.destroy_process_group()
