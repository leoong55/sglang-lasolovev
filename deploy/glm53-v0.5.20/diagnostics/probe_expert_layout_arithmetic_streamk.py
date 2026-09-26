"""EP8 fixed-layout arithmetic gate on actual W4 checkpoint layers.

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
for layer_id in [3,41,77]:
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
 for n in [1,40,2048,16384]:
  x=torch.empty((n,6144),device='cuda',dtype=torch.bfloat16);ids=torch.empty((n,8),device='cuda',dtype=torch.int32);weights=torch.empty((n,8),device='cuda',dtype=torch.float32);valid=torch.tensor(n,device='cuda',dtype=torch.int32)
  def fill(seed,concentrated=False,padded=False):
   torch.manual_seed(seed);x.normal_();chosen=torch.randn((n,256),device='cuda').topk(8,dim=-1).indices.int()
   if concentrated:chosen=torch.arange(8,device='cuda',dtype=torch.int32).expand(n,8)
   ids.copy_(chosen);weights.copy_(torch.softmax(torch.randn((n,8),device='cuda'),-1));valid.fill_(max(0,n-3) if padded else n)
  infos={}
  for name,meta in metas.items():
   set_global_expert_location_metadata(meta,allow_overwrite=True);infos[name]=ExpertLocationDispatchInfo.init_new(layer_id)
  def forward(name):
   physical=_biased_grouped_topk_postprocess(ids,infos[name],valid);layer=layers[name];dispatch=layer.dispatcher.dispatch(x,StandardTopKOutput(weights,physical,None));return layer.dispatcher.combine(layer.quant_method.apply(layer,dispatch))
  fill(7);graphs={};outputs={}
  for name in layers:
   for _ in range(3):forward(name)
   torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with model_capture_mode(),torch.cuda.graph(g):outputs[name]=forward(name)
   graphs[name]=g
  for seed,concentrated,padded in [(19,False,False),(41,True,False),(73,False,True)]:
   fill(seed,concentrated,padded);result={'layer':layer_id,'rows':n,'seed':seed,'concentrated':concentrated,'valid_rows':int(valid),'layouts':{}};summed={};refs={};bf16={}
   for name,layer in layers.items():
    # Independently map logical IDs through inverse of the candidate permutation.
    logicals=metas[name].physical_to_logical_map_cpu[layer_id].tolist();inverse=torch.tensor([logicals.index(i) for i in range(256)],device='cuda',dtype=torch.int32);physical=inverse[ids.long()];physical[int(valid):]=-1
    local=torch.where((physical>=rank*32)&(physical<(rank+1)*32),physical-rank*32,-1)
    ref=reference_local(x,local,weights,packed[name],scales[name],factor)
    eager=forward(name).clone();graphs[name].replay();torch.cuda.synchronize();actual=outputs[name].clone()
    graph_error=None
    try:torch.testing.assert_close(actual,eager,rtol=.001,atol=.002)
    except AssertionError as exc:graph_error=str(exc)
    flags=[None]*8;dist.all_gather_object(flags,graph_error)
    if any(v is not None for v in flags):
     (out/f'failure-rank{rank}.json').write_text(json.dumps({'rank':rank,'layer':layer_id,'rows':n,'seed':seed,'layout':name,'check':'graph_eager','rtol':.001,'atol':.002,'errors':flags},indent=2))
     raise AssertionError(('collective graph/eager gate failed',layer_id,n,seed,name))
    err=relative_error(actual,ref);assert err<=.08,(rank,layer_id,n,seed,int(valid),name,err)
    fp=actual.float();dist.all_reduce(fp);dist.all_reduce(ref);ep_error=relative_error(fp,ref);assert ep_error<=.08,(rank,layer_id,n,name,'EP8 reference',ep_error)
    low=actual.clone();dist.all_reduce(low)
    summed[name]=fp;refs[name]=ref;bf16[name]=low
    result['layouts'][name]={'local_relative_l2':err,'ep8_relative_l2':ep_error,'graph_eager_close':True,'local_assignments':int((local>=0).sum())}
   torch.testing.assert_close(refs['identity'],refs['fixed'],rtol=.001,atol=.002)
   result['cross_layout_fp32_sum_relative_l2']=relative_error(summed['fixed'],summed['identity']);result['cross_layout_bf16_sum_relative_l2']=relative_error(bf16['fixed'],bf16['identity'])
   try:torch.testing.assert_close(summed['fixed'],summed['identity'],rtol=.001,atol=.002);strict=True
   except AssertionError:strict=False
   result['cross_layout_strict_close']=strict;rows.append(result)
   (out/f'cases-rank{rank}.json').write_text(json.dumps(rows,indent=2))
   if rank==0:(out/'progress.json').write_text(json.dumps({'completed_cases_per_rank':len(rows),'last':result},indent=2));print(json.dumps({'stage':'case_passed_reference','layer':layer_id,'n':n,'seed':seed,'strict_cross_layout':strict}),flush=True)
  del graphs,outputs,g,eager,actual,fp,low,ref,refs,summed,bf16
 del layers,layer,packed,scales;gc.collect();torch.cuda.empty_cache()
all_rows=[None]*8;dist.all_gather_object(all_rows,rows)
if rank==0:
 assert all(len(x)==36 for x in all_rows)
 (out/'complete.json').write_text(json.dumps({'status':'component_reference_gates_passed','ranks':all_rows,'unique_cases':36,'cases_per_rank':36,'ranks_count':8,'elapsed_s':time.time()-started,'gates':{'graph_eager_rtol':.001,'graph_eager_atol':.002,'relative_l2_fp32_reference':.08,'source':'unchanged deploy/glm53-hicache-v9/benchmarks/check_w4_humming_gpu.py'},'all_cross_layout_strict_close':all(x['cross_layout_strict_close'] for r in all_rows for x in r),'full_model_numerical_parity':False,'scope':'Real W4 weights3layers,EP8 localMoE+allreduce,independentFP32reference; noattention/KV/end-to-endquality'},indent=2))
 print('LAYOUT_ARITHMETIC_COMPONENT_COMPLETE',flush=True)
dist.destroy_process_group()
