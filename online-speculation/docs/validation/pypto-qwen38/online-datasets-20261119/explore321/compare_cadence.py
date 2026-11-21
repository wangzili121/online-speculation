"""对照实际源码、资产和配置后比较更新频率；不把旧题用作新题确认。"""
from pathlib import Path
import argparse,json,hashlib
p=argparse.ArgumentParser(description='只读比较同机完整旧题频率实验；不派发或重跑模型。');p.add_argument('--validation-root',type=Path,required=True);p.add_argument('--cohort',choices=['longproc30','spoc64'],default='longproc30');a=p.parse_args();D=a.validation_root;Q=D/'explore321'
specs=([('explore314','c61-v314-explore-longproc30-kl-constant16-output512','16'),('explore321','c61-v314-explore-longproc30-kl-staged64-v321-output512','64')] if a.cohort=='longproc30' else [('explore314','c61-v314-explore-spoc64-kl-constant16-output192','16'),('explore321','c61-v314-explore-spoc64-kl-staged64-v321-output192','64')])
data=[];source=[];proofs=[]
for folder,tag,stride in specs:
 p=D/folder/(tag+'.json');raw=p.read_bytes();d=json.loads(raw);check=json.loads((D/folder/(tag+'.audit.json')).read_text());assert check['completed']and check['raw_sha256']==hashlib.sha256(raw).hexdigest()
 assert d['completed']and d['clean_shutdown']and d['visible_devices']=='2,3';data.append(d);source.append(json.loads((D/folder/(tag+'.source.json')).read_text())['files']);proofs.append(check)
assert source[0]==source[1]
fields=['head_step','head_shard_sha256','optimizer_init','physical_device_count','device_ids','learner_device','background','last_layers','parameter_scope','parameterization','publication_backend','training_rows','max_model_len','kv_pages','learning_rate','objective','warmup_updates','lr_schedule','replay_blocks','publish_every','inner_steps','teacher_targets','teacher_temperature','teacher_tokens','teacher_scaling','feedback_mode','feedback_timing','temperature','sampler','stop_on_eos','candidates','draft_width','paired_physical_rows','gdn_valid_rows','draft_convolution','stream_window','execution_max_num_seqs','verify_chunk_size','seed','prompts_sha256','holdout_sha256','output_tokens_per_request','requests','ar_checked_requests','max_updates']
for key in fields:assert data[0][key]==data[1][key],key
# 裁剪输出路径额外记录逐题预算；固定192路径不生成这些字段。
optional=['output_budget_mode','request_prompt_tokens','request_output_budgets']
for key in optional:
 assert (key in data[0])==(key in data[1]),key
 if key in data[0]:
  assert data[0][key]==data[1][key],key
  fields.append(key)
if a.cohort=='spoc64':
 for d in data:assert d['output_tokens_per_request']==192
 for folder,tag,_ in specs:
  plan=json.loads((D/folder/('queue-explore314.json'if folder=='explore314'else'queue-explore321.json')).read_text())
  case=next(c for c in plan['cases']if c['tag']==tag)
  assert '--clip-output-to-context'not in case['args']

for d in data:assert d['head_step']==128 and d['optimizer_init']=='fresh'and d['max_updates']is None
results=[]
for d,proof,(_,tag,stride)in zip(data,proofs,specs):
 on=d['results'][stride];st=on['stream']['stats'];cost=(st['feedback_ns']+st['update_ns']+st['publish_upload_ns'])/1e9
 results.append(dict(tag=tag,raw_sha256=proof['raw_sha256'],ar_tok_s=d['results']['ar']['stream']['tok_s'],fixed_tok_s=d['results']['frozen']['stream']['tok_s'],online_tok_s=on['stream']['tok_s'],online_seconds=on['stream']['seconds'],online_vs_ar=on['net_increment_vs_ar'],online_vs_fixed=on['net_increment'],accepted_per_round=on['stream']['accepted_per_round'],rounds=st['rounds'],updates=st['updates'],publications=st['publications'],direct_cost_seconds=cost,direct_cost_fraction=cost/on['stream']['seconds'],final_seen_tok_s=on['learned_replay']['stream']['tok_s'],independent_old_holdout_tok_s=on['holdout']['tok_s']))
for mode in ['ar','frozen']:
 assert [x['output_tokens']for x in data[0]['results'][mode]['request_timeline']]==[x['output_tokens']for x in data[1]['results'][mode]['request_timeline']]
proof=dict(completed=True,same_compute_native_launcher_source=True,same_hyperparameters_checked=fields,same_physical_cards='C61 2,3',same_requests_and_per_request_control_lengths=True,only_strategy_change='constant16 versus first64updates@16 then64 continuing',cases=results,online_throughput_change=results[1]['online_tok_s']/results[0]['online_tok_s']-1,fixed_baseline_drift=results[1]['fixed_tok_s']/results[0]['fixed_tok_s']-1,ar_baseline_drift=results[1]['ar_tok_s']/results[0]['ar_tok_s']-1,update_transfer_cost_saved_seconds=results[0]['direct_cost_seconds']-results[1]['direct_cost_seconds'],scope='Old exploration cohort only. Same host, sequential cases with within-case interleaved controls; not randomized repeated cadence trial, no reserved qualification or retuning. Original LongProc negative confirmations unchanged.')
(Q/(a.cohort+'-cadence-comparison.json')).write_text(json.dumps(proof,indent=2)+'\n')

print(json.dumps(dict(completed=proof['completed'],cases=proof['cases']),indent=2))
