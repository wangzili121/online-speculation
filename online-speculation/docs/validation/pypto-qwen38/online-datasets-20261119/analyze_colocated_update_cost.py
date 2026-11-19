"""拆分同步在线流的显式额外开销；不把减去开销的数值当成实测吞吐。"""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]/'release-repo/online-speculation/docs/validation/pypto-qwen38/online-datasets-20261119'
rows=[]
for name in ['v5-gsm128-prewarm-tv1e5-dense7-r1-retry2','v8-pseudo56-arcopy-tv1e5-dense31-r1']:
 data=json.loads((root/(name+'.json')).read_text());assert data['learner_device']==0 and not data['background']
 frozen=data['results']['frozen']['stream']
 for key,result in data['results'].items():
  if not result.get('updates'):continue
  stream=result['stream'];stats=stream['stats']
  # update_ns 已包含读取和 CPU pack，不能再加一次完整 publish_ns。
  measured_extra=sum(stats.get(k,0) for k in ['feedback_ns','update_ns','publish_upload_ns'])/1e9
  rows.append(dict(trial=name,interval=key,frozen_tok_s=frozen['tok_s'],online_tok_s=stream['tok_s'],net=result['net_increment'],wall_seconds=stream['seconds'],feedback_seconds=stats['feedback_ns']/1e9,update_seconds=stats['update_ns']/1e9,publication_cpu_pack_seconds=stats['publish_pack_ns']/1e9,upload_seconds=stats['publish_upload_ns']/1e9,explicit_synchronous_extra_seconds=measured_extra,counterfactual_excluding_explicit_extra_tok_s=stream['tokens']/(stream['seconds']-measured_extra),warning='cost subtraction is a diagnostic bound, not a measured alternative; excludes implicit feedback effects on serving kernels'))
(root/'colocated-cost-breakdown.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
