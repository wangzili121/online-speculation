"""只读核对真实更新事件、学习率、发布和计时；不派发或重跑模型。"""
from pathlib import Path
import argparse,gzip,hashlib,json,math


def check(condition, label):
    if not condition:
        raise ValueError(label)


def audit(report, stride):
    check(report.get('completed') and report.get('clean_shutdown'), 'incomplete suite')
    check(report['max_updates'] is None and report['inner_steps'] == 1, 'not uncapped one-step updates')
    check(report['optimizer_init'] == 'fresh', 'optimizer was not reset')
    online = report['results'][str(stride)]
    events, stats, schedule = online['updates'], online['stream']['stats'], online['update_schedule']
    check(events and stats['busy_updates'] == stats['feedback_skips'] == 0, 'missing scheduled update events')
    check(schedule['scheduled_updates'] == len(events) == stats['updates'], 'schedule/event count')
    check(schedule['stride'] == stride, 'steady stride')
    warm_count, warm_stride = report['warm_update_steps'], report['warm_update_stride']
    check(schedule['warm_update_steps'] == warm_count and schedule['warm_update_stride'] == warm_stride,
          'warm schedule metadata')
    previous_round = publications = 0
    width = report['stream_window']
    for number, event in enumerate(events, 1):
        interval = warm_stride if number <= warm_count else stride
        check(interval is not None and interval > 0, 'invalid interval')
        # finish_batch(active) 按累计验证行数跨过整倍数时更新，active为1或2。
        boundary = (previous_round // interval + 1) * interval
        check(boundary <= event['round'] < boundary + width, f'update {number}: skipped/early boundary')
        check(event['interval'] == interval and event['version'] == number, f'update {number}: interval/version')
        lr = report['learning_rate']
        warmup = max(1, report['warmup_updates'])
        if number <= warmup:
            lr *= number / warmup
        elif report['lr_schedule'] == 'inverse_sqrt':
            lr *= (warmup / number) ** .5
        else:
            check(report['lr_schedule'] == 'constant', 'unknown LR schedule')
        check(math.isclose(event['learning_rate'], lr, rel_tol=1e-12, abs_tol=1e-15), f'update {number}: LR')
        published = number % report['publish_every'] == 0
        check((event['published_at_round'] is not None) == published, f'update {number}: publication cadence')
        phases = ('publish_read_ns', 'publish_pack_ns', 'publish_upload_ns')
        check(event['publish_ns'] == sum(event[k] for k in phases), f'update {number}: publication subtotal')
        if published:
            publications += 1
            check(event['round'] <= event['published_at_round'] <= schedule['rounds'], 'publication boundary')
            check(event['publish_upload_ns'] > 0, 'missing upload')
        else:
            check(all(event[k] == 0 for k in phases), 'nonpublication performed transfer')
        # 参数读取与TP整理发生在train_scheduled内，已经包含在update_ns。
        check(event['update_ns'] >= event['publish_read_ns'] + event['publish_pack_ns'], 'nested update cost')
        check(0 < event['supervised'] <= report['teacher_tokens'] * event['feedback_blocks'], 'invalid supervision')
        check(len(event['feedback_age_rounds']) == event['feedback_blocks']
              and all(v >= 0 for v in event['feedback_age_rounds']), 'invalid feedback age')
        previous_round = event['round']
    check(publications == stats['publications'], 'publication count')
    check(online['optimizer_version'] == len(events), 'final optimizer version')
    published_version = publications * report['publish_every']
    check(online['published_version'] == published_version, 'final serving version')
    manifest = online['exported_head']['manifest']
    check(manifest['published_banks_exact'] and manifest['step'] == report['head_step'] + published_version,
          'export differs from published head')
    # 流末尾没有被省掉的下一次更新；不到间隔的剩余行不算停更。
    interval = warm_stride if len(events) < warm_count else stride
    next_boundary = (previous_round // interval + 1) * interval
    check(schedule['rounds'] < next_boundary, 'stream crossed another unexecuted update')
    for key in ('update_ns', 'publish_ns', 'publish_read_ns', 'publish_pack_ns', 'publish_upload_ns'):
        check(sum(v[key] for v in events) == stats[key], f'{key}: events versus full-stream counter')
    cost_ns = stats['feedback_ns'] + stats['update_ns'] + stats['publish_upload_ns']
    return dict(completed=True,actual_schedule_checked=True,updates=len(events),publications=publications,
                warmup_LR_checked=True,first_update_round=events[0]['round'],last_update_round=previous_round,
                final_optimizer_version=len(events),actual_published_version=published_version,
                final_exported_step=manifest['step'],tail_rounds=schedule['rounds']-previous_round,
                nonoverlap_feedback_update_upload_seconds=cost_ns/1e9,
                work_time_to_stream_wall_ratio=cost_ns/1e9/online['stream']['seconds'],
                colocated_direct_cost_fraction=(cost_ns/1e9/online['stream']['seconds'] if not report['background'] else None),
                read_pack_already_in_update=True,
                cost_scope='Colocated: nonoverlapping cost regions. Background learner: work duration may overlap generation; not a latency fraction.',
                background=report['background'],physical_cards=report['physical_device_count'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path)
    p.add_argument('--stride',type=int,required=True);p.add_argument('--output',type=Path)
    args=p.parse_args();raw=args.report.read_bytes()
    if args.report.suffix=='.gz':raw=gzip.decompress(raw)
    proof=audit(json.loads(raw),args.stride);proof['raw_sha256']=hashlib.sha256(raw).hexdigest()
    text=json.dumps(proof,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
