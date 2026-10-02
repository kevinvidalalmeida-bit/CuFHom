"""Exclusive CUDA timeline accounting; API wait time is not transfer time."""
import argparse,csv,json,sqlite3
import statistics
from pathlib import Path
from collections import defaultdict,Counter

def summarize(path):
    c=sqlite3.connect(path);c.row_factory=sqlite3.Row
    strings={r['id']:r['value'] for r in c.execute('select * from StringIds')}
    ranges=[dict(r) for r in c.execute('select * from NVTX_EVENTS where end is not null')]
    total=next(r for r in ranges if r['text']=='TOTAL');start,end=total['start'],total['end']
    inner=[r for r in ranges if r['text']=='inner_cg']
    def inside(t):return any(r['start']<=t<=r['end'] for r in inner)
    def phase(t):
        for name in ('postprocess','setup_material'):
            if any(r['text']==name and r['start']<=t<=r['end'] for r in ranges):return name
        return None
    runtime={r['correlationId']:dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')}
    events=[];kernels=defaultdict(list);apis=defaultdict(list);copies=defaultdict(list)
    def add(s,e,category,priority):
        s=max(s,start);e=min(e,end)
        if e>s:events.append((s,1,category,priority));events.append((e,-1,category,priority))
    def group(name,t):
        if phase(t):return phase(t)+'_kernels'
        if 'fft' in name:return 'operator_fft'
        if name in ('coordinates','adjoint_dot','indexed_sym21_apply','elasticity_direct_apply'):return 'operator_other'
        if name in ('update_rr','linear_cg_finish_smoothing','warp_update'):return 'solver_update'
        if 'Reduce' in name or name=='warp_dot':return 'reductions'
        if inside(t):return 'solver_elementwise'
        return 'outside_cg_kernels'
    for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
        r=dict(r);name=strings[r['demangledName']];api=runtime.get(r['correlationId'],{})
        g=group(name,api.get('start',r['start']));r.update(name=name,group=g)
        kernels[name].append(r);add(r['start'],r['end'],g,3)
    kinds={r['id']:r['label'] for r in c.execute('select * from ENUM_CUDA_MEMCPY_OPER')}
    for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY'):
        r=dict(r);k=kinds[r['copyKind']];copies[k].append(r)
        add(r['start'],r['end'],'transfers' if r['copyKind'] in (1,2) else 'memory_device_copy',4)
    for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMSET'):
        r=dict(r);copies['memset'].append(r);add(r['start'],r['end'],'memory_memset',4)
    for r in runtime.values():
        name=strings[r['nameId']];apis[name].append(r)
        if 'Synchronize' in name:g='idle_in_sync_api'
        elif 'Memcpy' in name:g='idle_in_memcpy_api'
        elif 'Launch' in name:g='idle_in_launch_api'
        elif 'Malloc' in name or 'Free' in name:g='idle_in_alloc_api'
        else:g='idle_in_other_api'
        add(r['start'],r['end'],g,1)
    # Include range boundaries so uncovered host time can be located relative to CG.
    events+=[(start,0,'none',0),(end,0,'none',0)]
    for r in inner:events.extend([(r['start'],0,'none',0),(r['end'],0,'none',0)])
    events.sort(key=lambda e:e[0]);active=Counter();exclusive=defaultdict(int);prev=start
    for t,delta,g,priority in events:
        if t>prev:
            if active:
                category=max((k for k,v in active.items() if v>0),key=lambda k:k[1])[0]
            else:
                mid=(t+prev)//2
                category='idle_host_'+phase(mid) if phase(mid) else ('idle_host_in_cg' if inside(mid) else 'idle_host_outside_cg')
            exclusive[category]+=t-prev
        if delta:
            active[(g,priority)]+=delta
            if active[(g,priority)]==0:del active[(g,priority)]
        prev=t
    assert sum(exclusive.values())==end-start
    rows=[]
    for name,runs in kernels.items():
        duration=sum(r['end']-r['start'] for r in runs)
        rows.append(dict(kernel=name,calls=len(runs),total_ms=duration/1e6,mean_us=duration/len(runs)/1e3,
            fraction_wall=duration/(end-start),group=runs[0]['group'],
            registers_per_thread_min=min(r['registersPerThread'] for r in runs),
            registers_per_thread_max=max(r['registersPerThread'] for r in runs),
            local_memory_per_thread_max=max(r['localMemoryPerThread'] for r in runs),
            shared_memory_bytes_max=max(r['staticSharedMemory']+r['dynamicSharedMemory'] for r in runs),
            grids=sorted(set((r['gridX'],r['gridY'],r['gridZ']) for r in runs)),
            blocks=sorted(set((r['blockX'],r['blockY'],r['blockZ']) for r in runs))))
        variants=defaultdict(list)
        for r in runs:variants[(r['registersPerThread'],r['group'])].append((r['end']-r['start'])/1e3)
        rows[-1]['variants']=[dict(registers=k[0],group=k[1],calls=len(v),mean_us=statistics.mean(v),
            median_us=statistics.median(v),p10_us=sorted(v)[int(.1*(len(v)-1))],min_us=min(v)) for k,v in variants.items()]
    rows.sort(key=lambda r:-r['total_ms'])
    api_summary={name:dict(calls=len(r),inclusive_ms=sum(x['end']-x['start'] for x in r)/1e6,
        inside_cg_calls=sum(inside(x['start']) for x in r)) for name,r in apis.items()}
    copy_summary={name:dict(calls=len(r),bytes=sum(x['bytes'] for x in r),
        device_ms=sum(x['end']-x['start'] for x in r)/1e6,
        inside_cg_calls=sum(inside(runtime.get(x.get('correlationId'),{}).get('start',x['start'])) for x in r)) for name,r in copies.items()}
    return dict(wall_ms=(end-start)/1e6,exclusive_ms={k:v/1e6 for k,v in exclusive.items()},
        kernels=rows,api=api_summary,transfers=copy_summary,
        scope='NVTX TOTAL range, separate profiled run. API inclusive times overlap GPU and must not be added. '
              'Exclusive timeline gives GPU activity precedence, then API occupied idle time, then host idle. '
              'No attribution of all outside-CG kernels to postprocess is attempted.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=['G08','G09','N64','N256']);a=p.parse_args()
    reports={name:summarize(a.input/f'trace_{name}.sqlite') for name in a.cases}
    a.output.write_text(json.dumps(reports,indent=2)+'\n')
    for name,r in reports.items():print(name,r['wall_ms'],r['exclusive_ms'],flush=True)
if __name__=='__main__':main()
