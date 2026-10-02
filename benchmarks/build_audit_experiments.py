"""Statistical and numerical summaries of isolated audit variants."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from build_performance_audit import write_csv,stats,clean_svg


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[];fields=[];micro=[]
    rng=np.random.default_rng(20261001)
    paths=list(a.input.glob('*/report.json'))+list((a.input/'earlier_pilots').glob('*/report.json'))
    for path in sorted(paths):
        r=json.loads(path.read_text());name=path.parent.name
        if 'median_s' in r:
            ref='auto' if 'auto' in r['median_s'] else 'current'
            if ref not in r['median_s']:continue
            rr={v['repetition']:v for v in r['runs'] if v['variant']==ref}
            for variant in r['median_s']:
                values=[v for v in r['runs'] if v['variant']==variant]
                times=[v['wall_s']*1000 for v in values]
                ratios=np.asarray([rr[v['repetition']]['wall_s']/v['wall_s'] for v in values])
                bootstrap=np.median(ratios[rng.integers(0,len(ratios),(10000,len(ratios)))],axis=1)
                lo,hi=np.percentile(bootstrap,[2.5,97.5])
                rows.append(dict(experiment=name,variant=variant,N=r['shape'][0],repeats=len(times),**stats(times),
                    median_time_saving_pct=100*(1-r['median_s'][variant]/r['median_s'][ref]),
                    ratio_of_median_times=r['median_s'][ref]/r['median_s'][variant],
                    paired_median_speedup=float(np.median(ratios)),paired_bootstrap_95_low=float(lo) if len(ratios)>1 else '',
                    paired_bootstrap_95_high=float(hi) if len(ratios)>1 else '',
                    iterations=values[0]['cg'],max_physical_residual=max(v['max_physical_residual'] for v in values),
                    max_Ceff_relative_error=max(v['Ceff_relative_difference'] for v in values),
                    qualifier='Bootstrap of paired repetition ratios; small sample, same GPU session; exploratory interval'))
        for v in r.get('rows',[]):
            if 'fields' not in v:continue
            fields.append(dict(experiment=name,N=r['shape'][0],variant=v['variant'],
                Ceff_absolute_error=v['Ceff_absolute_error'],Ceff_relative_error=v['Ceff_relative_error'],
                field_max_relative_l2=max(x['relative_l2'] for x in v['fields']),
                field_max_absolute_l2=max(x['absolute_l2'] for x in v['fields']),
                field_max_absolute_linf=max(x['absolute_linf'] for x in v['fields']),
                field_max_relative_linf=max(x['relative_linf'] for x in v['fields']),
                iterations=v['iterations'],max_physical_residual=v['max_physical_residual'],
                symmetry_relative=v['symmetry_relative'],min_Ceff_eigenvalue=v['min_Ceff_eigenvalue']))
    for path in sorted(a.input.glob('operator_*.json')):
        r=json.loads(path.read_text())
        for name,v in r['records'].items():
            if not isinstance(v,dict) or 'rows' not in v:continue
            micro.append(dict(experiment=path.stem,N=r['shape'][0],rhs_count=r.get('rhs_count',6),operation=name,
                median_wall_us=v['median_wall_s']*1e6,median_event_us=v['median_s']*1e6,
                relative_error=v.get('relative_error',''),scope=v.get('scope',r['scope'])))
    for name,records in (('experiment_statistics.csv',rows),('field_validation.csv',fields),('operator_microbenchmarks.csv',micro)):
        write_csv(a.output/name,records)
    path=a.input/'fft_trace_summary.json'
    if path.exists():
        trace=json.loads(path.read_text());records=[]
        for name,t in trace.items():
            for component,cost in t['exclusive_ms'].items():
                records.append(dict(case=name,component=component,exclusive_ms=cost,fraction_wall=cost/t['wall_ms']))
        write_csv(a.output/'fft_policy_trace.csv',records)
        groups={'cuFFT':['operator_fft'],'Coordinate maps / Cε / adjoint':['operator_other'],
            'CG updates':['solver_update','solver_elementwise'],'Explicit reductions':['reductions'],
            'Postprocess':['postprocess_kernels','idle_host_postprocess']}
        names=list(trace);bottom=np.zeros(2);fig,ax=plt.subplots(figsize=(10,4.5))
        for label,keys in groups.items():
            values=np.asarray([sum(trace[name]['exclusive_ms'].get(k,0) for k in keys) for name in names])
            ax.bar(['Default full','Components'],values,bottom=bottom,label=label);bottom+=values
        values=np.asarray([trace[name]['wall_ms'] for name in names])-bottom
        ax.bar(['Default full','Components'],values,bottom=bottom,label='Setup / host / transfer / other')
        ax.set(ylabel='Exclusive profiled wall time [ms]',title='Same short fibers, Vf 15%, AR 10, 192³, same physical residual')
        ax.legend(loc='center left',bbox_to_anchor=(1,.5),fontsize=9)
        fig.tight_layout();fig.savefig(a.output/'fft_policy_breakdown.png',dpi=180)
        fig.savefig(a.output/'fft_policy_breakdown.svg');plt.close(fig)
    clean_svg(a.output)


if __name__=='__main__':main()
