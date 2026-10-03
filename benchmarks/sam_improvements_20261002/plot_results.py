"""Static figures from measured kernel times and certified SAM commits."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
data = json.loads((HERE/'validation.json').read_text())
rows = data['kernels']
counts = [0]
for name in ('sam_dense_cylinder_ar30_60_20261002',
             'sam_dense_cylinder_ar30_resume_20261002',
             'sam_dense_cylinder_ar30_shake_20261002',
             'sam_dense_cylinder_ar30_candidates128_20261002'):
    path = ROOT/'benchmarks/results'/name/'events.jsonl'
    if not path.exists():
        continue
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['event']=='commit' and row['certificate']['accepted']:
            counts.append(row['proposed_count'])

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(left,right) = plt.subplots(1,2,figsize=(10.8,4.2),layout='constrained')
x = np.arange(len(rows))
left.bar(x-.18,[1000*r['cached_objective_reference']['median_s'] for r in rows],
         .36,label='Contactos anteriores',color='#8999ad')
left.bar(x+.18,[1000*r['cached_objective_new']['median_s'] for r in rows],
         .36,label='Contactos nuevos',color='#147a73')
left.set(xticks=x,xticklabels=[str(r['fibers']) for r in rows],xlabel='Número de fibras',
         ylabel='Tiempo por evaluación (ms)',yscale='log')
left.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,fontsize=9)
left.grid(axis='y',alpha=.18)
for k,r in enumerate(rows):
    left.text(k,120,f"{r['speedup']:.2f}×",ha='center',color='#147a73')
left.set_ylim(.2,200)
vf = np.asarray(counts)*np.pi*30/4/60**3*100
right.plot(np.arange(len(vf)),vf,'o-',markersize=3,color='#147a73',label='Cilindros certificados')
right.axhline(60,color='#b04c38',linestyle='--',label='Objetivo 60 %')
right.set(xlabel='Inserciones colectivas aceptadas',ylabel='Vf nominal de cilindros (%)',ylim=(0,65))
right.legend(frameon=False,loc='center left')
right.grid(axis='y',alpha=.18)
right.annotate(f'{vf[-1]:.2f} %',xy=(len(vf)-1,vf[-1]),xytext=(-4,12),
               textcoords='offset points',ha='right',color='#147a73')
fig.suptitle('AR = 30: rendimiento medido y avance del empaquetamiento')
fig.savefig(HERE/'sam_improvements.png',dpi=180)
fig.savefig(HERE/'sam_improvements.pdf')
