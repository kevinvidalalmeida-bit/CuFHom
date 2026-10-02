"""Rebuild measurements and the review report from portable evidence, offline."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASELINE='97939c6252ac81d4bea0b7e80eb4bbb962ab9cb2'
CASES=['G08','G09','F192','N64','N256','R128']


def digest(data):return hashlib.sha256(data).hexdigest()
def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def collect(raw):
    files={}
    for case in CASES:
        for prefix in ('split_variants_','reuse_','fields_split_','fields_reuse_verified_'):
            directory=raw/(prefix+case)
            assert (directory/'report.json').exists(),directory
            for path in sorted(directory.glob('*.json')):
                files[str(path.relative_to(raw))]=json.loads(path.read_text())
    for path in sorted((raw/'fields_split_N8').glob('*.json')):
        files[str(path.relative_to(raw))]=json.loads(path.read_text())
    for name in ('compliance_screen_with_functionals.json','softness_verified.json'):
        files[name]=json.loads((raw/name).read_text())
    trace=json.loads(gzip.decompress((raw/'trace_evidence.json.gz').read_bytes()))
    sources={};history={}
    paths=list((ROOT/'src').rglob('*.py'))+list(Path(__file__).parent.glob('*.py'))
    paths+=list((ROOT/'descartados/reference_split_20261002').glob('*.py'))
    paths += [ROOT/'tests/test_linear_final_audit.py',ROOT/'benchmarks/profile_g09_linear.py',
              ROOT/'benchmarks/audit_trace_summary.py',ROOT/'benchmarks/linear_strategy_20261002/validate_fields.py']
    for path in sorted(set(paths)):
        value=path.read_text();sha=digest(value.encode())
        sources[str(path.relative_to(ROOT))]=dict(text=value,sha256=sha)
        if path.is_relative_to(ROOT/'src'):
            old=subprocess.check_output(['git','show',f'{BASELINE}:{path.relative_to(ROOT)}'],cwd=ROOT).decode()
            if old!=value:history[digest(old.encode())]=dict(path=str(path.relative_to(ROOT)),text=old)
    def historic(path,value):history[digest(value.encode())]=dict(path=path,text=value)
    # Exact original runner bytes, before extending the cases and moving pilots.
    for name in ('run_split.py','validate_split_fields.py'):
        path=Path(__file__).parent/name
        old=path.read_text().replace("sys.path.insert(0,str(ROOT/'descartados/reference_split_20261002'))\n",'')
        old=old.replace("ROOT/'descartados/reference_split_20261002/reference_split.py'","Path(__file__).with_name('reference_split.py')")
        old=old.replace("dict(settings,solver_rtol=args.rtol,linear_final_audit_reuse=False),context)\n              for name,(settings,context) in variants.items()}",
                        "dict(settings,solver_rtol=args.rtol),context) for name,(settings,context) in variants.items()}")
        historic(str(path.relative_to(ROOT)),old)
        old=old.replace("choices=['G08','G09','N64','N256','R128','F192']","choices=['G08','G09','N64','F192']")
        old=old.replace("choices=['G08','G09','N64','N256','R128','F192','N8']","choices=['G08','G09','N64','F192','N8']")
        old=old.replace("source=('verified_'+args.case if args.case in ('G09','F192') else 'final_'+args.case)",
                        "source=('final_'+args.case if args.case in ('G08','N64') else 'verified_'+args.case)")
        historic(str(path.relative_to(ROOT)),old)
    historic('benchmarks/profile_g09_linear.py',
             (ROOT/'benchmarks/profile_g09_linear.py').read_text().replace('    linear_final_audit_reuse=False,\n',''))
    for path,value in trace['trace_sources'].items():historic(path,value)
    logs={name:Path(path).read_text() for name,path in dict(
        production='/tmp/cufhom_code_review_full_tests.txt',
        identity='/tmp/cufhom_reference_split_final_tests.txt',
        audit='/tmp/cufhom_final_audit_specific_tests.txt').items()}
    return dict(schema=1,baseline_commit=BASELINE,raw_root=str(raw.resolve()),files=files,
        trace=trace,source_snapshot=sources,historical_sources=history,test_logs=logs,
        documents={'DERIVATION.md':Path(__file__).with_name('DERIVATION.md').read_text()},
        exclusions='Preliminary split_G09 and initial strict bitwise field run are excluded from timing/validation tables. Verified field reports include repeated disabled controls for cleanup variability.')


def summarize(bundle):
    files=bundle['files'];snap=bundle['source_snapshot']
    known={s['sha256'] for s in snap.values()}|set(bundle['historical_sources'])
    for sha,value in bundle['historical_sources'].items():assert digest(value['text'].encode())==sha
    for value in snap.values():assert digest(value['text'].encode())==value['sha256']
    for sha in bundle['trace']['source_sha256'].values():assert sha in known
    split=[];reuse=[];fields=[];trace=[]
    for case in CASES:
        for prefix,variants,destination in [('split_variants_',('reference_split','reference_split_recurrence'),split),
                                           ('reuse_',('reuse',),reuse)]:
            directory=prefix+case;report=files[directory+'/report.json']
            assert report['inputs_unchanged'] and len(report['load_ids'])==6
            for sha in report['source_sha256'].values():assert sha in known
            if prefix=='reuse_':
                assert report['source_sha256']=={p:snap[p]['sha256'] for p in report['source_sha256']}
            for sha in report.get('pilot_source_sha256',{}).values():assert sha in known,sha
            assert report['benchmark_sha256'] in known
            reference={row['repetition']:row for row in report['runs'] if row['variant']=='current'}
            assert len(reference)==5
            for variant in variants:
                samples=[row for row in report['runs'] if row['variant']==variant]
                assert len(samples)==5
                gains=[100*(1-row['wall_s']/reference[row['repetition']]['wall_s']) for row in samples]
                for row in report['runs']+report['warmups']:
                    C=np.asarray(row['Ceff']);assert np.linalg.eigvalsh((C+C.T)/2)[0]>0
                    assert row['max_physical_residual']<=1e-5 and row['Ceff_relative_difference']<1e-6
                    detail=files[f"{directory}/{row['variant']}_{row['repetition']}.json"]
                    infos=detail['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
                    assert len(infos)==6 and all(i['true_residual_converged'] and i['true_norm_res_rel']<=1e-5 for i in infos)
                    if prefix=='reuse_':assert all(i['linear_final_audit_reuse']==(row['variant']=='reuse') for i in infos)
                base,new=report['median_s']['current'],report['median_s'][variant]
                assert all(row['cg']==reference[row['repetition']]['cg'] for row in samples)
                destination.append(dict(case=case,variant=variant,shape=report['shape'],current_s=base,new_s=new,
                    saving_percent=100*(1-new/base),paired_min=min(gains),paired_max=max(gains),cg=samples[0]['cg'],
                    max_physical_residual=max(row['max_physical_residual'] for row in samples),
                    max_Ceff_relative_difference=max(row['Ceff_relative_difference'] for row in samples)))
        for prefix in ('fields_split_','fields_reuse_verified_'):
            report=files[prefix+case+'/report.json']
            for key in ('source_sha256','pilot_source_sha256'):
                for sha in report.get(key,{}).values():assert sha in known,sha
            for row in report['rows']:
                assert len(row['fields_sha256'])==6 and row['max_physical_residual']<=1e-5
                assert row['min_Ceff_eigenvalue']>0 and row['Ceff_relative_error']<2e-6
                maximum=max(m['relative_l2'] for m in row['fields']);assert maximum<2e-4
                if prefix=='fields_split_' and row['variant'].startswith('reference_split'):
                    assert len(row['split_activation'])==6 and all(i['split_calls']>0 for i in row['split_activation'])
                fields.append(dict(case=case,method=prefix,variant=row['variant'],
                    max_field_l2=maximum,Ceff_relative_error=row['Ceff_relative_error'],
                    physical_residual=row['max_physical_residual'],
                    fields_bitwise_equal=row.get('fields_bitwise_equal'),Ceff_bitwise_equal=row.get('Ceff_bitwise_equal')))
    small=files['fields_split_N8/report.json']
    assert small['rows'][0]['variant']=='cpu' and small['rows'][0]['max_physical_residual']<1e-10
    for row in small['rows']:
        assert row['Ceff_relative_error']<2e-6 and max(m['relative_l2'] for m in row['fields'])<2e-4
    for case,record in bundle['trace']['cases'].items():
        t=record['trace'];e=t['exclusive_ms'];s=record['solve']['diagnostic']
        assert abs(sum(e.values())-t['wall_ms'])<1e-6
        trace.append(dict(case=case,wall_ms=t['wall_ms'],fft_ms=e['operator_fft'],
            material_maps_update_ms=e['operator_other']+e['operator_update_fused'],
            host_in_cg_ms=e['idle_host_in_cg'],reductions_ms=e['reductions'],
            postprocess_kernel_ms=e['postprocess_kernels'],
            D2H_inside_cg=t['transfers']['Device-to-Host']['inside_cg_calls'],
            D2H_bytes=t['transfers']['Device-to-Host']['bytes'],stages_s=s['stages_s'],
            exclusive_ms=e,iterations=s['iterations']))
    compliance=files['compliance_screen_with_functionals.json']
    assert all(r['reference_split_relative_error']<4e-16 and r['compliance_eigen_min']>1-2e-11 for r in compliance['rows'])
    softness=files['softness_verified.json']
    for sha in softness['source_sha256'].values():assert sha in known
    assert max(r['full_relative_error'] for r in softness['rows'])<2e-6
    assert '345 passed' in bundle['test_logs']['production'] and '7 subtests passed' in bundle['test_logs']['production']
    assert '13 passed' in bundle['test_logs']['identity'] and '7 passed' in bundle['test_logs']['audit']
    return dict(split=split,reuse=reuse,fields=fields,small_CPU=small,trace=trace,
                compliance=compliance,softness=softness)


def csv_file(path,rows):
    with path.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def figure(summary,output):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='cufhom-code-review-20261002'
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,4.2),layout='constrained')
    labels=['FFT','Material, mapas y update','Otros kernels GPU','Copias GPU','CPU/API sin actividad GPU']
    colors=['#287b99','#62a5ac','#abcece','#a99d8c','#e8af6b']
    traces=summary['trace'];left=np.zeros(len(traces))
    values=[]
    for r in traces:
        e=r['exclusive_ms'];host=sum(v for k,v in e.items() if k.startswith('idle_'))
        copies=e.get('transfers',0)+e.get('memory_device_copy',0)
        other=r['wall_ms']-r['fft_ms']-r['material_maps_update_ms']-host-copies
        values.append(np.asarray([r['fft_ms'],r['material_maps_update_ms'],other,copies,host])*100/r['wall_ms'])
    for i,(label,color) in enumerate(zip(labels,colors)):
        v=np.asarray(values)[:,i];ax.barh(np.arange(len(traces)),v,left=left,label=label,color=color);left+=v
    ax.set_yticks(range(len(traces)),[f"{r['case']} ({r['wall_ms']:.1f} ms)" for r in traces]);ax.invert_yaxis()
    ax.set_xlim(0,100);ax.set_xlabel('Porcentaje exclusivo de pared (%)')
    ax.set_title('CuFHom antes de reutilizar la auditoría: una traza warm por caso')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,frameon=False,fontsize=8)
    fig.savefig(output/'timeline.png',dpi=170);fig.savefig(output/'timeline.svg',metadata={'Date':None});plt.close(fig)
    p=output/'timeline.svg';p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')


def render(s):
    lines=['# CuFHom: identidad de referencia y reutilización de la auditoría','',
        '**Se integra la reutilización final; la identidad queda como piloto retirado.** '
        'La identidad ahorra 1–4% en varias mallas grandes, sin cambiar sus iteraciones, '
        'pero pierde precisión para fases muy blandas. Reutilizar deformación y tensión '
        'evita trabajo repetido y conserva la ecuación y su auditoría física.','',
        '## Identidad probada en GPU','',
        '`H = I + Q_R* (C-C₀) Q_R` sobre las coordenadas admisibles. La norma directa '
        'se fusiona con el mapa; el material de referencia escribe contraste cero. '
        'Seis cargas, FP32 en campos/FFT, sumas FP64 y residual físico 1e-5. '
        'RTX 5070 Ti; cinco repeticiones rotadas tras calentamiento. El tiempo incluye '
        'setup de material, solve, auditorías, posprocesado y JSON; las entradas se '
        'cargan/compilan antes de medir. Se mantienen las optimizaciones A/B previas.','',
        '| Caso | Actual s | Split directo s | Ahorro % | CG, ambos | Ahorro pareado min–max % |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in s['split']:
        if r['variant']=='reference_split':lines.append(f"| {r['case']} | {r['current_s']:.6f} | {r['new_s']:.6f} | {r['saving_percent']:.2f} | {r['cg']} | {r['paired_min']:.2f}–{r['paired_max']:.2f} |")
    lines += ['', 'F192: fibras Vf 15%, AR 10, orientación aleatoria. N64/N256: '
        'inclusiones compactas; R128: regiones con dos orientaciones. El método no '
        'usa etiquetas de fibras. G08 empeora ligeramente; N64 tiene variación que '
        'impide afirmar una ganancia. Los intervalos son extremos de cinco pares, '
        'no intervalos de confianza.','',
        'La variante recurrente de `||p||²` tampoco supera a la directa en las '
        'cuatro mallas grandes; sus tiempos quedan en [split.csv](split.csv). '
        'La igualdad de H conserva espectro y condición: **no reduce CG ni FFT**.','',
        '## Límite numérico encontrado','',
        '| C = η C₀ | Error relativo de p*Hp, original | Error relativo, split |',
        '| ---: | ---: | ---: |']
    for r in s['softness']['rows']:lines.append(f"| {r['eta']:.0e} | {r['full_relative_error']:.3g} | {r['split_relative_error']:.3g} |")
    lines += ['', 'En una fase 1e6 veces más blanda, el split pierde 7,44% por cancelación. '
        'Los productos FP64 no reparan el redondeo previo de `C-C₀` y ε en FP32. '
        'No equivale a 7,44% de error medido de Ceff: este cribado mide el denominador '
        'de una dirección. La ruta original conserva su precisión. Por eso no se '
        'activa el split como opción general predeterminada. '
        '[Código aislado y decisión](../../descartados/reference_split_20261002/README.md).','',
        '## Reutilización final integrada','',
        '`linear_final_audit_reuse=True` retiene deformación/tensión exclusivamente '
        'en las comprobaciones verdaderas de CG. Reutiliza solo el mismo iterado '
        'sin un campo inicial añadido. La proyección exige los mismos factores '
        'material/GN del operador físico; las otras rutas evalúan el operador completo. '
        'Una corrección posterior vuelve a auditar. La escala, la proyección, el '
        'umbral y los productos FP64 permanecen. Se elimina una IFFT de seis '
        'componentes y un producto de material por carga reutilizada; no se conserva '
        'el espectro de tensión.','',
        '| Caso | Reutilización desactivada s | Activada s | Ahorro % | CG, ambos |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in s['reuse']:lines.append(f"| {r['case']} | {r['current_s']:.6f} | {r['new_s']:.6f} | {r['saving_percent']:.2f} | {r['cg']} |")
    lines += ['', 'Son cinco pares alternados con el mismo código final y entradas, '
        'comparando solo la opción de reutilización. Los ahorros de ambas tablas '
        'no se suman: son campañas distintas y la identidad sigue retirada. '
        'En 64³ la diferencia queda dentro de la variación. G09, F192 y N256 '
        'mejoran en los cinco pares; R128 mejora en cuatro de cinco. [reuse.csv](reuse.csv).','',
        '## Validación','',
        '**345 pruebas y 7 subpruebas de producción; 13 controles independientes '
        'del split.** Los siete controles nuevos de reutilización verifican '
        'formas pares/impares, fusiones activadas/desactivadas, campos completos, '
        'campo inicial con material cambiado, iterado distinto y operador físico '
        'incompatible. La falsa convergencia todavía provoca corrección y auditoría.','',
        'El split se compara en seis campos completos de los seis casos de la '
        'tabla, además de 8³ contra CPU64 a 1e-10. El tensor CUDA se compara '
        'también contra el operador CPU64 con tensores SPD anisótropos y '
        'contrastes negativos. Todos los solves aceptados tienen Ceff SPD y '
        'residual físico ≤1e-5.','']
    for r in s['fields']:
        if r['method']=='fields_reuse_verified_' and r['variant']=='reuse':
            lines.append(f"- {r['case']}, reutilización: diferencia L2 máxima de campo {r['max_field_l2']:.3g}; Ceff relativa {r['Ceff_relative_error']:.3g}; campos bit a bit: {'sí' if r['fields_bitwise_equal'] else 'no'}.")
    lines += ['', 'G08/N64 incluyen una segunda ejecución con reutilización desactivada '
        'para mostrar la variación del cleanup con reducciones atómicas. Los '
        'valores y banderas exactos se conservan en [fields.csv](fields.csv); '
        'no se afirma igualdad bit a bit cuando esos controles difieren.','',
        '## Dónde atacar después, según el código y la traza','',
        'La traza siguiente corresponde al código A/B previo a esta reutilización '
        f'(`{BASELINE[:12]}`), una ejecución warm instrumentada por caso. '
        'No se usa como muestra estadística de velocidad. Las categorías exclusivas '
        'no se solapan; los tiempos inclusivos de API pueden esperar a GPU y no '
        'son tiempo de transferencia.','',
        '![Desglose exclusivo de pared](timeline.png)','',
        '| Caso | Pared ms | FFT % | Material/mapas/update % | Host en CG sin GPU % | D2H dentro de CG |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in s['trace']:lines.append(f"| {r['case']} | {r['wall_ms']:.2f} | {100*r['fft_ms']/r['wall_ms']:.2f} | {100*r['material_maps_update_ms']/r['wall_ms']:.2f} | {100*r['host_in_cg_ms']/r['wall_ms']:.2f} | {r['D2H_inside_cg']} |")
    lines += ['', 'En G09 esas tres familias de kernels ocupan 93,13% de pared; incluyen '
        'RHS/auditorías además de H. El posprocesado ocupa ≈1,3% del total. '
        'En G08 hay 508 lecturas D2H dentro de CG, pero solo 4.676 bytes D2H '
        'en toda la traza: predomina la frecuencia de sincronización, no el volumen '
        'de transferencia. `float(cp.sum(...))` aparece dos veces por iteración '
        'en energía y update. Las opciones lazy/check-every del CG antiguo no '
        'eliminan esas lecturas en `fused_cg` de coordenadas.','',
        'La siguiente prioridad para mallas pequeñas es mantener alpha, beta y '
        'normas FP64 en GPU, comprobando estado por grupos sin cambiar el '
        'residual verdadero. Para grandes, probar conservar el espectro final '
        'y eliminar buffers/tráfico dentro de H; las sincronizaciones ofrecen '
        'mucho menos margen. Estas extensiones no se implementaron en esta campaña.','',
        '## Qué enseñó el cribado matemático adicional','',
        'Se probó la compliance fina proyectada, distinta del Q1/GenEO retirado. '
        'Invertir C y proyectar no equivale a invertir el operador restringido. '
        'La condición baja en los cinco problemas CPU64 5³, pero las iteraciones '
        'suben en tres y la aplicación añade otra pareja de FFT. El coste '
        'aproximado con H/B iguales queda entre 1,42 y 2,36 veces la base; '
        'es un cribado, no un tiempo GPU. [Derivación](DERIVATION.md).','',
        'Un laminado tiene κ≈30 y solo siete CG para las seis cargas; una inclusión '
        'compacta con κ≈30 requiere 107 en la misma malla de 372 grados reales. '
        'El espectro completo, su agrupamiento y los pesos del RHS explican '
        'por qué tamaño/κ por sí solos no predicen iteraciones. La memoria aumenta '
        'el coste de cada H en una malla grande; no explica por sí misma más CG.','',
        'Ceff por tensiones medias se comprobó solo en esos problemas CPU64: '
        'difiere hasta 2,5e-9 de la matriz de energía a rtol 1e-8. Sigue pendiente '
        'una validación GPU al umbral físico real antes de sustituir el posprocesado.','',
        '## Evidencia y reproducción','',
        '[evidence.json.gz](evidence.json.gz) conserva JSON crudos, logs de pruebas, '
        'trazas resumidas y las versiones exactas del código por SHA256. Las '
        'mediciones split/trazas preceden el cambio de producción; las de '
        'reutilización coinciden con sus hashes finales. [manifest.json](manifest.json) '
        'verifica archivos. Entradas grandes, SQL y `.nsys-rep` permanecen en '
        '`benchmarks/results/code_review_20261002/`, fuera de Git.','',
        '```bash',
        '.venv/bin/python benchmarks/code_review_20261002/build_report.py \\',
        '  --evidence benchmarks/code_review_20261002/evidence.json.gz \\',
        '  --output /tmp/cufhom_code_review',
        '```','',
        'Reconstruir el informe no requiere GPU ni sudo. Para repetir los tiempos '
        'GPU, cargar `benchmarks/reference_solvers/env_gpu.sh` y ejecutar '
        '`run_split.py` o `run_reuse.py` con `--case`, `--repeats 5` y un '
        'directorio de salida nuevo. Los hashes históricos permiten evaluar '
        'la procedencia; medir con código futuro puede producir otros resultados.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path)
    parser.add_argument('--evidence',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if (args.raw is None)==(args.evidence is None):parser.error('Choose raw or evidence')
    args.output.mkdir(parents=True,exist_ok=True)
    bundle=collect(args.raw) if args.raw else json.loads(gzip.decompress(args.evidence.read_bytes()))
    summary=summarize(bundle)
    evidence=gzip.compress(encoded(bundle),mtime=0);(args.output/'evidence.json.gz').write_bytes(evidence)
    (args.output/'summary.json').write_bytes(encoded(summary))
    for name in ('split','reuse','fields'):csv_file(args.output/(name+'.csv'),summary[name])
    csv_file(args.output/'trace.csv',[{k:v for k,v in r.items() if k not in ('stages_s','exclusive_ms')} for r in summary['trace']])
    figure(summary,args.output);(args.output/'RESULTS.md').write_text(render(summary))
    for name,value in bundle['documents'].items():(args.output/name).write_text(value)
    generated=['evidence.json.gz','summary.json','split.csv','reuse.csv','fields.csv','trace.csv','timeline.png','timeline.svg','RESULTS.md','DERIVATION.md']
    manifest=dict(schema=1,baseline_commit=bundle['baseline_commit'],raw_root=bundle['raw_root'],
        output_sha256={name:digest((args.output/name).read_bytes()) for name in generated},
        source_sha256={path:value['sha256'] for path,value in bundle['source_snapshot'].items()},
        historical_source_sha256=sorted(bundle['historical_sources']),exclusions=bundle['exclusions'])
    (args.output/'manifest.json').write_bytes(encoded(manifest))
    print('Verified split/reuse: six cases, five repeats each; full fields and exact source versions.',flush=True)


if __name__=='__main__':main()
