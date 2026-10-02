"""Build the strategy report and offline evidence; no GPU or sudo required."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CASES=['G08','G09','F96','F128','F192','F256','N64','N256','R64','R128']
PRIMARY={case:('verified_'+case if case in ('G09','F192') else 'final_'+case) for case in CASES}
PILOTS=[('verified_auxiliary_G09','auxiliary','Auxiliar Q1'),
        ('verified_auxiliary_G09','geneo','GenEO auxiliar, τ=0,15'),
        ('verified_fine_geneo_G09','fine_geneo','GenEO fino, τ=0,15'),
        ('verified_fine_geneo_tau05_G09','fine_geneo','GenEO fino, τ=0,5')]


def digest(data):return hashlib.sha256(data).hexdigest()
def encoded(obj):return (json.dumps(obj,indent=2,allow_nan=False)+'\n').encode()


def collect(raw):
    files={}
    prefixes=('final_','verified_','fft_','energy_','auxiliary_','fine_geneo_','fields_','memory_')
    for directory in sorted(raw.iterdir()):
        if directory.is_dir() and directory.name.startswith(prefixes) and (directory/'report.json').exists():
            for path in sorted(directory.glob('*.json')):
                files[str(path.relative_to(raw))]=json.loads(path.read_text())
    files['cpu_spectrum_v2.json']=json.loads((raw/'cpu_spectrum_v2.json').read_text())
    sources={}
    paths=list((ROOT/'src').rglob('*.py'))+list(Path(__file__).parent.glob('*.py'))
    paths+=list((ROOT/'descartados/auxiliary_geneo_20261002').glob('*.py'))
    paths+=list((ROOT/'descartados/energia_integrada_20261002').glob('*.py'))
    paths+=[ROOT/'benchmarks/profile_g09_linear.py',ROOT/'tests/test_linear_energy.py',
            ROOT/'tests/test_linear_fft_tuning.py',ROOT/'tests/test_fft_component_batches.py']
    for path in sorted(set(paths)):
        source=path.read_text();sources[str(path.relative_to(ROOT))]=dict(text=source,sha256=digest(source.encode()))
    logs={name:Path(path).read_text() for name,path in dict(
        production='/tmp/cufhom_strategy_final_suite.txt',
        auxiliary='/tmp/cufhom_strategy_aux_tests.txt').items()}
    return dict(schema=1,baseline_commit='0cc0d47b04a423ff491250fa4deda105bd88b658',
        raw_root=str(raw.resolve()),files=files,source_snapshot=sources,test_logs=logs)


def infos(files,directory,name,rep=0):
    return files[f'{directory}/{name}_{rep}.json']['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']


def summarize(bundle):
    files=bundle['files'];rows=[];pilot_rows=[]
    for case,directory in PRIMARY.items():
        report=files[f'{directory}/report.json']
        assert report['inputs_unchanged'] and len(report['load_ids'])==6
        source_differences=[path for path,sha in report['source_sha256'].items()
                            if bundle['source_snapshot'][path]['sha256']!=sha]
        # Eight timings preceded the profile-only material counter fix.
        # G09/F192 were repeated after it; all computational kernels match.
        if case in ('G09','F192'):assert not source_differences
        else:assert set(source_differences)<= {'src/cufhom_core/applications.py'}
        samples={name:[r for r in report['runs'] if r['variant']==name] for name in ('current','combined')}
        assert all(len(v)==5 for v in samples.values())
        base,new=(report['median_s'][name] for name in ('current','combined'))
        pairs=[100*(1-samples['combined'][i]['wall_s']/samples['current'][i]['wall_s']) for i in range(5)]
        per_load=infos(files,directory,'combined')
        assert all(i['linear_energy_update'] and i['true_residual_converged'] and i['true_norm_res_rel']<=1e-5 for i in per_load)
        tuning=infos(files,directory,'combined','warmup')[0]['linear_fft_tuning']
        assert tuning['status']=='tuned'
        for row in report['runs']+report['warmups']:
            assert row['max_physical_residual']<=1e-5 and row['Ceff_relative_difference']<1e-6
            C=np.asarray(row['Ceff']);assert np.linalg.eigvalsh((C+C.T)/2)[0]>0
        rows.append(dict(case=case,directory=directory,shape=report['shape'],current_s=base,default_s=new,
            saving_percent=100*(1-new/base),paired_saving_min=min(pairs),paired_saving_max=max(pairs),
            fft_policy=tuning['policy'],tuning_s=tuning['tuning_s'],
            amortization_solves=int(np.ceil(tuning['tuning_s']/(base-new))) if base>new else None,
            cg_current=[r['cg'] for r in samples['current']],cg_default=[r['cg'] for r in samples['combined']],
            max_physical_residual=max(r['max_physical_residual'] for r in report['runs']),
            Ceff_relative_difference=max(r['Ceff_relative_difference'] for r in report['runs']),
            input_sha256=report['input_sha256'],source_sha256=report['source_sha256'],
            source_differences_from_snapshot=source_differences))
    for directory,name,label in PILOTS:
        report=files[f'{directory}/report.json'];loads=infos(files,directory,name)
        details=loads[0]['auxiliary_preconditioner'][0]
        assert all(i['true_residual_converged'] and i['true_norm_res_rel']<=1e-5 for i in loads)
        assert report['inputs_unchanged'] and len([r for r in report['runs'] if r['variant']==name])==3
        # These measurements use exactly the source snapshot bundled here.
        for path,sha in report['source_sha256'].items():assert bundle['source_snapshot'][path]['sha256']==sha
        for path,sha in report['strategy_source_sha256'].items():assert bundle['source_snapshot'][path]['sha256']==sha
        pilot_rows.append(dict(label=label,directory=directory,variant=name,
            reference_s=report['median_s']['current'],median_s=report['median_s'][name],
            iterations=sum(i['cg_iterations'] for i in loads),setup_s=details['setup_s'],
            retained_gpu_bytes=details['retained_gpu_bytes'],details=details))
    validations=[]
    for name,report in files.items():
        if name.startswith('fields_') and name.endswith('/report.json'):
            for row in report['rows']:
                assert len(row['fields'])==6 and row['max_physical_residual']<=1e-5
                assert row['min_Ceff_eigenvalue']>0 and row['Ceff_relative_error']<2e-6
                assert max(r['relative_l2'] for r in row['fields'])<2e-4
            validations.append(dict(file=name,shape=report['shape'],variants=[r['variant'] for r in report['rows']],
                max_field_relative_l2=max(m['relative_l2'] for r in report['rows'] for m in r['fields']),
                max_Ceff_relative_error=max(r['Ceff_relative_error'] for r in report['rows']),
                max_physical_residual=max(r['max_physical_residual'] for r in report['rows'])))
    assert '338 passed' in bundle['test_logs']['production'] and '7 subtests passed' in bundle['test_logs']['production']
    assert '4 passed' in bundle['test_logs']['auxiliary']
    return dict(primary=rows,pilots=pilot_rows,validations=validations,
        memory=files['memory_final_G09/report.json'],cpu_spectrum=files['cpu_spectrum_v2.json'])


def write_csv(path,rows,keys):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)


def plot(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt']='cufhom-linear-strategy-20261002'
    import matplotlib.pyplot as plt
    rows=summary['primary'];values=np.asarray([r['saving_percent'] for r in rows])
    labels=[r['case'].replace('F','Fibras ').replace('N','Compactas ').replace('R','Regiones ') for r in rows]
    labels=[s+'³' if not s.startswith('G') else s for s in labels]
    low=np.maximum(0,values-np.array([r['paired_saving_min'] for r in rows]))
    high=np.maximum(0,np.array([r['paired_saving_max'] for r in rows])-values)
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    colors=['#1b7b93' if v>1 else '#999999' for v in values]
    y=np.arange(len(rows));ax.barh(y,values,color=colors,xerr=np.array([low,high]),capsize=3)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.axvline(0,color='black',linewidth=.6)
    ax.set_xlabel('Reducción de tiempo de las seis cargas (%)')
    ax.set_title('CuFHom: FFT seleccionada + energía/update fusionados')
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    for i,v in enumerate(values):
        label_x=max(0,v,rows[i]['paired_saving_max'])+1.2
        ax.text(label_x,i,f'{v:.1f}%',va='center',fontsize=9)
    ax.set_xlim(-6,42)
    fig.savefig(output/'warm_savings.png',dpi=170)
    svg=output/'warm_savings.svg';fig.savefig(svg,metadata={'Date':None});plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def render(summary):
    rows=summary['primary'];by_case={r['case']:r for r in rows};g=by_case['G09'];f=by_case['F192']
    lines=['# CuFHom: estrategias lineales implementadas y medidas','',
        f'**Se integran FFT y energía/update por defecto en la ruta compatible.** En G09: '
        f'{g["current_s"]:.3f} → {g["default_s"]:.3f} s ({g["saving_percent"]:.2f}%); '
        f'en fibras 192³: {f["current_s"]:.3f} → {f["default_s"]:.3f} s ({f["saving_percent"]:.2f}%).',
        'Se implementaron también el auxiliar Q1, GenEO y su Galerkin fino real. '
        'No mejoran el tiempo y quedan en `descartados`, con derivación y código reproducible.','',
        '## Seis cargas: configuración final','',
        'RTX 5070 Ti. Cinco repeticiones alternadas tras calentamiento; medianas de pared. '
        'Incluye material, solve, auditorías físicas, posprocesado y JSON. Carga/compilación '
        'de geometría se mide aparte. La referencia desactiva solo las dos opciones nuevas; '
        'mantiene las fusiones y el suavizado previamente aceptados. Campos/FFT FP32, '
        'productos de Krylov FP64, mismo residual físico 1e-5 y máximo 1000 iteraciones.','',
        '| Caso | Previa s | Predeterminados s | Ahorro % | FFT elegida | CG antes/después |',
        '| --- | ---: | ---: | ---: | --- | ---: |']
    for r in rows:
        lines.append(f'| {r["case"]} | {r["current_s"]:.6f} | {r["default_s"]:.6f} | {r["saving_percent"]:.2f} | '
            f'{r["fft_policy"]} | {r["cg_current"][0]}/{r["cg_default"][0]} |')
    lines += ['', 'F = Vf 15%, AR 10, orientación aleatoria y dominio 2L; N = inclusiones '
        'compactas generales; R = regiones oblicuas con dos orientaciones. G08/G09 '
        'conservan sus entradas originales. La ganancia de estas rutas viene del coste '
        'por aplicación; no disminuyen las iteraciones. 96³ conserva lote completo y '
        'su diferencia de tiempo queda dentro de la variación de esta campaña.','',
        '![Ahorro de tiempo completo](warm_savings.png)','',
        'Las barras muestran el cociente de medianas; las líneas, el mínimo/máximo de '
        'los cinco ahorros pareados. No son intervalos de confianza ni una cota universal.','',
        'G09 y F192 se repitieron con hashes idénticos al código final. Las otras ocho '
        'mediciones preceden la corrección del contador de material en '
        '`profile_timing`: solo difiere `applications.py`; los kernels y las rutas '
        'no instrumentadas medidas conservan sus operaciones. Los hashes originales '
        'y esa diferencia se registran por caso en `summary.json`.','',
        '## Selección inicial y amortización','',
        'El tuner prueba lotes de 1, 2, 3 y 6 componentes con H completo; valida su salida, '
        'mide cinco rondas rotadas y guarda la elección. Una política explícita tiene '
        'prioridad. La caché incluye forma, dtype, GPU y versiones de CUDA/cuFFT/CuPy. '
        'Cada primer uso cobra el tiempo siguiente; las siguientes cargas/ejecuciones '
        'reutilizan la decisión. No se afirma que cinco muestras de H certifiquen el '
        'plan óptimo de todo solve.','',
        '| Caso | Selección inicial ms | Solves para amortizar selección, estimación |',
        '| --- | ---: | ---: |']
    for r in rows:lines.append(f'| {r["case"]} | {1000*r["tuning_s"]:.2f} | {r["amortization_solves"] or "Sin ahorro medible"} |')
    lines += ['', 'La estimación divide el coste de selección por la diferencia entre medianas warm '
        'de los predeterminados y su referencia. Excluye creación del contexto/JIT: '
        'no sustituye una campaña independiente de arranque completamente frío. '
        'Los warmups completos y sus tiempos siguen disponibles en el bundle.','',
        '## Fusión de energía y memoria','',
        '`Re(p*Hp) = <QRp, CQRp>` mueve la suma FP64 al kernel de material. '
        'Después del R2C se aplican adjunto y actualización de x/r en el mismo kernel. '
        'Se elimina el vector Hp y se libera el espectro consumido antes del siguiente H. '
        'En G09 se evitan 334,54 MB de escritura/lectura **lógica** por iteración; '
        'no se presentan como bytes DRAM medidos.','']
    memory=summary['memory']['rows'];base=memory[0];new=memory[-1]
    lines += [f'El diagnóstico de asignaciones warm registra {base["peak_active_pool_bytes"]/2**30:.3f} → '
        f'{new["peak_active_pool_bytes"]/2**30:.3f} GiB de máximo activo en el pool CuPy. '
        'Excluye escritorio, contexto y asignaciones fuera del pool. Cuenta 276 H de '
        'iteraciones y 6 H de comprobación coordenada en ambas rutas; la auditoría '
        'Afun física y la preparación/RHS están fuera de ese conteo. No se usan sus '
        'tiempos instrumentados para calcular aceleración.',
        'Los atributos compilados de material/adjunto/adjunto suavizado indican '
        '40/32/40 registros y cero bytes locales. Es información del compilador, '
        'no una nueva campaña de contadores ni prueba de ausencia de spill en todo el programa.','',
        '## Auxiliar y GenEO: implementación y rechazo','',
        'Tres repeticiones alternadas por variante, con preparación una vez por material '
        'reutilizada en las seis cargas e incluida en el total. Cada fila usa su propia '
        'referencia contemporánea.','',
        '| G09 | Referencia s | Variante s | Preparación s, primera repetición | CG |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in summary['pilots']:
        lines.append(f'| {r["label"]} | {r["reference_s"]:.3f} | {r["median_s"]:.3f} | {r["setup_s"]:.3f} | {r["iterations"]} |')
    last=summary['pilots'][-1]['details']
    nonrigid=sum(r['selected_nonrigid'] for r in last['geneo']['local_spectra'])
    lines += ['',f'El umbral 0,5 selecciona {last["fine_energy_setup_H"]} columnas, '
        f'incluidas {nonrigid} deformables; rango de energía fino {last["fine_energy_rank"]}. '
        'Su preparación aplica H una vez por columna. Ampliar la base y usar la energía '
        'fina no reduce las 276 iteraciones. El auxiliar promedia coeficientes: puede '
        'perder contraste y modos útiles. Estas pruebas rechazan las variantes ensayadas, '
        'sin descartar todas las posibles construcciones GenEO.','',
        'La comprobación independiente CPU64 construye el espectro fino completo de '
        '5³ impar (372 grados reales sin traslaciones), con inclusiones y laminado. '
        'El auxiliar Q1 empeora κ de ≈30 a ≈105–121; GenEO con energía fina queda '
        'en ≈31 y tampoco mejora los CG de esos RHS. Son condiciones de esos '
        'problemas pequeños, no estimaciones de κ de G09.','',
        '[Derivación propia](DERIVATION.md): adjunto de transferencia, normalización, '
        'V-cycle SPD, cociente local GenEO, Galerkin fino y suavizado PCG en la métrica B. '
        '[Código retirado](../../descartados/auxiliary_geneo_20261002/README.md).','',
        '## Validación y reproducción','',
        '**338 pruebas y 7 subpruebas de producción; 4 controles del prototipo auxiliar.** '
        'Se comparan los seis campos completos en G08/G09, compactas 64³, fibras 96³/192³ '
        'y dos casos 8³ contra CPU64. Todas las rutas mantienen residual físico ≤1e-5 '
        'y Ceff SPD. CPU64 comprueba 8³; no se afirma haber resuelto G09 completo en CPU64.','',
        f'Máximo error relativo de Ceff en esas validaciones: '
        f'{max(r["max_Ceff_relative_error"] for r in summary["validations"]):.3g}; '
        f'máxima diferencia L2 de campos: {max(r["max_field_relative_l2"] for r in summary["validations"]):.3g}. '
        'Se conservan los valores por carga, incluida la comparación CPU64 y la '
        'variación debida al mismo umbral de parada.','',
        '[summary.json](summary.json) y [timings.csv](timings.csv) conservan cifras sin '
        'redondear; [evidence.json.gz](evidence.json.gz) contiene JSON de solves, cachés, '
        'validación, diagnóstico de memoria, logs de pruebas y una copia del código. '
        '[manifest.json](manifest.json) verifica hashes y procedencia. Las geometrías '
        'y los campos volumétricos grandes siguen fuera de Git.','',
        'Recrear tablas e imagen sin ejecutar CUDA ni usar sudo:','',
        '```bash',
        '.venv/bin/python benchmarks/linear_strategy_20261002/build_report.py \\',
        '  --evidence benchmarks/linear_strategy_20261002/evidence.json.gz \\',
        '  --output /tmp/cufhom_strategy_report',
        '```','',
        'Repetir una comparación GPU con un directorio nuevo:','',
        '```bash',
        'source benchmarks/reference_solvers/env_gpu.sh',
        'OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmarks/linear_strategy_20261002/run_strategy.py \\',
        '  --case G09 --route combined --repeats 5 --output benchmarks/results/new_strategy_G09',
        '```','',
        'Rutas del runner: `fft`, `energy`, `combined`, `auxiliary`, `fine_geneo`; '
        '`--geneo-threshold 0.5` amplía la selección. La formulación fina, las frecuencias '
        'y la tolerancia se conservan. Las decisiones se basan en forma/operador/coste, '
        'sin detección de fibras ni reglas por nombre de geometría.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,default=ROOT/'benchmarks/results/linear_strategy_20261002')
    p.add_argument('--evidence',type=Path)
    p.add_argument('--output',type=Path,default=Path(__file__).parent)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.evidence:
        with gzip.open(a.evidence,'rt') as stream:bundle=json.load(stream)
    else:bundle=collect(a.raw_root)
    for item in bundle['source_snapshot'].values():assert digest(item['text'].encode())==item['sha256']
    summary=summarize(bundle)
    path=a.output/'evidence.json.gz'
    # mtime=0 keeps the offline bundle deterministic.
    path.write_bytes(gzip.compress(encoded(bundle),mtime=0))
    (a.output/'summary.json').write_bytes(encoded(summary))
    write_csv(a.output/'timings.csv',summary['primary'],['case','current_s','default_s','saving_percent',
        'paired_saving_min','paired_saving_max','fft_policy','tuning_s','amortization_solves','max_physical_residual','Ceff_relative_difference'])
    write_csv(a.output/'pilots.csv',summary['pilots'],['label','directory','reference_s','median_s','iterations','setup_s','retained_gpu_bytes'])
    plot(summary,a.output);(a.output/'RESULTS.md').write_text(render(summary))
    artifacts=['evidence.json.gz','summary.json','timings.csv','pilots.csv','warm_savings.png','warm_savings.svg','RESULTS.md']
    manifest=dict(schema=1,baseline_commit=bundle['baseline_commit'],raw_root=bundle['raw_root'],
        bundled_json_files=len(bundle['files']),source_files=len(bundle['source_snapshot']),
        production_test_result='338 passed, 7 subtests passed',auxiliary_test_result='4 passed',
        primary_directories=PRIMARY,pilot_directories=[r[0] for r in PILOTS],
        source_sha256={path:item['sha256'] for path,item in bundle['source_snapshot'].items()},
        artifacts={name:dict(sha256=digest((a.output/name).read_bytes()),bytes=(a.output/name).stat().st_size) for name in artifacts})
    (a.output/'manifest.json').write_bytes(encoded(manifest))
    print(json.dumps(dict(primary_cases=len(summary['primary']),pilots=len(summary['pilots']),
        validation_reports=len(summary['validations']),bundle_bytes=path.stat().st_size)))


if __name__=='__main__':main()
