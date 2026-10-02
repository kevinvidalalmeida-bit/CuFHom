"""Summarize sampled Nsight Compute launches without changing baseline timings.

Read-only .ncu-rep imports need no administrator access. The compressed bundle
keeps CSV values, instruction instances and solve checks for portable rebuilds.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
from pathlib import Path


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    try:
        return float(value.replace(',', ''))
    except ValueError:
        return None


def normalize_kernel(name):
    # CUPTI preserves template argument casts that ncu's demangler omits.
    name = re.sub(r'\((unsigned int|padding_t|twiddle_t|loadstore_modifier_t|layout_t)\)', '', name)
    return re.sub(r'\s+', '', name)


def load_raw(directory, ncu_python):
    sys.path.insert(0, str(ncu_python))
    import ncu_report
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert len(manifest['jobs']) == 60, 'Incomplete campaign'
    records = {}
    for job in manifest['jobs']:
        assert job['returncode'] == 0 and job['report_exists'] and job['csv_export_ok'], job
        key = job['case'] + '_' + job['kernel']
        command = job['command']
        solve_path = Path(command[command.index('--output') + 1]) / 'report.json'
        files = {ext: directory / (key + ext) for ext in ('.csv', '.log', '.export.log', '.ncu-rep')}
        context = ncu_report.load_report(files['.ncu-rep'])
        assert context.num_ranges() == 1
        action = context.range_by_idx(0).action_by_idx(0)
        opcodes = {}
        for metric_name in ('sass__thread_inst_executed_true_per_opcode',
                            'sass__thread_inst_executed_true_per_opcode_with_modifier_all'):
            metric = action.metric_by_name(metric_name)
            assert metric is not None and metric.has_correlation_ids()
            ids = metric.correlation_ids()
            opcodes[metric_name] = {ids.value(i): metric.value(i) for i in range(metric.num_instances())}
        records[key] = dict(case=job['case'], tag=job['kernel'],
            csv=files['.csv'].read_text(), log=files['.log'].read_text(),
            export_log=files['.export.log'].read_text(), solve=json.loads(solve_path.read_text()),
            opcodes=opcodes, file_sha256={str(p): sha(p) for p in [*files.values(), solve_path]})
    return dict(manifest=manifest, records=records)


def summarize(bundle, trace, reference, require_external_inputs=False):
    rows = []
    for key, record in bundle['records'].items():
        report = record['solve']
        assert report['source_sha256'] == reference and report['inputs_unchanged'], key
        paths = {Path(path): digest for path, digest in report.get('input_sha256', {}).items()
                 if Path(path).is_absolute()}
        available = {path: digest for path, digest in paths.items() if path.is_file()}
        if require_external_inputs:
            assert len(available) == len(paths), key
        assert all(sha(path) == digest for path, digest in available.items()), key
        diagnostic = report['diagnostic']
        assert len(diagnostic['physical_infos']) == 6
        assert all(x['true_residual_converged'] and x['true_norm_res_rel'] <= 1e-5
                   for x in diagnostic['physical_infos']), key
        assert diagnostic['minimum_Ceff_eigenvalue'] > 0
        assert diagnostic['Ceff_relative_difference'] < 1e-6
        values = list(csv.reader(io.StringIO(record['csv'])))
        assert len(values) == 3 and len(values[0]) == len(values[1]) == len(values[2])
        raw = dict(zip(values[0], values[2]))
        metric = {k: number(v) for k, v in raw.items()}
        get = lambda name: metric[name]
        seconds = get('gpu__time_duration.sum') / 1e9
        prefix = 'smsp__pcsamp_warps_issue_stalled_'
        stalls = {k[len(prefix):]: v for k, v in metric.items()
                  if k.startswith(prefix) and not k.endswith('_not_issued')}
        samples = get('smsp__pcsamp_sample_count')
        assert sum(stalls.values()) == samples
        ordered = sorted(stalls, key=stalls.get, reverse=True)
        occupancy = get('sm__warps_active.avg.pct_of_peak_sustained_active')
        l2 = get('lts__t_sector_hit_rate.pct')
        flags = []
        if seconds < 20e-6:
            flags.append('short_launch_below_20us')
        if occupancy > 100 or l2 > 100:
            flags.append('raw_percentage_out_of_range')
        registers = int(get('launch__registers_per_thread'))
        variant = record['tag']
        if variant in ('update', 'update_late'):
            variant = {34: 'normal', 36: 'smoothing'}.get(registers, 'unclassified')
        if variant == 'coordinates':
            variant = {38: 'map', 40: 'map_with_deferred_direction'}.get(registers, 'unclassified')
        counts = record['opcodes']['sass__thread_inst_executed_true_per_opcode']
        flops32 = 2 * counts.get('FFMA', 0) + counts.get('FADD', 0) + counts.get('FMUL', 0)
        flops64 = 2 * counts.get('DFMA', 0) + counts.get('DADD', 0) + counts.get('DMUL', 0)
        dram = get('dram__bytes.sum.per_second')
        memory_pct = get('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed')
        fp64_pct = get('sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_elapsed')
        interpretation = 'memory throughput evidence' if memory_pct >= 75 else 'latency/throughput unresolved'
        if fp64_pct >= 50:
            interpretation += '; high FP64 pipeline use'
        if flags:
            interpretation += '; interpret flagged metrics cautiously'
        row = dict(sample_id=key, case=record['case'], requested_tag=record['tag'],
            observed_variant=variant, kernel=raw['Kernel Name'], block=raw['Block Size'], grid=raw['Grid Size'],
            registers_per_thread=registers, registers_allocated=get('launch__registers_per_thread_allocated'),
            shared_memory_bytes=get('launch__shared_mem_per_block'),
            sampled_duration_us=seconds * 1e6, DRAM_GB_s=dram / 1e9,
            DRAM_bytes_reconstructed_rate_times_duration=dram * seconds,
            DRAM_bytes_basis='derived rate x duration; direct dram__bytes.sum not collected; replay/global traffic',
            achieved_occupancy_raw_pct=occupancy, L2_hit_rate_raw_pct=l2,
            memory_peak_raw_pct=memory_pct, SM_peak_raw_pct=get('sm__throughput.avg.pct_of_peak_sustained_elapsed'),
            FP64_pipe_peak_raw_pct=fp64_pct,
            FP32_pipe_peak_raw_pct=get('sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed'),
            eligible_warps_per_scheduler=get('smsp__warps_eligible.avg.per_cycle_active'),
            issue_active_raw_pct=get('smsp__issue_active.avg.pct_of_peak_sustained_active'),
            register_spill_instructions=get('sass__inst_executed_register_spilling'),
            sampled_warp_states=samples, dominant_warp_state=ordered[0],
            dominant_warp_state_share_pct=stalls[ordered[0]] / samples * 100 if samples else '',
            warp_state_share_json=json.dumps({k: v / samples * 100 for k, v in stalls.items()}, separators=(',', ':')) if samples else '{}',
            ordinary_FP32_FLOPs=flops32, ordinary_FP64_FLOPs=flops64,
            ordinary_FP32_TFLOP_s=flops32 / seconds / 1e12, ordinary_FP64_TFLOP_s=flops64 / seconds / 1e12,
            ordinary_arithmetic_intensity_reconstructed=(flops32 + flops64) / (dram * seconds),
            FLOPs_basis='predicated thread FFMA/DFMA x2 + FADD/DADD + FMUL/DMUL; excludes MUFU, packed half, shuffles, integer ops',
            HFMA2_thread_instructions=counts.get('HFMA2', 0), MUFU_thread_instructions=counts.get('MUFU', 0),
            physical_residual=diagnostic['max_residual'], Ceff_relative_difference=diagnostic['Ceff_relative_difference'],
            minimum_Ceff_eigenvalue=diagnostic['minimum_Ceff_eigenvalue'], iterations=diagnostic['iterations'],
            external_input_files_verified=len(available), external_input_files_unavailable=len(paths)-len(available),
            quality_flags=';'.join(flags), interpretation=interpretation,
            sample_scope='one sampled warm launch; kernel replay; cache/clock control none; not a timing baseline')
        case = {'fiber192_auto': 'sfrp_vf15_ar10_N192',
                'fiber192_component': 'component_sfrp_vf15_ar10_N192'}.get(row['case'], row['case'])
        candidates = [k for k in trace[case]['kernels'] if normalize_kernel(k['kernel']) == normalize_kernel(row['kernel'])]
        group = 'postprocess_kernels' if record['tag'] == 'postprocess_dot' else None
        variants = [v for k in candidates for v in k['variants']
                    if v['registers'] == registers and (group is None or v['group'] == group)
                    and (record['tag'] != 'warp_dot' or v['group'] != 'postprocess_kernels')]
        row['CUPTI_matched_variant_calls'] = sum(v['calls'] for v in variants)
        row['CUPTI_matched_variant_mean_us'] = (sum(v['mean_us'] * v['calls'] for v in variants) /
            row['CUPTI_matched_variant_calls']) if variants else ''
        rows.append(row)
    return rows


def attach_counters(kernel_rows, samples):
    """Attach one named sample, never pretend it is the whole kernel average."""
    for row in kernel_rows:
        case = 'fiber192_auto' if row['case'] == 'sfrp_vf15_ar10_N192' else row['case']
        matched = [s for s in samples if s['case'] == case and
                   normalize_kernel(s['kernel']) == normalize_kernel(row['kernel'])]
        row['counter_sample_ids'] = ';'.join(s['sample_id'] for s in matched)
        row['counter_details'] = 'gpu_counters.csv'
        row['counters_status'] = 'sampled launch/variant; see counter_sample_ids' if matched else 'not sampled in this campaign'
        if not matched:
            continue
        sample = matched[0]
        row['primary_counter_sample'] = sample['sample_id']
        row['counter_scope'] = sample['sample_scope']
        row['sample_interpretation'] = sample['interpretation']
        if 'fft' in row['kernel']:
            row['basis'] = '1D reference FFT operation/I-O model; sampled ordinary SASS counts and rate-derived traffic are separate in gpu_counters.csv'
            row['classification'] = 'static FFT bandwidth/layout/latency hypothesis; compare named counter sample and full-solve timings'
        row['achieved_occupancy'] = sample['achieved_occupancy_raw_pct']
        row['DRAM_throughput'] = sample['DRAM_GB_s']
        row['L2_hit_rate'] = sample['L2_hit_rate_raw_pct']
        row['dominant_stalls'] = sample['dominant_warp_state']
        row['sample_DRAM_bytes_reconstructed'] = sample['DRAM_bytes_reconstructed_rate_times_duration']
        row['sample_quality_flags'] = sample['quality_flags']
        row['counter_units'] = 'occupancy/hit pct; DRAM throughput GB/s; reconstructed traffic bytes'
        # DRAM_bytes and warp_efficiency remain blank: neither was directly measured.


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def plot_counters(output, rows, calibration):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = [r for r in rows if r['case'] == 'G09']
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, precision, roof in zip(axes, ('FP32', 'FP64'),
            (calibration['arithmetic']['float']['TFLOP_s'], calibration['arithmetic']['double']['TFLOP_s'])):
        points = [(r, r['ordinary_' + precision + '_FLOPs'] /
                   r['DRAM_bytes_reconstructed_rate_times_duration']) for r in selected
                  if r['ordinary_' + precision + '_FLOPs'] > 0]
        intensities = [x for _, x in points]
        domain = np.geomspace(min(intensities) / 2, max(intensities) * 6, 200)
        # A bus specification and a standalone FMA calibration are guides, not
        # a certified bound for this partial instruction mix/replayed traffic.
        ax.loglog(domain, np.minimum(0.896 * domain, roof), '--', color='0.45',
                  label='Guía: bus 896 GB/s + FMA calibrada')
        offsets = {'material': (7, 12), 'adjoint_dot': (7, -12), 'coordinates': (-20, -26),
                   'update': (5, -18), 'update_late': (-48, -15), 'postprocess_dot': (5, 12),
                   'warp_dot': (5, -12), 'regular_fft': (6, -15), 'r2c': (6, 12), 'c2r': (6, -15)}
        for index, (row, intensity) in enumerate(points):
            throughput = row['ordinary_' + precision + '_TFLOP_s']
            ax.scatter(intensity, throughput, s=28, color='#176fa6')
            ax.annotate(row['requested_tag'], (intensity, throughput),
                        xytext=offsets[row['requested_tag']] if precision == 'FP32' else
                            (5, 5 if index % 2 else -13), textcoords='offset points', fontsize=7)
        ax.set_title(precision + ': FLOPs ordinarios contados')
        ax.set_xlabel('FLOPs / byte DRAM reconstruido')
        ax.set_ylabel('TFLOP/s ordinarios')
        ax.grid(True, which='both', alpha=.2)
        ax.legend(fontsize=7, loc='upper left')
    figure.suptitle('G09: contadores por lanzamiento y guías roofline')
    figure.text(.5, .015, 'Kernel replay; caché/relojes sin controlar. Bytes = tasa × duración; conteo parcial de FLOPs. No certifica un límite global.',
                ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .05, 1, .95))
    figure.savefig(output / 'counter_roofline.png', dpi=180)
    figure.savefig(output / 'counter_roofline.svg')
    plt.close(figure)
    path = output / 'counter_roofline.svg'
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')


def write_report(output, rows):
    selected = {r['requested_tag']: r for r in rows if r['case'] == 'G09'}
    table = markdown_table(['Kernel G09', 'GB/s DRAM', 'Ocupación %', 'L2 hit %', 'FP64 pico %', 'Estado predominante / % muestras'],
        [[tag, f"{r['DRAM_GB_s']:.1f}", f"{r['achieved_occupancy_raw_pct']:.1f}", f"{r['L2_hit_rate_raw_pct']:.1f}",
          f"{r['FP64_pipe_peak_raw_pct']:.1f}", f"{r['dominant_warp_state']} / {r['dominant_warp_state_share_pct']:.1f}"]
         for tag, r in selected.items()])
    fft_rows = []
    for tag in ('regular_fft', 'r2c', 'c2r'):
        pair = [next(r for r in rows if r['case'] == 'fiber192_' + policy and r['requested_tag'] == tag)
                for policy in ('auto', 'component')]
        fft_rows.append([tag, *[f"{r['CUPTI_matched_variant_calls']} / {r['CUPTI_matched_variant_mean_us']:.2f}" for r in pair],
                        *[f"{r['DRAM_bytes_reconstructed_rate_times_duration']/1e6:.1f}" for r in pair]])
    flagged = [r for r in rows if r['quality_flags']]
    text = f'''# Contadores GPU de CuFHom

Fecha: 1 de octubre de 2026. **60 capturas y 60 exportaciones completas**, obtenidas con una ejecución
administrativa puntual, sin reiniciar ni cambiar la configuración del controlador. El acceso normal sigue
sujeto a `RmProfilingAdminOnly: 1`. Se corrigió la exportación: esta versión usa `--print-units base`.
El archivo `.ncu-rep` anterior se conservó y se exportó sin repetir esa captura.

## Resultado

En G09 los mapas, Cε y updates alcanzan **747–798 GB/s**, con ocupación observada de **81–99%**.
Su estado predominante es `long_scoreboard`: esperan dependencias de operaciones de memoria.
Esta evidencia respalda priorizar reducir tráfico y aplicaciones de H. Aumentar ocupación por sí solo
ofrece poco margen en esos kernels. Los FFT muestreados alcanzan **719–757 GB/s**.
Las reducciones `warp_dot` alcanzan **84–88% del pico de la tubería FP64**; el dot final también
alcanza 841 GB/s. Tienen costes mixtos de memoria, instrucciones y reducción.

{table}

Son porcentajes de los recursos indicados y estados de warps muestreados. El porcentaje de un estado
**no es el porcentaje de pared recuperable**. No se debe sumar con los tiempos de Nsight Systems.
Los 60 lanzamientos reportan cero instrucciones de spill de registros; eso describe los lanzamientos
muestreados y no certifica todos los kernels del programa.

## FFT por componentes, 192³

{markdown_table(['Familia', 'Auto: llamadas / media CUPTI µs', 'Component: llamadas / media CUPTI µs',
                'Auto: MB reconstruidos/lanzamiento', 'Component: MB reconstruidos/lanzamiento'], fft_rows)}

El lanzamiento por componentes trabaja sobre una porción menor y hay más llamadas. Su duración
individual no representa el ahorro global. La evidencia comparable sigue siendo la campaña alternada:
**1,561 → 1,190 s, 23,77% de ahorro**, seis cargas y campos/Ceff idénticos.
La traza completa reduce el tiempo FFT exclusivo de **795,2 a 473,7 ms**. La ocupación del `regular_fft`
muestreado baja de 63,4% a 26,4% y aun así mejora el solve completo. Conviene elegir el plan por el
tiempo total medido. [Estadística y validación](experiment_statistics.csv).

## Alcance y calidad de la medición

- Una instancia warm por etiqueta y caso: G08, G09, caso general 64³/256³ y fibras 192³ con auto/component.
  Se conservaron las seis cargas y el criterio físico original. Todos los residuales son ≤1e-5 y Ceff es SPD.
- `kernel` replay, `cache-control none`, `clock-control none`, nueve secciones de Nsight Compute.
  El perfilador advierte que cachés y relojes sin controlar pueden producir inconsistencias. Los valores
  describen esta captura; el replay no conserva exactamente la caché de la ejecución completa.
- Los contadores DRAM/L2 son recursos compartidos con otras actividades del dispositivo. En kernels cortos
  la actividad del escritorio puede pesar mucho. Hay **{len(flagged)} muestras marcadas** por duración <20 µs
  o porcentajes fuera de rango. Por ejemplo G08 r2c reporta ocupación 106,96%; se conserva el valor original
  y se marca. No se transforma en una eficiencia física ni se recorta a 100%.
- `update_late` es una etiqueta de selección por índice, no una prueba de variante. Los registros muestran
  que en G08/64³ ambas muestras son el update normal (34 registros). En G09/256³/192³ sí se capturó
  suavizado (36 registros). El mapa muestreado tiene 38 registros; no se capturó el mapa fusionado de 40.

NVIDIA documenta el replay, la interferencia de otras actividades, los porcentajes fuera de rango y las
diferencias de duración entre herramientas en su
[guía de Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

## Roofline: datos medidos y datos derivados

`DRAM_GB_s` procede de `dram__bytes.sum.per_second`. No se recogió `dram__bytes.sum` directamente:
los bytes de esta tabla se **reconstruyen como tasa × duración**. Se identifican como derivados y pueden
mezclar variación entre pases y tráfico de otros motores. Los FLOPs ordinarios se calculan con las
instancias de instrucciones por hilo predicadas: FFMA/DFMA cuentan dos, FADD/DADD y FMUL/DMUL cuentan uno.
MUFU, shuffles, enteros y operaciones empaquetadas HFMA2 se guardan aparte y quedan fuera de esa suma.
Es un conteo parcial de trabajo aritmético, no un modelo completo de todas las tuberías.

![Contadores por lanzamiento y guías roofline](counter_roofline.png)

La guía usa el bus teórico de 896 GB/s y la calibración FMA independiente. Cada panel cuenta solo
aritmética ordinaria de esa precisión; el kernel puede estar limitado por otras instrucciones o recursos.

[gpu_counters.csv](gpu_counters.csv) distingue métricas, bytes reconstruidos, instrucciones, stalls,
variante observada y banderas de calidad. [kernel_profile.csv](kernel_profile.csv) enlaza muestras con
el kernel de la traza; las columnas de contadores describen la primera muestra nombrada en cada fila,
no la media de todas sus llamadas. Los tiempos e hipótesis originales de esa tabla se conservaron.
`DRAM_bytes` y `warp_efficiency` permanecen vacíos porque no se midieron directamente.

Esta campaña resuelve el acceso a contadores y fortalece el diagnóstico. **No certifica un mínimo hardware
global**: faltan cobertura de todas las variantes, control de caché/reloj y repeticiones de contadores.
Los escenarios roofline anteriores siguen siendo condicionados. La ganancia de FFT por componentes
proviene de los tiempos alternados sin Nsight Compute.

## Siguientes cambios que justifica la evidencia

1. Evaluar una política general de planes FFT por forma, dtype, dispositivo y tiempo del solve; las cifras
   de 96³ impiden escoger component para todas las mallas.
2. Probar un precondicionador SPD que reduzca aplicaciones de H y cuyo ahorro supere setup/aplicación.
   La base adaptativa propuesta en el [informe principal](PERFORMANCE_AUDIT.md) sigue pendiente de validación.
3. Medir si una nueva fusión elimina lecturas/escrituras de campos grandes sin repetir fusiones existentes.
   Los kernels cerca del techo de memoria necesitan reducir bytes, no solo instrucciones de launch.
4. Revisar la reducción FP64 con el mismo criterio físico y precisión. Su tiempo exclusivo y Amdahl delimitan
   el beneficio posible; no se justifica cambiar precisión con estos contadores.

## Evidencia y reproducción

El bundle [counter_evidence.json.gz](counter_evidence.json.gz) conserva CSV crudos con unidades, logs,
instancias por opcode, informes de las seis cargas y hashes de `.ncu-rep`/entradas/código.
[counter_manifest.json](counter_manifest.json) resume procedencia y verificaciones.
Los 60 `.ncu-rep` permanecen en `benchmarks/results/ncu_counters_20261001_7saBZK/`.

Recrear las tablas desde el bundle no requiere sudo ni ejecutar el solver:

```bash
.venv/bin/python benchmarks/build_gpu_counters.py \\
  --evidence benchmarks/performance_audit_20261001/counter_evidence.json.gz \\
  --output benchmarks/performance_audit_20261001
```

Para una nueva captura, `sudo bash benchmarks/profile_gpu_counters.sh` solicita la contraseña en la terminal.
`--resume DIRECTORIO` conserva capturas previas; la exportación offline usa `--export-only` en el script Python.
'''
    (output / 'GPU_COUNTERS.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path)
    source.add_argument('--evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ncu-python', type=Path, default=Path('/opt/nvidia/nsight-compute/2025.3.1/extras/python'))
    args = parser.parse_args()
    if args.input:
        bundle = load_raw(args.input, args.ncu_python)
        target = args.output / 'counter_evidence.json.gz'
        with target.open('wb') as f:
            with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as stream:
                stream.write(json.dumps(bundle, separators=(',', ':'), allow_nan=False).encode())
    else:
        target = args.evidence
        with gzip.open(target, 'rt') as stream:
            bundle = json.load(stream)
    with gzip.open(args.output / 'evidence.json.gz', 'rt') as stream:
        evidence = json.load(stream)
    reference = evidence['G09/report.json']['source_sha256']
    root = Path(__file__).resolve().parents[1]
    assert all(sha(root / path) == digest for path, digest in reference.items()), 'Production changed'
    trace = dict(evidence['trace_summary.json'])
    trace.update(evidence['fft_trace_summary.json'])
    rows = summarize(bundle, trace, reference, require_external_inputs=bool(args.input))
    write_csv(args.output / 'gpu_counters.csv', rows)
    plot_counters(args.output, rows, evidence['calibration_with_cache.json'])
    kernels = list(csv.DictReader((args.output / 'kernel_profile.csv').open()))
    attach_counters(kernels, rows)
    write_csv(args.output / 'kernel_profile.csv', kernels)
    write_report(args.output, rows)
    manifest = dict(captures=len(rows), exports=len(rows), physical_checks_passed=len(rows),
        source_commit=bundle['manifest']['source_commit'], production_unchanged=True,
        production_sha256=reference, bundle_sha256=sha(target),
        flags=sum(bool(r['quality_flags']) for r in rows), max_physical_residual=max(r['physical_residual'] for r in rows),
        max_Ceff_relative_difference=max(r['Ceff_relative_difference'] for r in rows),
        profiler=dict(replay_mode='kernel', cache_control='none', clock_control='none',
                      sections=bundle['manifest']['sections']),
        status='administrator capture succeeded; normal-user driver restriction unchanged',
        scope='one sampled warm launch per tag; not a statistical baseline or certified hardware ceiling')
    (args.output / 'counter_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Verified captures:', len(rows), '; marked samples:', manifest['flags'])


if __name__ == '__main__':
    main()
