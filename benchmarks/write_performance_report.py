"""Write the audit diagnosis from measured CSV/JSON, with explicit model limits."""
import argparse
import csv
import json
from pathlib import Path


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    read=lambda name:list(csv.DictReader((a.output/name).open()))
    baseline=read('performance_baseline.csv');fiber=read('fiber_baseline.csv')
    experiments=read('experiment_statistics.csv');fields=read('field_validation.csv')
    breakdown=read('component_breakdown.csv');fits=read('scaling_fits.csv')
    models=json.loads((a.output/'models.json').read_text())['models']
    trace=json.loads((a.input/'trace_summary.json').read_text())
    b={r['case']:r for r in baseline};g09=models['G09'];cal=json.loads((a.input/'calibration_with_cache.json').read_text())
    g09wall=trace['G09']['wall_ms'];g09fft=trace['G09']['exclusive_ms']['operator_fft']
    idle=sum(v for k,v in trace['G09']['exclusive_ms'].items() if k.startswith('idle_'))
    primary=[next(r for r in fiber if r['suite']=='sfrp_vf15_ar10' and int(r['N'])==n) for n in (32,48,64,96,128,192,256)]
    fixed=[]
    for vf in (.05,.15,.25):
        for ar in (5,10,20):
            rs=[next(r for r in fiber if float(r['Vf_target'])==vf and float(r['AR'])==ar and int(r['N'])==n) for n in (96,192,256)]
            fixed.append([f'{vf:.0%}',ar,*[f"{float(r['median_ms'])/1000:.4f} / {r['iterations']}" for r in rs]])
    main_table=table(['N','Vóxeles','Mediana ms','Media ms','DE ms','Mínimo ms','P95 ms','CG coord. + corrección','CG ms/iter.'],
        [[r['N'],int(r['voxels']),*[f'{float(r[k]):.3f}' for k in ('median_ms','mean_ms','std_ms','minimum_ms','p95_ms')],
            f"{r['coordinate_iterations']} + {r['correction_iterations']}",f"{float(r['inner_cg_per_coordinate_iteration_ms']):.4f}"] for r in primary])
    comp_table=table(['Componente','ms exclusivos','% pared','Llamadas kernel','µs kernel/llamada'],
        [[r['component'],f"{float(r['exclusive_ms']):.3f}",f"{100*float(r['fraction_wall']):.2f}",r['calls'],
            f"{float(r['time_per_call_ms'])*1000:.2f}" if r['time_per_call_ms'] else '—']
            for r in breakdown if r['case']=='G09'])
    fft_rows=[]
    for n in (96,128,192,256):
        rs={r['variant']:r for r in experiments if r['experiment']==f'fft_sfrp_vf15_ar10_N{n}'}
        v=rs['component'];fft_rows.append([n,f"{float(rs['auto']['median_ms'])/1000:.4f}",
            f"{float(v['median_ms'])/1000:.4f}",f"{float(v['median_time_saving_pct']):.2f}%",
            f"{float(v['ratio_of_median_times']):.3f}×",v['iterations']])
    dual_rows=[]
    for name in ('g08_dual_metric','g09_dual_metric','dual_sfrp_vf25_ar12.5_N256',
            'dual_sfrp_vf05_ar10_N128','dual_sfrp_vf05_ar10_N256','accuracy_dual_sfrp_vf05_ar10_N256'):
        rs={r['variant']:r for r in experiments if r['experiment']==name}
        for variant in ('dual_metric','dual_strict_half_rtol'):
            if variant not in rs:continue
            v=rs[variant];dual_rows.append([name.replace('sfrp_',''),variant,
                f"{float(rs['current']['median_ms'])/1000:.4f} → {float(v['median_ms'])/1000:.4f}",
                f"{rs['current']['iterations']} → {v['iterations']}",f"{float(v['median_time_saving_pct']):.2f}%",
                f"{float(v['max_physical_residual']):.3g}"])
    amdahl=read('amdahl.csv')
    amdahl_table=table(['Componente G09','2×','5×','10×','Coste cero'],
        [[component,*[f"{float(next(r['global_speedup'] for r in amdahl if r['case']=='G09' and r['component']==component and r['component_speedup']==speed)):.3f}×"
            for speed in ('2','5','10','infinite')]] for component in ('cuFFT','largest_kernel','solver_update','explicit_reductions','postprocess')])
    setup_table=table(['Caso','Importar ms','Contexto ms','Leer entradas ms','Compilar geometría ms','Primer solve ms','Warm ms'],
        [[name,*[f'{float(b[name][key]):.2f}' for key in ('import_ms','context_ms','geometry_read_or_generation_ms',
            'geometry_compile_ms','first_solve_existing_caches_ms','median_ms')]]
            for name in ('sfrp_vf15_ar10_N32','sfrp_vf15_ar10_N192','sfrp_vf15_ar10_N256','G09')])
    pvalues={r['quantity']:r for r in fits if r['suite']=='sfrp_vf15_ar10' and r['subset']=='N>=96'}
    native=[r for r in fiber if r['native_resolution_6']=='True']
    native_table=table(['Vf','AR','N a 6 vóxeles/µm','Tiempo s','CG'],
        [[f"{float(r['Vf_target']):.0%}",r['AR'],r['N'],f"{float(r['median_ms'])/1000:.4f}",r['iterations']]
            for r in sorted(native,key=lambda r:(float(r['Vf_target']),float(r['AR'])))])
    half=next(r for r in read('ideal_solver_scenarios.csv') if r['case']=='G09' and r['scenario']=='CG_iterations_reduced_50pct')
    strict=next((r for r in experiments if r['experiment']=='accuracy_dual_sfrp_vf05_ar10_N256' and
        r['variant']=='dual_strict_half_rtol'),None)
    strict_text=('El control con residual físico ≤5e-6 obtuvo '
        f"{float(strict['median_ms'])/1000:.4f} s y {strict['iterations']} iteraciones, "
        f"un cambio de tiempo de {float(strict['median_time_saving_pct']):.2f}% (positivo = ahorro) frente a su referencia alternada. "
        'Esta fila separa el cambio de método del beneficio de dejar de resolver por debajo de la tolerancia pública.'
        if strict else 'El control con residual físico ≤5e-6 todavía no tiene un resultado completo.')
    counter_path=a.output/'counter_manifest.json'
    counters=json.loads(counter_path.read_text()) if counter_path.exists() else None
    counter_status=('**Se obtuvieron 60 capturas completas de Nsight Compute con una ejecución administrativa puntual, sin reiniciar.** '
        'En G09 los mapas/updates alcanzan 747–798 GB/s de DRAM y 81–99% de ocupación; las reducciones muestran '
        'uso alto de FP64. El acceso normal sigue restringido por el controlador. '
        '[Contadores y límites de interpretación](GPU_COUNTERS.md). '
        '**No se ha demostrado un límite físico global:** son muestras por lanzamiento con replay y caché/relojes sin controlar.'
        if counters else '**No se ha demostrado un límite físico global.** Nsight Compute devuelve `ERR_NVGPUCTRPERM` '
        'para el usuario normal; los modelos de roofline son escenarios condicionados.')
    counter_classification=('Los contadores muestreados ahora respaldan el coste de memoria de los mapas/updates grandes '
        'y el uso alto de FP64 en reducciones. Los datos y banderas de calidad están en [gpu_counters.csv](gpu_counters.csv); '
        'no cubren todas las variantes ni certifican un techo global.' if counters else
        'Las clasificaciones de bandwidth/latencia/FP64 son hipótesis respaldadas por tiempos e intensidad lógica; '
        'faltan contadores para confirmarlas.')
    counter_registers=('Las columnas de contadores en `kernel_profile.csv` describen la muestra nombrada por '
        '`primary_counter_sample`, con unidades y alcance explícitos. Los valores fuera de rango se conservan y marcan. '
        '`DRAM_bytes` y `warp_efficiency` siguen vacíos: no se recogieron directamente. '
        'El tráfico reconstruido se guarda por separado y no se mezcla con los bytes lógicos del modelo.' if counters else
        'Los registros/memoria local estática de CUPTI no certifican ocupación, stalls ni ancho de banda; '
        'las columnas correspondientes quedan vacías con el motivo del bloqueo.')
    counter_conclusion=('La nueva captura resuelve el acceso a contadores y orienta las siguientes mejoras. '
        'Todavía faltan cobertura de variantes y control/repeticiones para estimar un límite hardware global. '
        '[Informe de contadores](GPU_COUNTERS.md).' if counters else
        'Queda pendiente medir contadores Nsight Compute; esta entrega conserva esa limitación explícita.')
    report=fr'''# Auditoría de rendimiento de CuFHom

Fecha: 1 de octubre de 2026. GPU: RTX 5070 Ti de 16 GB. Referencia de producción:
`92c9ac5b8204288244e82536b254078d6d767e66`. **El código de producción y sus opciones predeterminadas no se modificaron.**

## Diagnóstico principal

Hay margen de implementación demostrado: con la misma geometría de fibras, seis cargas, precisión y tolerancia,
elegir FFT por componentes reduce **192³ de 1,561 a 1,190 s**, aproximadamente **24%** o **1,31×**.
En 128³ ahorra alrededor del 22%; en 96³ empeora. El plan de cuFFT debe elegirse por dimensiones,
precisión y GPU; hacerlo depende del coste de las transformadas y funciona para materiales generales.

También hay un coste algorítmico concreto: algunas geometrías activan CG de corrección en deformaciones
después de terminar CG en coordenadas. En Vf=5%, AR=10, 256³ son **48 iteraciones y 827 ms adicionales**.
La aceptación con dos normas evita parte de ese trabajo a la tolerancia pública, pero **el control de precisión
comparable pierde la mejora**. Ese resultado descarta promoverla como aceleración general.

{counter_status}

## 1. Diseño de la campaña y controles

- **12 geometrías continuas, 96 pares geometría/malla y 864 solves warm de seis cargas** en el barrido de fibras.
  Los nueve casos acordados usan Vf={{5,15,25}}% y AR={{5,10,20}}; los tres AR=12,5 conservan evidencia de la campaña anterior.
- Para cada master: N={{32,48,64,96,128,192,256}}, más la malla que representa 6 vóxeles/µm.
  El diámetro es 1 µm; L=AR µm y el dominio cúbico periódico tiene lado 2L.
- L es la longitud del eje de la cápsula usada por el generador. La distancia entre puntas es L+d.
  El dominio corresponde a 2L en esa convención; las tapas se incluyen al calcular Vf.
- Se conservan centros, orientaciones y fibras al cambiar N. Los hashes de los masters y de las entradas
  se guardaron. Se regeneraron 12 rasters de 32³: **las 24 matrices phase/ori son idénticas por SHA256**.
- Orientación aleatoria y A2 medido, con una realización por Vf/AR. Repetir nueve veces un solve mide
  variación temporal; no representa nueve realizaciones de microestructura. No se infiere una ley universal de Vf/AR.
- Auditoría periódica independiente de imágenes vecinas y autoimágenes. Se mantuvo la tolerancia histórica
  de contacto relativo 1/120; no se afirma ausencia estricta de solapamiento. G04 requirió eliminar una fibra
  de la **copia local**: Vf continuo quedó en 0,149635. Los archivos originales permanecen intactos.
- Se guarda Vf voxelizado, A2 y diámetro en vóxeles. D<2 vóxeles se marca como subresuelto; esas filas sirven
  para medir ejecución, no para demostrar convergencia geométrica.
- El solve usa una celda unidad equivalente por cambio de escala. La elasticidad lineal local periódica
  aquí no introduce una longitud interna; se conserva la proporción geométrica y Ceff. Las unidades físicas
  originales del dominio están en los metadatos.

Todas las filas de fibras usan el mismo material de G09: Em=4,338421, nu_m=0,354302,
Ef_L=347,979995, Ef_T=11,339842, nu_LT=0,242344, nu_TT=0,393962 y G_LT=13,605780.
Campos y FFT FP32; reducciones escalares FP64; residual físico relativo **1e-5**, absoluto 0;
máximo 1000 iteraciones. Los seis residuos verdaderos deben aprobar antes de registrar un resultado.

Cada par se ejecutó en un proceso nuevo: primer solve, warm-up, nueve repeticiones y un diagnóstico separado.
La referencia incluye preparar el material, resolver seis cargas, Ceff y escribir el JSON de tiempos.
Leer/compilar geometría, importar Python y crear el contexto CUDA se registran por separado.
El primer solve aprovecha cachés existentes en disco; no representa un equipo sin cachés.
Los candidatos se intercalan y rotan para reducir sesgo por deriva temporal.

## 2. Baseline estadístico y tendencia de fibras

Caso central Vf=15%, AR=10; las iteraciones se suman sobre seis cargas:

{main_table}

El tiempo por iteración procede de **nueve diagnósticos sincronizados separados** para este caso central.
El coste de CG de corrección se mide por separado; no se incluye en el divisor de las iteraciones en coordenadas.
Otras geometrías usan un diagnóstico por punto, excepto los dos casos con corrección más cara, que también tienen nueve.
[Estadística del CG](iteration_statistics.csv), [baseline completo](performance_baseline.csv),
[fibras](fiber_baseline.csv) y [etapas](stage_breakdown.csv).

Comparación con el mismo número de vóxeles: cada celda indica **segundos / iteraciones**.
Dentro de cada columna cambia la resolución física del diámetro con AR.

{table(['Vf','AR','96³','192³','256³'],fixed)}

![Barrido de fibras](fiber_scaling.png)

![Comparación a igual N](fiber_fixed_grid.png)

La comparación que conserva **6 vóxeles/µm y 6 vóxeles por diámetro** es distinta:
al crecer AR aumenta L, el dominio 2L y N. Estos puntos se excluyen de los ajustes de refinamiento uniforme.

{native_table}

![Resolución física fija](fiber_fixed_resolution.png)

## 3. Formulación actual, coste por iteración y número de iteraciones

El sistema se resuelve en tres componentes de coordenadas espectrales admisibles:

$$H z=b,\qquad H=R Q^* C Q R.$$

Aquí Q representa la deformación compatible en Mandel y C la elasticidad voxelizada. R contiene la raíz inversa
de la referencia isotrópica: factor transversal $(2G_0)^{{-1/2}}$ y longitudinal
$(K_0+4G_0/3)^{{-1/2}}$. Las coordenadas incorporan esa transformación simétrica;
CG opera sobre un sistema SPD. Se mantienen las restricciones originales de media y planos Nyquist.
La representación RFFT pondera las frecuencias y usa el producto escalar real con reducciones FP64.

Por paso: p → deformación → FFT inversa → C(x)ε → FFT → adjunto → p.Hp;
después α, actualización x/r y r.r, β y actualización diferida de p. Ya están fusionadas la actualización
de dirección con el mapa de deformación, el adjunto con p.Hp y la actualización x/r con r.r.
El suavizado tardío ya existente exige comprobar el residual verdadero; se conserva en todos los controles.

La auditoría física independiente vuelve a evaluar la ecuación original. Si falla, la referencia ejecuta
CG en deformaciones con rtol interno de 0,5×rtol. Por eso puede haber saltos de ocho iteraciones por carga.
El criterio público siempre se conserva en 1e-5; el número total incluye ambas fases.

**Una malla pequeña no implica pocas iteraciones.** Importan el espectro de H, el contraste,
las orientaciones, el RHS, la discretización y el salto a corrección. Las Ritz estimadas de G08 y G09
tienen razones aproximadamente 60–63 y 65,7–65,8. Son estimaciones Krylov, no cotas certificadas del espectro.
G09 contiene 64× los vóxeles de G08: un H pasa de 94,6 µs a 7,49 ms, mientras el total de iteraciones
solo pasa de 256 a 276. Eso separa claramente cantidad de trabajo y dificultad de convergencia.

La idea de que un precondicionador de referencia puede evitar crecimiento fuerte de iteraciones con el refinamiento
también aparece en [Ladecký y colaboradores](https://arxiv.org/html/2203.02962v1).
Ese trabajo usa una discretización FE distinta; no se trasladan sus cotas al operador actual.

En el caso central, ajustando N≥96, se obtiene experimentalmente:

{table(['Cantidad','p en aN^p','R² log','Puntos'],[[key,f"{float(pvalues[key]['p']):.3f}",f"{float(pvalues[key]['R2_log']):.4f}",pvalues[key]['points']]
    for key in ('median_total_ms','diagnostic_per_iteration_ms','six_load_solves_ms','postprocess_ms')])}

Son ajustes de intervalo finito, con cambio de régimen de caché y planes FFT; no una prueba de complejidad asintótica.
El ajuste de toda la serie mezcla latencia dominante en mallas pequeñas con tráfico/cálculo en grandes.
[Todos los ajustes](scaling_fits.csv) y [DOF/memoria](scaling_results.csv).

## 4. Perfil temporal completo y costes fuera del solver

G09 240³: baseline warm **{float(b['G09']['median_ms']):.3f} ms**; traza independiente **{g09wall:.3f} ms**.
Se da prioridad a actividad GPU, luego al tiempo API cuando la GPU está ociosa y luego a los huecos de CPU.
Cada instante pertenece a una sola categoría; las ocho categorías suman el total.

{comp_table}

Las llamadas y tiempos por llamada de esta tabla corresponden a **kernels GPU**. Las partes CPU de una categoría
no se dividen entre esas llamadas. Las transferencias, memset y API se cuentan individualmente en
[cuda_memory_profile.csv](cuda_memory_profile.csv) y [cuda_api_profile.csv](cuda_api_profile.csv).
[Kernel por kernel](kernel_profile.csv) contiene su coste absoluto, llamadas y duración media.

![Descomposición exclusiva](breakdown.png)

Las FFT consumen **{g09fft:.1f} ms, {100*g09fft/g09wall:.2f}%** del tiempo de G09.
Sus kernels suman alrededor de 96,6% del tiempo pared. Los huecos de host/API son
**{idle:.2f} ms, {100*idle/g09wall:.2f}%**. Quitarlos todos solo daría **{g09wall/(g09wall-idle):.3f}×**.
El posproceso es aproximadamente 1,27%; eliminarlo tiene poco impacto global.

En G08 y N64 el profiler aumenta el tiempo cerca de 12–16%. Por eso sus porcentajes de huecos describen
la ejecución instrumentada; no se presentan como porcentajes exactos del baseline sin instrumentar.
En G09 y N256 esa perturbación es inferior a aproximadamente 1%.

Costes de una llamada desde un proceso nuevo:

{setup_table}

**Memoria:** muestrear únicamente al entrar/salir de CG subestima el máximo activo. La campaña de espectro
muestrea cada actualización: G08 alcanza 51,7 MiB activos y G09 aproximadamente 3,20 GiB,
con 9,69 GiB libres como mínimo en el controlador para G09. El pool reservado, planes, contexto y escritorio
no equivalen a campos activos. No hay evidencia de que la capacidad de 16 GB provoque las iteraciones de estos casos.
La capacidad disponible y el ancho de banda son restricciones diferentes.

## 5. Roofline, calibración y límites que se pueden afirmar

Calibración independiente, nueve repeticiones y conjuntos de datos mayores que L2 cuando corresponde:

{table(['Medición','Resultado'],[
    ['Triad, tráfico lógico',f"{cal['bandwidth_triad']['GB_s']:.2f} GB/s"],
    ['Copy, tráfico lógico',f"{cal['bandwidth_copy']['GB_s']:.2f} GB/s"],
    ['Triad residente en caché',f"{cal['bandwidth_cache_resident']['GB_s']:.2f} GB/s"],
    ['FMA FP32 independientes',f"{cal['arithmetic']['float']['TFLOP_s']:.3f} TFLOP/s"],
    ['FMA FP64 independientes',f"{cal['arithmetic']['double']['TFLOP_s']:.3f} TFLOP/s"],
    ['Launch vacío ordinario',f"{cal['empty_launch']['median_s']*1e6:.3f} µs"],
    ['Nodo de graph vacío',f"{cal['empty_graph']['per_kernel_s']*1e6:.3f} µs"]])}

Los productos FMA no calibran divisiones, raíces, shuffles ni reducciones. Los GB/s son bytes lógicos del patrón,
no contadores DRAM. L2 tiene 48 MiB. La propiedad del dispositivo da aproximadamente 896 GB/s teóricos
de bus; no se usa esa cifra como ancho de banda sostenible medido.

El CSV estima bytes y FLOPs de los mapas, Cε, updates, reducciones y kernels FFT que cubren aproximadamente
91% de CUDA en G08, 90% en N64, 98% en G09 y 96% en N256. En FFT se usa un modelo de referencia de operaciones
y E/S por transformación 1D; cuFFT puede mover más o menos datos y ejecuta instrucciones diferentes.
{counter_classification}

Se calcula `max(bytes/BW, FLOPs32/F32, FLOPs64/F64, latencia)` y una sensibilidad optimista a caché
`max(max(bytes-2L2,0)/BW, bytes/BW_cache, …)`. La supuesta residencia de 2L2 no se ha medido.
Si el modelo predice un tiempo mayor al observado, esa fila se marca **calibration roof exceeded: invalid bound**.
No se interpreta una eficiencia >1 como eficiencia física; invalida esa calibración para ese kernel/patrón.

Los principales mapas/updates usan aproximadamente 32–40 registros y memoria local estática reportada 0 en CUPTI.
{counter_registers}
La metodología y las diferencias entre métricas se documentan en
[Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) y
[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

Para G09:

- Suma extrapolada de los P10 por kernel/variante, eliminando todo coste host:
  **{g09['fixed_kernel_p10_floor_ms']:.1f} ms**, razón tiempo trazado/escenario **{g09['fixed_kernel_ratio']:.3f}×**.
- Modelo optimista de caché/kernels, sin host: **{g09['optimistic_cache_kernel_model_ms']:.1f} ms**,
  razón **{g09['cache_model_ratio']:.3f}×**. No es un límite hardware certificado.
- Escenarios a 70/80/90% del BW sostenible, sin transferencias, sin allocations, sin launch API expuesto
  y coste cero del kernel más caro: [ideal_solver_scenarios.csv](ideal_solver_scenarios.csv).

**No es válido escribir una desigualdad certificada con esos escenarios como extremo inferior.**
El P10 no asegura que todos los kernels alcancen simultáneamente ese tiempo, y el modelo de caché no mide
tráfico real. El intervalo aproximado de escenarios para G09 es 1,59–2,44 s frente a 2,94 s actuales,
con dependencias y costes excluidos. Se conserva como análisis de sensibilidad.
Para 192³ sí hay una cota útil observada: el mínimo práctico con el mismo sistema matemático es
**≤1,190 s**, porque esa ejecución se realizó, frente a 1,561 s del default. Eso demuestra al menos 1,31×
de margen de implementación en ese punto, sin atribuirlo a un roofline hipotético.

## 6. Amdahl

Todas las fracciones de esta tabla usan la misma traza exclusiva de G09. La mejora es del solver completo:

{amdahl_table}

La fila de coste cero es un techo hipotético para actuar únicamente sobre ese componente.
No se suman aceleraciones de componentes que se solapan. [Tabla completa](amdahl.csv).

## 7. Costes ocultos y reutilización de cargas

- Las seis cargas son **secuenciales**, cada una con CG propio. Se construyen una vez por material
  el operador, tabla de elasticidad indexada, referencia, proyector y geometría compilada.
  Los planes se conservan; el pool reutiliza allocations.
- La traza warm no contiene cudaMalloc/cudaFree. Sí hay solicitudes al pool de CuPy y temporales;
  el ensayo de buffers persistentes no demuestra ahorro en el solve.
- Hay aproximadamente dos decisiones escalares CPU por iteración. En G09 aparecen 622 cudaMemcpyAsync
  y 646 cudaStreamSynchronize, con 564 llamadas de cada una dentro del CG instrumentado.
  El tiempo API inclusivo de memcpy es aproximadamente 2,75 s porque espera a la GPU;
  **el tiempo de transferencia GPU real es solo unos 2,6 ms**. Sumarlos contaría trabajo dos veces.
- Las copias D2H son principalmente escalares, unos pocos KiB. También se observan copias H2H ligadas
  a staging/preparación. Se conservan direcciones, bytes y llamadas en el CSV de memoria.
- El hueco expuesto en launch API de G09 es cerca de 7,15 ms. La cifra inclusiva de todas las llamadas
  de lanzamiento es mayor porque se solapa con los kernels.
- Cε y los campos usan SoA; no hay un layout AoS que deba cambiarse por mera intuición.
  No se midieron caché misses, accesos no coalescentes o divergencia; se registran como cuestiones pendientes.
- Campos/FFT FP32 y escalares FP64 son deliberados para conservar precisión. Los kernels de reducción
  fusionados se contabilizan dentro del operador/update, además de las reducciones explícitas.
  Reducir solo las explícitas no elimina el coste total de normas.

## 8. Microbenchmarks aislados de implementación

**Política existente de cuFFT**, siete repeticiones por variante, mismas iteraciones y criterio físico:

{table(['N','Default s','Componentes s','Ahorro tiempo','Speedup','CG'],fft_rows)}

En 128³ y 192³ la comparación de los seis campos completos dio **error 0** y Ceff idéntico.
En 256³ el error relativo máximo de campo es aproximadamente 1,52e-9 y de Ceff 2,30e-13.
El orden de ejecución alternado y la fila `full` casi igual a `auto` confirman el cambio de plan/batching.
No se recomienda activar componentes globalmente: en 96³ cuesta alrededor de 12% más tiempo.

Nsight confirma la procedencia del ahorro de 192³: el tiempo exclusivo en kernels FFT baja
**795,2 → 473,7 ms**. Ese descenso explica **321,5 ms de los 373,4 ms** recuperados en la traza;
el resto incluye cambios de coste/localidad de los demás kernels y de espera. El tiempo expuesto de launch API
sube 4,79 → 6,63 ms, aun así el solve acelera. Contar únicamente launches no determina el mejor plan.
[Comparación de trazas](fft_policy_trace.csv).

![FFT antes y después](fft_policy_breakdown.png)

Otros ensayos, medidos como suboperadores y no como una aceleración automática de seis cargas:

- Graph de H con input congelado: N64, **94,0 → 61,8 µs**; N128, **1,018 → 1,022 ms**;
  G09, **7,496 → 7,497 ms**. Beneficio claro en pequeño, ninguno en grande.
  Este ensayo evita la copia de entrada de un prototipo histórico, pero no captura las decisiones α/β de CG.
- FFT con buffers persistentes: N64 **24,9 → 28,6 µs**, N128 **373 → 381 µs**.
  La prueba G09 usa full batch frente a auto component, por lo que no permite atribuir la diferencia únicamente a allocation.
- Seis RHS, Cε aislado: N64 **99,2 → 92,1 µs**, N128 **902 → 902 µs**.
  Al incluir Cε+FFT, N64 **221 → 307 µs**, N128 **3,095 → 3,501 ms**: el batching empeora el suboperador completo.
- Streams independientes: aproximadamente **271 µs** en N64 y **3,524 ms** en N128 para ese suboperador;
  se midió pared sincronizando los streams, no events del stream nulo.
- G09 con dos RHS: secuencial **7,767 ms**, batched **7,797 ms**, streams **7,796 ms**: sin beneficio.
- El fixture de seis RHS de G09 agotó memoria mientras coexistían buffers seriales, batched y graph.
  No demuestra que una implementación compacta de seis RHS sea imposible; ese resultado se marca como fallo del fixture.

[Microbenchmarks](operator_microbenchmarks.csv). La salida congelada se verificó antes de medir.
La eliminación de sincronizaciones mediante single-reduction CG se evaluó separadamente abajo;
no se omiten controles de convergencia para ganar tiempo.

## 9. Diagnóstico matemático de iteraciones y precisión

El residual de coordenadas está ponderado de forma distinta al residual físico. Para el cambio R usado aquí,
la métrica física en frecuencia contiene

$$M(k)=R(k)^{{-2}}=2G_0 I+(K_0+4G_0/3-2G_0)\,n\otimes n.$$

Se verificó la identidad de norma con las ponderaciones RFFT y modos excluidos, en mallas pares/impares.
La **doble norma** mantiene CG y el suavizado actuales: al aceptar exige residual verdadero en coordenadas
y métrica física; si falla esta última, continúa con un reinicio fiable en coordenadas. La auditoría original
en deformaciones permanece activa como condición final.

{table(['Caso','Variante','Referencia → candidato s','CG','Ahorro','Residual físico máximo'],dual_rows)}

{strict_text}

La referencia puede terminar en ~5e-6 por la corrección a media tolerancia; el candidato de tolerancia pública
puede terminar más cerca de 1e-5. Por eso **no se atribuye todo el ahorro a mejor convergencia**.
En los tres casos con corrección, el error relativo de Ceff queda entre 1,38e-8 y 9,29e-8,
y el máximo L2 relativo de campo entre 1,26e-5 y 1,55e-5. Las dos soluciones tienen errores de truncamiento
iterativo distintos pero cumplen el criterio acordado. El control más estricto y sus campos se guardan aparte.
La doble norma no mejora G09 sin corrección y añade un pequeño coste. **En el control de precisión comparable,
4,412 → 4,771 s: empeora aproximadamente 8%. Se descarta como mejora de rendimiento demostrada.**

Otros planteamientos probados, fuera de producción:

| Piloto | Evidencia | Decisión |
| --- | --- | --- |
| CG con una reducción, recurrence de Chronopoulos/Gear | G08: 59,38 → 56,23 ms; G09: 2,944 → 3,196 s, mismas 276 iteraciones | No usar como default general; el vector extra perjudica mallas grandes |
| Precondicionador acústico local SPD en tres componentes | G08: 60,95 → 342,46 ms; 256 → 1027 iteraciones | Descartado para este diseño; SPD por sí solo no garantiza buen precondicionamiento |
| Mezcla 75% I + 25% acústico | G08: 61,92 → 106,16 ms; 256 → 263 | Descartada |
| Suavizado con métrica física y productos cruzados | G08: 60,05 → 67,80 ms; 256 → 249 | Descartado; reducir iteraciones no compensó el coste por paso |

La variante de una reducción se contrasta con la implementación/documentación de
[PETSc CG](https://petsc.org/release/src/ksp/ksp/impls/cg/cg.c.html).
El suavizado ponderado se relaciona con
[Zhou y Walker](https://users.wpi.edu/~walker/Papers/residual_smoothing,SISC_15,1994,297-312.pdf).
No se repiten como novedades el ajuste isotrópico, reciclaje Ritz, 18 modos Fourier,
CG por bloques o suavizado continuo ya descartados en campañas anteriores.

Todos los controles de campos incluyen seis cargas, errores absoluto/relativo L2/Linf, Ceff,
residual, iteraciones, simetría y autovalor mínimo. [field_validation.csv](field_validation.csv).
La validación de campos usa maxiter=1500 únicamente para cubrir el piloto acústico rechazado;
las mediciones de rendimiento conservan 1000 y la tolerancia física permanece idéntica.

## 10. Qué formulación nueva merece el siguiente ensayo

La prioridad demostrada es **calibrar los planes de cuFFT por dimensiones/GPU**. La corrección de residuales es
una oportunidad cuantificada, pero el piloto de doble norma no ahorra tiempo al mantener precisión comparable.
Hace falta una estrategia distinta para ese coste. Los cambios de default se deciden después de este diagnóstico.

Para una reducción sustancial adicional de iteraciones, propongo un **espacio grueso adaptativo de energía**,
con modos de problemas espectrales locales asociados a heterogeneidad y contraste. No es una base de 18 modos
Fourier ni reciclar simplemente los últimos residuos. Se escogerían modos de energía lenta y se comprobaría
el espectro resultante. La inspiración metodológica es
[Chung, Efendiev y Leung, precondicionamiento adaptativo de dos niveles](https://research.cuhk.edu.hk/en/publications/an-adaptive-generalized-multiscale-finite-element-method-based-tw-2/).
La formulación del paper es diferente; esta adaptación a H es una propuesta, sin aceleración medida.

Con columnas admisibles Z, E=Z*HZ y un inverso base B SPD, una corrección balanceada sería

$$P^{{-1}}=ZE^{{-1}}Z^*+(I-ZE^{{-1}}Z^*H)\,B\,(I-HZE^{{-1}}Z^*).$$

Se mantendrían H, campos FP32, productos FP64 y el residual físico final. Hay que medir coste de construcción,
aplicación, transferencias y memoria; una base global densa puede ser demasiado cara. A 256³, 16 vectores de
tres componentes complex64 ya ocupan aproximadamente 3,25 GB decimales. Almacenar además HZ duplica esa cifra.
Se necesita probar una base pequeña o soporte local y construcción por bloques antes de ampliar.

El criterio de éxito tiene que ser temporal:

$$N_{{nuevo}} < \frac{{N_{{actual}}t_{{iter}}-T_{{setup}}}}{{t_{{iter}}+t_{{P}}}}.$$

En G09, 276 pasos a unos 9,86 ms: con setup de 100 ms y aplicación de 2 ms se necesitarían menos de
aproximadamente **221 pasos**, una reducción superior al 20%, solo para empatar. Son costes hipotéticos usados
para establecer un umbral; no se presentan como mediciones de ese precondicionador.
Si se redujera 50% de CG sin ningún coste nuevo, Amdahl daría aproximadamente
**{float(b['G09']['median_ms'])/float(half['total_ms']):.2f}×** en G09. El setup/aplicación disminuye esa ganancia.

## 11. Respuestas A–G

**A. ¿Está cerca del límite práctico?** No de manera uniforme. En 192³ el default deja 24% recuperable
medido, o al menos 1,31×. Para G09 se calcula 1,208× frente a una extrapolación P10 sin host y 1,85× frente
al modelo optimista de caché. Ninguna razón usa un mínimo hardware certificado. Los contadores muestreados
no bastan para afirmar un `T_actual/T_practical_lower_bound` definitivo para G09.

**B. ¿Algoritmo o implementación?** Ambos. El exceso demostrado en 128³/192³ es de implementación cuFFT.
En algunas geometrías el cambio de norma dispara corrección, un coste algorítmico medido cuyo reemplazo probado
no mejora con precisión comparable. En G09 predominan
las aplicaciones GPU de H y FFT; quitar CPU/transferencias por sí solo no cambia ese diagnóstico.

**C. ¿T_iter o N_iter?** Primero T_iter donde el cambio de plan ya demuestra 22–24%. Después las correcciones
cuando existen. Para G09 sin corrección, la doble norma y los precondicionadores probados no prueban una mejora;
una gran reducción adicional requiere un precondicionador que supere su coste por aplicación.

**D. ¿Porcentaje recuperable?** Alrededor de 24% demostrado en el caso central 192³, 22% en 128³ y 4% en 256³
con el cambio de política FFT. En casos con correcciones, el candidato de doble norma ronda 10% bajo el criterio
público; **ese ahorro no se acepta como mejora porque desaparece con precisión comparable**. En G09,
eliminar todo hueco host/API solo recuperaría 3,2% en la traza.
No se suman estos porcentajes entre casos distintos ni se convierte el escenario de caché en porcentaje garantizado.

**E. ¿Máximo razonable sin cambiar el algoritmo?** Hay un 1,31× realizado en 192³ y 1,28× en 128³.
Son mejoras alcanzadas, no máximos absolutos. En G09, la extrapolación con kernels actuales da aproximadamente
1,21× en el escenario optimista sin host. El techo absoluto de implementación queda sin certificar.

**F. ¿Con cambios algorítmicos?** Ningún piloto nuevo demuestra mejora general a precisión comparable;
la doble norma se rechaza por el control estricto.
Un hipotético 50% menos iteraciones produciría ~1,86× en G09 antes de pagar el nuevo precondicionador.
Ningún piloto actual demuestra esa reducción; no hay base para prometer 2× o 10× por una formulación nueva.

**G. ¿Qué impide seguir acelerando?** En grande, las FFT más el recorrido de campos/métricas en H y los updates
de CG; en pequeño, la combinación de launch/control escalar domina parte importante de la ejecución.
Las correcciones finales añaden coste evitable en determinados espectros. La capacidad de VRAM no explica
por sí misma los saltos de iteraciones observados.

## 12. Artefactos, trazabilidad y reproducción

Entregables principales: [performance_baseline.csv](performance_baseline.csv),
[kernel_profile.csv](kernel_profile.csv), [scaling_results.csv](scaling_results.csv),
[optimization_opportunities.csv](optimization_opportunities.csv), gráficos y este informe.
El campo `max_global_speedup` distingue en `evidence` **techo Amdahl**, **extrapolación** y **ratio observado**;
un ratio observado no es un máximo certificado. Se ordenan por ahorro absoluto medido o condicionado, con esa salvedad.

Además: [experiment_statistics.csv](experiment_statistics.csv) incluye dispersión e intervalos bootstrap pareados,
[geometry_manifest.csv](geometry_manifest.csv) guarda aceptación/Vf/A2/hashes, `geometries/` conserva los 12 masters,
`material.json` fija la física y `evidence.json.gz` contiene los informes medidos y las trazas resumidas.
Los rasters pesados y .nsys-rep permanecen locales en `benchmarks/results/performance_audit_20261001/`.
Las versiones de los harness y hashes de producción/entradas se registran en `evidence_manifest.json`.

Para reproducir el barrido, desde la raíz del repositorio con el mismo entorno CUDA/CuPy:

```bash
source benchmarks/reference_solvers/env_gpu.sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmarks/reproduce_fiber_audit.py \
  --output benchmarks/results/performance_audit_repeat --run --repeats 9
```

Después de completar los diagnósticos/experimentos, regenerar las tablas:

```bash
.venv/bin/python benchmarks/build_performance_audit.py \
  --input benchmarks/results/performance_audit_20261001 --output benchmarks/performance_audit_20261001
.venv/bin/python benchmarks/build_audit_experiments.py \
  --input benchmarks/results/performance_audit_20261001 --output benchmarks/performance_audit_20261001
.venv/bin/python benchmarks/write_performance_report.py \
  --input benchmarks/results/performance_audit_20261001 --output benchmarks/performance_audit_20261001
```

Los pilotos se ejecutan exclusivamente desde `descartados/performance_audit_20261001` mediante PYTHONPATH explícito.
No se importan desde producción. [Estado y criterios de descarte](../../descartados/performance_audit_20261001/README.md).
{counter_conclusion}
'''
    (a.output/'PERFORMANCE_AUDIT.md').write_text(report)
    print(a.output/'PERFORMANCE_AUDIT.md')


if __name__=='__main__':main()
