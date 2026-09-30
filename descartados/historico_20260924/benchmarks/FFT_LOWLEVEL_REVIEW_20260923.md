# Opciones HPC y bajo nivel: revisión y experimentos aislados

Actualización 24/09: [pruebas completas de las tres ideas](FFT_LOWLEVEL_TRIALS_20260924.md).
Fusión confirmada en curva: 57,801→56,362 s (2,49 %); las referencias
isótropas ensayadas y el prototipo híbrido cuFFTDx no mejoraron a esa fusión.
No se cambiaron los defaults de producción. El resto de este informe
conserva los resultados del cribado inicial.

## Conclusión

La opción inmediata con evidencia es **fusionar el mapa adjunto de coordenadas
con el producto `p.Ap`**: 1,121 → 0,912 ms, incluyendo lectura del escalar por
el host, en un microbenchmark 240³. Es un **18,7 % menos en esa etapa**, no en
la curva completa. Todavía no está integrado ni se ha medido dentro de CG.

CUDA Graphs sobre el operador congelado no produjo una mejora clara. Añadir
una copia de entrada al grafo lo hizo más lento. No se cambia ningún default.
Siguen vigentes los tiempos de producción previamente confirmados:
**57,769 s no lineal / 3,164 s seis cargas lineales**, con los alcances de
[la campaña anterior](FFT_HOTSPOTS_20260923.md).

Sólo se añaden experimentos, pruebas e informes. No se cambia malla, física,
trayectoria, tolerancias, Superbody ni el pipeline de la tesis.

## Dónde atacar

El perfil previo de la curva completa, no un nuevo perfil integral, separaba:

| Trabajo | Intervalo exclusivo, s | Prioridad |
| --- | ---: | --- |
| FFT directa + inversa | 26,760 | Reducir CG; estudiar fusión con pasadas FFT |
| Tangente aplicada a vectores | 8,004 | Evitar lecturas/intermedios; no confundir con integración J2 |
| Coordenadas | 6,615 | Fusionar su salida con productos internos |
| Actualización y dirección PCG | 9,767 | Menos pasadas y menos iteraciones |
| Productos internos/sincronización | 3,109 | Fusionar; separar espera GPU de transferencia real |
| Integración constitutiva J2 | 0,324 | Baja prioridad para acelerar esta curva |

Estos son intervalos CUDA instrumentados, incluidos huecos de lanzamiento,
no contadores de tiempo activo de kernels. El núcleo ya utiliza código CUDA
compilado y cuFFT. Reescribir el coordinador Python en C++ no elimina por sí
solo el tráfico de los campos ni el trabajo de las FFT.

## Experimento 1: buffers persistentes y CUDA Graphs

Operador `A p = Q* C_alg Q p` con fase/orientaciones reales G09 240³, tangente
congelada de un campo de deformación **sintético**, float32, RTX 5070 Ti.
Nueve grupos rotados de diez aplicaciones; planes/kernels calientes.

| Variante | Evento CUDA, ms | Pared sincronizada, ms |
| --- | ---: | ---: |
| Operador actual | 8,061 | 8,068 |
| Buffers persistentes, sin grafo | 8,021 | 8,023 |
| Grafo, entrada en dirección de memoria fija | 8,036 | 8,043 |
| Grafo con copia de entrada incluida | 8,520 | 8,528 |

La dispersión se solapa entre las primeras tres variantes. No hay evidencia
para prometer una reducción relevante de curva. Captura y primera ejecución:
15,5 ms; buffers propios del prototipo: 1,667 GB decimales, **sin** contar
estado, respuesta constitutiva, planes y reserva del pool. Esta cifra no es
la memoria máxima total ni un incremento neto respecto al solver actual.

Se verificó igualdad bit a bit del operador para el campo ensayado y uso de
una entrada distinta tras replay. El grafo conserva vivos buffers, vistas,
planes y respuesta; la inversa C2R consume su entrada y ésta se reconstruye
siempre. El resultado es prestado, no una salida con propiedad independiente.
No se permite activar esto como sustitución transparente del operador.

La documentación distingue capturar lanzamientos de fusionar el trabajo de
los kernels: el grafo no elimina sus intermedios globales.
[NVIDIA, Kernel Fusion (2026)](https://developer.nvidia.com/blog/?p=119743).
cuFFT admite grafos, pero exige respetar dependencias, streams y buffers.
[Documentación cuFFT](https://docs.nvidia.com/cuda/cufft/index.html#cuda-graphs-support).

[Resultados y hashes](results/g09_cuda_graph_frozen_tangent_20260923.json).

## Experimento 2: fusionar mapa adjunto + p.Ap

Prototipo `benchmark_fused_adjoint_dot.py`: mientras cada hilo escribe las
tres componentes complejas de `Ap`, acumula su contribución a
`Re(p* Ap)` en FP64. Incluye modos medios libres, planos de Nyquist y bloques
parciales. No usa atomics globales, `fast_math` ni reducción de precisión.

Evita volver a leer **167.270.440 bytes por aplicación** del vector de salida
en este tamaño. Es una cuenta de accesos algorítmicos evitados; no una medición
de bytes DRAM ni una garantía de que toda lectura anterior fallara en caché.
`Ap` todavía se escribe porque CG lo necesita para actualizar el residuo.

| Mapa adjunto + producto | Original, ms | Fusionado, ms |
| --- | ---: | ---: |
| Eventos CUDA, escalar residente | 1,102 | 0,894 |
| Pared sincronizada, escalar residente | 1,102 | 0,897 |
| Pared incluyendo lectura del escalar en cada llamada | 1,121 | 0,912 |

Nueve grupos rotados de veinte aplicaciones, campo sintético 240³. Salida
bit a bit igual y diferencia del producto cero en esta muestra. El orden de
reducción cambia; **no se promete igualdad bit a bit de todos los escalares,
iteraciones o curvas**. Las pruebas incluyen productos de signo arbitrario
y cero; no sólo el producto positivo usado para medir.

Antes de integrar:

1. Conectar explícitamente operador/producto con PCG; gestionar la propiedad
   de la salida sin sobrescribir datos todavía en uso.
2. Tratar PTC: el producto debe corresponder exactamente a `A + shift I`, no
   omitir `shift * p.p`. Conservar chequeos de positividad y residuo verdadero.
3. Adaptar por separado la raíz del precondicionador en coordenadas lineales;
   el prototipo actual sólo acepta `DisplacementCoordinates` no lineales.
4. Comparar curvas completas G09 en corridas alternadas y contar CG/Newton,
   residuo, Hill–Mandel, fluencia y disipación. Conservar las tolerancias.

[Resultados y hashes](results/g09_fused_adjoint_dot_20260923.json).

## Otras estrategias viables, todavía sin ahorro demostrado

### A. Referencia SPD que siga la tangente: prioridad matemática

Construir una referencia homogénea del estado tangente actual y congelarla
dentro de cada CG. La reducción de iteraciones amortiza FFT, tangente,
coordenadas y vectores simultáneamente; una fusión local no reduce CG.
[DBFFT, Lucarini–Segurado, §2.5](https://arxiv.org/html/1905.12725v2)
fundamenta una referencia basada en rigidez tangente media. Su aplicación a
nuestro espacio isométrico y cinco modos medios requiere una derivación y
validación propias; no basta sustituir una constante sin auditar SPD.

Evaluar primero tangentes congeladas elástica, de transición y plastificada
de soluciones reales. Incluir coste de construcción y aplicación. Una
referencia anisótropa puede reducir CG y aun perder tiempo por sus accesos
a memoria: no repetir sin más los candidatos lineales descartados.

Green–Jacobi queda como alternativa, no como ganador garantizado. El artículo
destaca datos suaves de alto contraste y reconoce crecimiento de iteraciones
al refinar. No transportar su diagonal FE de soporte local directamente al
operador Fourier denso. [Paper J-FFT](https://doi.org/10.1016/j.cma.2026.119269),
[revisión específica del repositorio](JACOBI_GREEN_REVIEW_20260923.md).

El criterio de decisión es coste total:

`T ≈ T_preparación + Σ_CG (T_FFT + T_tangente + T_vectores + T_precondicionador)`.

### B. FFT con procesamiento fusionado mediante cuFFTDx: mayor desarrollo

Estudiar pasadas 1D por bloques que fusionen los extremos de la FFT con
operaciones del solver, reduciendo escritura/lectura de campos intermedios.
NVIDIA ofrece ejemplos de convoluciones 3D R2C/C2R y múltiples pasadas cuando
el volumen no cabe en memoria compartida.
[Ejemplos oficiales cuFFTDx](https://docs.nvidia.com/cuda/cufftdx/examples.html).

No implica meter 240³ en un bloque, cambiarlo a 256³ ni trasladar al solver
las aceleraciones medidas en una H100 para otra convolución. Las seis
componentes Mandel se acoplan por la ley material: un callback que lee otras
salidas cuFFT antes de su finalización introduce una carrera. Hay que diseñar
explícitamente los puntos de sincronización y mantener el orden del operador.
El SDK cuFFTDx no se encontró en los includes CUDA inspeccionados; no se ha
instalado ni se ha probado este backend en la presente revisión.

### C. Escalares y control de PCG residentes en GPU

Calcular `alpha` y `beta` sin devolver cada escalar al host puede reducir
sincronizaciones. Es un candidato distinto de capturar sólo el operador,
que fue lo probado arriba. Debe mantener detección inmediata de NaN/pérdida
de positividad, parada, límites y auditoría del residuo real; no ejecutar
bloques ciegos de CG cuando ya convergió. Medir el balance contra el trabajo
extra y la ocupación. No se ha implementado ni medido este control residente.

### D. Paralelismo: escoger trabajo realmente independiente

Una GPU ya paraleliza por vóxel/frecuencia. Los incrementos plásticos tienen
dependencia de estado; no pueden ejecutarse simultáneamente sin otro algoritmo
y una validación distinta. Materiales independientes o cargas elásticas sí
admiten concurrencia si memoria y ancho de banda lo permiten. En una GPU,
varios streams pueden competir por el mismo ancho de banda; necesitarían
planes/workspaces separados y un benchmark que incluya toda sincronización.
cuFFT no garantiza que repartir un problema entre varias GPU sea más rápido.
[cuFFT: streams y varias GPU](https://docs.nvidia.com/cuda/cufft/index.html#streamed-cufft-transforms).

## Verificación y límites de esta revisión

**287 pruebas aprobadas**, 6,02 s: regresiones lineales/no lineales existentes
más 26 pruebas nuevas. Incluyen FP32/FP64, todos/ninguno/algunos modos medios
libres, dimensiones pares/impares, bloques incompletos, replay con nueva
entrada y ausencia de mutación del estado confirmado y la respuesta.
Sólo aviso preexistente de `numpy.matlib`.

Se intentó Nsight Compute `SpeedOfLight`, pero devolvió
`ERR_NVGPUCTRPERM`: los contadores de hardware no están habilitados para este
usuario. No se modificaron permisos ni el driver. Por tanto, no se informa
ocupación, ancho de banda DRAM efectivo o posición Roofline medidos. Los
tiempos válidos son los benchmarks CUDA/pared sin profiler; el intento de
Nsight no forma parte de las comparaciones. La guía explica qué métricas
faltan para distinguir límites de memoria, instrucciones y cómputo.
[Nsight Compute, Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

Reproducir usando rutas de salida nuevas:

```bash
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_cuda_graph_operator.py \
  --output /tmp/fft-graph-review-new.json
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_fused_adjoint_dot.py \
  --output /tmp/fft-fused-dot-review-new.json
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_fft_lowlevel_candidates.py
```

Orden propuesto: **validar fusión `p.Ap` en CG completo → evaluar referencia
tangente SPD → prototipo de FFT fusionada**. CUDA Graphs aislados no justifican
cambiar el camino rápido actual; tampoco se adopta FP16/TF32 ni una tolerancia
más laxa para fabricar una aceleración.
