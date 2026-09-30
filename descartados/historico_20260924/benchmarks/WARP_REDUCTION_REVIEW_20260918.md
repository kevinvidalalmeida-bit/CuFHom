# Ensayo sin cambiar malla: reducciones CG por warp

## Alcance y reversión

A petición del usuario se detuvo el experimento de malla. Se restauró
`applications.py` a su estado posterior a la mejora validada del CG. El hash
coincide con el registrado en `g09_cg_cleanup.json`. Se conservaron los cambios
de eliminación de A(0) y de la última dirección innecesaria.

Los archivos nuevos de aquel experimento se movieron, sin borrarlos, a
`/tmp/fft-nested-grid-paused.c283Xa`: `nested_grid.py` (antes en `FFT/pipeline`),
`test_nested_grid.py` (antes en `tests`), `profile_g09_nested_grid.py` (antes en
`FFT/benchmarks`) y sus tres JSON de resultados (antes en `benchmarks/results`).
Esta ubicación temporal permite recuperación, pero no es un archivo permanente.
No se alteraron otros cambios previos de usuario ni la geometría G09.

## Estrategia alternativa ensayada

`general/cupy_warp_reduction.py` implementa una alternativa experimental a las
reducciones de CG. Usa sumas por warp con `__shfl_down_sync`, una barrera por
bloque, recorrido grid-stride y reducción final de los parciales. Los productos
y sumas se acumulan en float64; los campos mantienen float32. Los escalares
siguen en GPU. No se utilizan sumas atómicas globales, ni se cambian ecuaciones,
materiales, malla, tolerancia o frecuencia de comprobación de convergencia.

Se midieron 1024, 4096, 16384 y 65536 bloques sobre campos (6,240,240,240).
El mínimo conjunto de dot + actualización x/r fue 16384 bloques. Es un ajuste
para esta GPU y este tamaño; no se ha declarado óptimo universal.

La clase no se conecta por defecto al solver. El benchmark sustituye
temporalmente la clase de reducción opcional dentro de su proceso y la restaura
al terminar, incluso si ocurre un error. La configuración productiva no cambia.

## G09 240³: resultados completos

RTX 5070 Ti, seis cargas, float32, rtol=1e-5, cuatro repeticiones por variante
después de calentamiento, orden alternado, geometría/material idénticos.
Lectura de geometría, JIT, microbenchmarks y auditorías fuera de los tiempos.

| Variante | Mediana del solve | Iteraciones |
|---|---:|---:|
| Atómicas actuales | 4.680 s | 360 |
| Árbol existente, float64 | 4.671 s | 360 |
| Árbol por warp, 16384 bloques | 4.663 s | 360 |

La diferencia es aproximadamente 16 ms, 0.35%. El rango de tiempos de la
referencia fue 4.671–4.688 s: el efecto es pequeño y no justifica un cambio
predeterminado. No comparar directamente con los 4.767 s de otra sesión: la
conclusión usa la referencia medida en esta misma ejecución.

Todas las auditorías pasan el residuo verdadero original (máximo aproximado
8.58e-6). Las diferencias L2 de los campos fluctuantes frente a una ejecución
de referencia son del mismo orden que repetir la referencia: 1.15e-5 frente
a 1.14e-5. El JSON incluye Ceff de cada repetición y diferencias relativas.
Se exige diferencia de Ceff <1e-6 y diferencia de campos no superior a dos
veces la variación medida de la referencia (con suelo 1e-6).

35 pruebas aprobadas: precisión float32/float64, tamaños no múltiplos de warp,
actualización fusionada, propiedad de los escalares (no sobrescribir rr previo),
fallback para arrays no contiguos, CG, RFFT y pruebas constitutivas existentes.

```bash
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/profile_g09_warp_reductions.py --repeats 4
OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_cg_warp_reduction.py tests/test_cg_redundant_work.py \
  tests/test_rfft_unscaled_cuda.py tests/test_composite_voxels.py
```

Resultados crudos: `results/g09_warp_reductions.json`.

## Decisión

Conservar como experimento reproducible, no activar automáticamente. La
variante conserva las ecuaciones y pasa esta prueba, pero el beneficio es
marginal. El siguiente candidato de mayor alcance, aún no implementado ni
medido, es fusionar `P = R + beta*P` con `C*P`, para evitar un recorrido de
lectura del campo P en cada iteración. Requiere respetar redondeo, propiedad de
buffers y callbacks, y repetir las auditorías antes de integrarlo.
