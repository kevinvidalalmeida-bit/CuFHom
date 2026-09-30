# Polarización GPU, integración rápida y papers 2026

2026-09-22. G09 original 240³, seis cargas, RTX 5070 Ti, float32,
residuo físico ≤ 1e-5. No cambia la discretización ni la ley material.

## Integración de la combinación validada

La combinación dejó de depender de parches experimentales. La ruta compatible
`linear_formulation='auto'` usa ahora `linear_preconditioning='symmetric_fused'`.
La referencia anterior se conserva con `linear_preconditioning='matrix'`.
Implementación en `pipeline/linear_krylov.py` y `pipeline/linear_coordinates.py`.
No se modificó ningún módulo del solver no lineal.

Comparación nueva: tres repeticiones por variante, orden rotativo, GPU caliente,
geometría compilada precargada; preparación del solve, auditorías y escritura
de metadatos incluidas, sin exportación de campos:

| Variante | Mediana | CG totales |
| --- | ---: | ---: |
| PCG anterior | 3,7984 s | 288 |
| Combinación experimental | 3,4184 s | 288 |
| **Combinación integrada** | **3,4177 s** | **288** |

La integración reduce aproximadamente **10,02 %** el tiempo frente al PCG
anterior de esta campaña. La diferencia relativa de C_eff es 6,2931e-8 y el
residuo máximo 9,2716e-6. No hubo iteraciones correctoras. Es ahorro por
iteración, **no reducción del número de CG**. El objetivo pendiente señalado
por el usuario es reducir también esas 288 iteraciones y, con ello, las FFT.

[Reporte de integración](results/g09_combined_production_20260922/report.json).
Los benchmarks conservan `baseline` como PCG anterior explícito;
`production` usa el nuevo default. No comparar por nombre sin revisar la ruta.

El [perfil actualizado de producción](results/g09_integrated_stage_profile_20260922/report.json)
dio 3,4192/3,4223 s sin instrumentar alrededor de 3,4236 s instrumentados,
con C_eff idéntico y 288 CG. Los intervalos CUDA internos fueron 1,7088 s en
FFT de coordenadas, 0,3924 s en conversión, 0,3828 s en actualización/norma
fusionadas, 0,1855 s en productos internos/sincronización y 0,3202 s en
constitutiva. Son intervalos por eventos, incluyen huecos y no incluyen todas
las FFT de auditoría: no confundirlos con los 1,7741 s de kernels FFT del perfil
Nsight del combinado ni sumar padres e hijos inclusivos.

## Qué pasó con la polarización

Se implementó `linear_polarization_candidate.py`, con kernels CUDA de
inserción/restricción, tablas constitutivas por orientación y raíces calculadas
en float64. No almacena una rigidez global ni una raíz 6×6 por cada vóxel.
Reutiliza RFFT/Green existentes y acepta sólo después de auditar el problema
físico original. Las tablas provienen del mismo material indexado del solver.

| Variante | Mediana | CG reducidos + correctores | Diferencia C_eff vs PCG |
| --- | ---: | ---: | ---: |
| Polarización de fluctuación / Woodbury | 3,8211 s | 252 + 72 | 1,81e-7 |
| Polarización macroscópica directa | 3,6528 s | 312 + 18 | 3,29e-8 |
| Directa con escalado local SPD | 3,5943 s | 303 + 18 | 3,47e-8 |

Tres repeticiones por variante, cada campaña intercalada con PCG y combinado.
Todas cumplen las auditorías; ninguna supera al combinado (~3,42 s).
No se suman mejoras de campañas diferentes.

Evidencia: [Woodbury](results/g09_polarization_gpu_20260922/report.json),
[directa](results/g09_polarization_macro_20260922/report.json),
[local SPD](results/g09_polarization_local_20260922/report.json).
Los informes guardan hashes de los archivos usados; son campañas sucesivas.

### Diferencia entre las formulaciones

Para un RHS compatible arbitrario b y D = (Cf-Cm)^(1/2), la primera usa

    u0 = Γm b
    (I + D S* Γm S D) p = D S* u0
    u = u0 - Γm S D p

La directa usa la deformación macroscópica conocida E:

    (I + D S* Γm S D) p = D S* E
    u = -Γm S D p

S inserta valores de fibra; S* restringe. La directa evita una resta de campos
en la reconstrucción, pero esto **no demostró ser una aceleración suficiente**.
El residuo reducido no es el residuo físico: no se acepta el primero como
sustituto del segundo. Por eso se reportan también los CG correctores.

En la variante local, L = (I + D Gbar D)^(-1/2), donde Gbar es una aproximación
angular positiva del Green. Se aplica la congruencia SPD con K = D L:

    [L² + K* S* Γm S K] y = K* S* E

Se mantiene K y su **transpuesta**, sin asumir que D y L conmutan. Gbar no es
la diagonal exacta del Green discreto. Esta es una propuesta propia para el
operador reducido, no una reproducción del J-FFT ni un resultado de su paper.

La matriz/resina no se borra físicamente. La reducción por vector Krylov es
167,27 → 82,88 MB, pero siguen los buffers FFT completos. El
[perfil separado de asignaciones](results/g09_polarization_live_memory_20260922.json)
obtuvo estos picos vivos en el pool CuPy: **4,43 GB** en producción/combinado,
**6,60 GB** en Woodbury y **6,27 GB** en directa/local. No incluye contexto CUDA,
asignaciones externas al pool ni bloques libres reservados. En estos prototipos
la coexistencia de buffers y correcciones pesa más que el ahorro por vector:
**no se ha demostrado una reducción de memoria total**.

### Por qué reducir incógnitas no basta para reducir CG

En aritmética exacta, con T = D S* Q R, el operador simétrico original tiene
forma I + T* T sobre coordenadas admisibles. El reducido es I + T T*.
Comparten los autovalores asociados a valores singulares no nulos de T;
pueden diferir en multiplicidades del autovalor 1 y en el RHS. Por tanto,
eliminar incógnitas de matriz **no elimina por sí solo el contraste espectral**.
Esto es un diagnóstico algebraico de nuestro caso, no una promesa de CG iguales
en precisión finita ni con criterios de parada distintos.

## Ajuste de referencia para atacar CG

También se ensayó el cociente longitudinal/transversal del inverso de
referencia, manteniendo SPD y la ecuación física. El valor de matriz es
aproximadamente 0,4513. La búsqueda preliminar no dio una mejora:

| Cociente | Mediana | CG totales |
| --- | ---: | ---: |
| 0,30 | 4,1576 s | 348 |
| 0,48 | 3,6743 s | 303 |
| 0,52 | 3,4850 s | 294 |
| 0,60 | 3,5489 s | 300 |
| 0,90 | 3,9977 s | 342 |
| 1,20 | 4,5133 s | 390 |

Dos repeticiones por punto: **screening**, no campaña equilibrada definitiva
ni prueba de óptimo global. Se conserva la referencia de matriz.
[Barrido](results/g09_reference_ratio_screen_20260922/report.json),
[refinamiento](results/g09_reference_ratio_refined_20260922/report.json).

## Papers revisados y decisión para nuestro solver

- **UNO-CG, Herb–Fritzen, 2026:** aprender precondicionadores SPD, conservando
  CG y verificación. La aplicación FFT sigue siendo O(N log N). La comparación
  con FANS no certifica superar nuestro solver. Próximo desarrollo: bloques
  espectrales aprendidos, evaluados fuera del conjunto de ajuste y contabilizando
  entrenamiento. [Artículo](https://onlinelibrary.wiley.com/doi/10.1002/nme.70277).
- **X-FFT, Gehrig–Schneider, 2026:** cotas espectrales sin ensamblaje para elegir
  pasos robustos. Su §5 indica que el cambio de paso no beneficia al CG lineal.
  No sustituir nuestro CG por un esquema básico sólo por esas cotas.
  [Artículo](https://onlinelibrary.wiley.com/doi/full/10.1002/pamm.70174).
- **FNO universales, Nguyen–Schneider, enero 2026:** conexión entre operadores
  neuronales y esquemas iterativos FFT. La universalidad no demuestra una
  aceleración de este G09 con error 1e-5. No reemplazar la solución auditada por
  una predicción. [Artículo](https://www.sciencedirect.com/science/article/pii/S0022509625003928).
- **J-FFT, Ladecký et al.:** Green–Jacobi se orienta a coeficientes suaves con
  alto contraste; nuestro material tiene interfaces abruptas. Consultado el
  [preprint](https://arxiv.org/html/2508.02613v2); el
  [registro editorial 2026](https://doi.org/10.1016/j.cma.2026.119269) aparece
  asignado a noviembre, posterior a la fecha de esta revisión. No se presenta
  como un benchmark nuevo ejecutado aquí ni se confunde con nuestro escalado local.
- **FANS:** el artículo original es de 2017/volumen 2018, no de 2026. Utiliza
  operadores nodales FE y precondicionamiento FFT; cambiar directamente a FANS
  cambiaría la discretización. Su implementación oficial es C++/MPI/FFTW.
  Reutilizamos conceptualmente referencias SPD y precondicionamiento espectral,
  no copiamos su discretización ni instalamos otro solver para atribuirle tiempos.
  [Artículo](https://link.springer.com/article/10.1007/s00466-017-1501-5),
  [código oficial](https://github.com/DataAnalyticsEngineering/FANS).

## Validación y reproducción

**152 pruebas pasaron** tras integrar: lineal, polarización, campo indexado,
composite voxels y regresión no lineal. Incluyen comparación CPU64 en mallas
pares/impares, materiales G09 con orientaciones aleatorias, simetría/positividad,
reutilización de buffers, rechazo de contraste negativo, RHS nulo, restauración
de parches y recuperación ante falsa convergencia interior. No equivalen a
validación universal para cualquier material ni a una curva no lineal G09 nueva.

```bash
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_linear_strategies.py \
  --modes baseline combined production --repeats 3 --output /tmp/fft-production-new

OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_linear_strategies.py \
  --modes baseline combined polarization polarization_macro polarization_local \
  --repeats 5 --output /tmp/fft-polarization-new
```

Los prototipos usan contextos de parcheo en un proceso secuencial dedicado;
no deben llamarse concurrentemente. Producción no importa estos prototipos.
El desarrollo de un precondicionador que reduzca los **288 CG del combinado**
permanece pendiente: las variantes anteriores no justifican activarse por defecto.
