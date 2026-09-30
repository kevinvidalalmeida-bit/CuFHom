# G09 240³: reducción de latencia sin modificar la trayectoria

## Condiciones de comparación

Geometría G09 original, 40 incrementos uniformes de 0 a 1 %, tracción axial
libre, matriz J2 con fluencia 15 MPa y endurecimiento 30 MPa demostrativos.
Mismas propiedades elásticas registradas en `profile_g09_fft.MATERIAL`.
GPU RTX 5070 Ti, CuPy float32, reducciones float64, residuo exterior ≤ 1e-5.
La CPU float64 conserva la referencia de verificación. No se alteran malla,
ley constitutiva, frecuencias, estado plástico confirmado ni carga.

Los tiempos son de curva completa, con planes/kernels calientes, preparación
de geometría/estado, solución, auditorías y confirmación de estados. Excluyen
exportación de campos/checkpoints, escritura de metadatos y gráfica. No son
tiempos de seis problemas elásticos ni predicciones para otra GPU.

## Perfil inicial

`results/g09_nonlinear_stage_profile_20260923/report.json`: curvas sin
instrumentación 77,987 y 78,049 s; curva instrumentada 78,274 s. Todas tienen
5.603 CG, 100 Newton, 40 incrementos, cero rechazos y curvas idénticas.
Intervalos CUDA exclusivos aproximados:

| Trabajo | s |
|---|---:|
| RFFT + IRFFT | 35,23 |
| Acción de tangente constitutiva | 9,29 |
| Conversiones de coordenadas | 8,46 |
| Productos internos y sincronización asociada | 7,65 |
| Actualización x/r | 7,29 |
| Actualización p | 3,60 |
| Precondicionamiento | 2,68 |

Son intervalos delimitados por eventos, que incluyen posibles huecos de
lanzamiento/espera, no exclusivamente tiempo de kernels activos. El tiempo
host de una reducción puede incluir trabajo GPU previo. No sumar etapas
inclusivas anidadas. El perfil cuenta 241 evaluaciones constitutivas: 100 son
repeticiones de la evaluación ya aceptada por la búsqueda de paso.

## Cambios matemáticamente equivalentes

- Reutilizar respuesta, residuo y norma de un candidato **aceptado** dentro
  del mismo intento de incremento. Se descarta al rechazar/confirmar un paso.
  El estado material no se modifica durante Newton ni line search.
- Fusionar actualización x/r y las reducciones `r·r`, `r·M⁻¹r`; aplicar
  `M⁻¹r` dentro de la actualización de p. No materializar un vector z global
  por iteración. Se conservan modos medios libres y desplazamiento PTC.
- Mantener la norma isométrica original de CG, positividad y comprobación
  explícita de `b-Ax` antes de declarar convergencia. No se cambia la norma
  de aceptación por una norma precondicionada más permisiva.

### Precisión interna ligada al objetivo final

Se prueba `inner_forcing='ew_target'`: partiendo del término EW existente,

```text
η_k = min(0.3, max(η_EW, 0.5 * tolerancia_exterior / error_actual))
||b - A δ|| <= η_k ||b||
```

El error exterior es `RMS(P σ)/σ_y`. La transformación Q es isométrica, por
lo que el cociente corresponde a la misma norma en el CG reducido. El factor
0,5 deja margen para el error del modelo lineal de Newton. Si la respuesta
constitutiva real no satisface el residuo exterior, Newton continúa; no se
confirma el estado por cumplir solamente esta condición interna. PTC mantiene
su forcing previo. No se rebaja la tolerancia exterior ni se omiten puntos de
la curva.

## Ablación preliminar de fusión/reutilización

Una curva completa por variante; no confundir con una mediana de repeticiones.
Archivo: `results/g09_nonlinear_fusion_pilot_20260923/report.json`.

| Variante | Tiempo (s) | CG |
|---|---:|---:|
| Original | 78,015 | 5.603 |
| Reutilización solamente | 77,035 | 5.603 |
| PCG fusionado solamente | 70,828 | 5.604 |
| Ambas | 69,911 | 5.604 |

La reutilización da una curva idéntica bit a bit. La fusión cambia el orden de
operaciones flotantes: diferencia máxima de esfuerzo 1,07e-7 MPa, normalizada
por 15 MPa = 7,12e-9. Todas las auditorías pasan; máximo residuo 9,77e-6.
La fusión por sí sola **no reduce el número de CG**: reduce su coste.
El pool CUDA reservado fue 6.331.701.248 bytes en este proceso compartido;
no es una medida aislada de memoria activa máxima por variante.

## Resultado confirmado e integración

`results/g09_nonlinear_target_forcing_20260923/report.json`: tres repeticiones
por variante, orden alternado, GPU sin otro trabajo concurrente.

| Variante | Repeticiones (s) | Mediana (s) | CG | Newton |
|---|---|---:|---:|---:|
| Original | 77,991; 78,099; 78,108 | 78,099 | 5.603 | 100 |
| Fusión + reutilización + EW objetivo | 61,324; 61,323; 61,331 | 61,324 | 4.839 | 100 |

Reducción de tiempo **21,48 %** (1,274×), 16,775 s por curva. Reducción CG
**13,64 %**, 764 iteraciones. En el último Newton del incremento 20, el
forcing pasa de aproximadamente 0,008 a 0,3: 135 → 36 CG en esa corrección.
La reducción global no se deduce de ese único paso: se mide la curva completa.

Auditorías de la combinación: cero rechazos, 40 incrementos, máximo residuo
9,9012e-6, máximo esfuerzo medio libre normalizado 2,8787e-7, máximo
Hill–Mandel 2,3397e-7, disipación no negativa y máxima violación de fluencia
normalizada 2,5146e-6. Diferencia máxima de los seis esfuerzos medios:
**1,7646e-5 MPa**, igual a 1,1764e-6 de la escala de 15 MPa
(0,00011764 %, frente al límite 0,1 %). Diferencia máxima de deformaciones
medias 4,7474e-10 y de plasticidad equivalente media 3,5581e-9.

Se integra por defecto mediante `coordinate_pcg='auto'`,
`reuse_trial_response=True`; `inner_forcing` elige `ew_target` en la ruta
fusionada compatible y `ew` en las demás. CPU conserva su referencia clásica.
No se cambió el default lineal en esta integración. La ruta previa se puede
seleccionar explícitamente para ablar las mejoras.

```bash
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/run_nonlinear_fft.py \
  --case g09 --backend cupy --output FFT/benchmarks/results/mi_curva_nueva

OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_nonlinear_fusions.py \
  --variants baseline target_combined --repeats 3 \
  --output FFT/benchmarks/results/mi_comparacion_nueva
```

Verificación adicional en inclusión sintética 12³, carga–descarga:
`results/nonlinear_fused_target_validation_20260923/validation.json`.
Diferencia CPU64/GPU32 = 2,6367e-6 de 15 MPa; al dividir el incremento por
dos en CPU = 8,6508e-5 de 15 MPa. Ambas menores que 0,001. Máximos residuos
CPU 9,9712e-9 y GPU 9,9722e-6. No extrapolar esta prueba pequeña a convergencia
temporal de G09. Regresión conjunta lineal/no lineal: **223 pruebas pasan**,
incluyendo tangente, descarga, estados inmutables, rechazo tras Newton
aceptado, PTC desplazado, reinicio y detección de falso residuo recursivo.

### Perfil de la ruta integrada

`results/g09_nonlinear_optimized_profile_20260923_v2/report.json`: 61,442 s
instrumentados, 4.839 CG y curva auditada contra la referencia. No es otra
mediana; las tres repeticiones sin instrumentar están arriba.

| Trabajo | Original (s) | Integrado (s) |
|---|---:|---:|
| FFT | 35,228 | 30,250 |
| Acción de tangente | 9,286 | 8,036 |
| Conversiones de coordenadas | 8,462 | 7,278 |
| Actualizaciones + referencia + reducciones de CG | 21,212 | 12,208 |

La última fila agrupa categorías equivalentes: en el nuevo camino las
reducciones `rr/rz` forman parte de la actualización y la referencia de la
dirección. No atribuir la caída del contador `dot` únicamente a eliminar
reducciones, porque varias se trasladan al kernel fusionado. Se reduce una
sincronización de host por iteración no terminal. Evaluaciones constitutivas
241 → 141, llamadas a cada transformada 6.184 → 5.320, acciones de tangente
5.703 → 4.939. La reducción de llamadas FFT combina menor CG y la eliminación
de 100 evaluaciones de equilibrio duplicadas.

Las FFT siguen representando aproximadamente la mitad del tiempo; después
conviene atacar el kernel de tangente y el mapeo espectral. Son objetivos
de desarrollo posteriores, no ahorros adicionales ya demostrados. Lanzar
incrementos plásticos independientes en paralelo no elimina este coste sin
cambiar el método de integración.

Reproducir la captura optimizada contra metadatos de una curva original:

```bash
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/profile_nonlinear_stages.py \
  --coordinate-pcg auto --reuse-trial-response --inner-forcing ew_target \
  --modes profiled --reference /ruta/baseline/metadata.json \
  --output FFT/benchmarks/results/mi_perfil_nuevo
```

Nota de trazabilidad: el primer intento de esta captura resolvió la curva,
pero falló al comparar `lengths` como tupla en memoria y lista en JSON. Se
normalizó la serialización y se añadió una prueba de regresión del comparador
del perfil. Sólo la captura `v2` tiene informe completo; no se contabiliza el
intento fallido como repetición de rendimiento. El primer intento del
benchmark lineal también se detuvo antes de resolver por exigir arrays de
geometría de sólo lectura; la evidencia lineal válida es la carpeta `v2`.

## Transferencia al solver lineal

Se probó la misma fusión del precondicionador, con todos los modos medios
fijos, sin modificar producción. Tres repeticiones por variante, orden rotativo:
`results/g09_linear_fused_reference_20260923_v2/report.json`.

| Ruta lineal, seis cargas | Mediana (s) | CG |
|---|---:|---:|
| Actual: simétrica fusionada | 3,4143 | 288 |
| PCG de matriz anterior | 3,7906 | 288 |
| Candidato PCG de matriz fusionado | 3,4241 | 288 |

La propuesta mejora al PCG antiguo, **no al default actual**. Diferencias
relativas de C_eff ≤ 6,30e-8; residuo verdadero máximo ≤ 9,274e-6.
No se cambia el default lineal ni se afirma una mejora lineal nueva. El
benchmark incluye preparación del solve y metadatos de timing, sin exportar
campos; geometría constitutiva compilada precargada y planes calientes.

Gráfica de equivalencia de las curvas no lineales:
[`comparison.png`](results/g09_nonlinear_optimized_curve_20260923/comparison.png).
Curva exportada: [`curve.csv`](results/g09_nonlinear_optimized_curve_20260923/optimized/curve.csv),
con metadatos y gráfica en la misma carpeta; procede de la primera repetición
combinada, sin resolver ni alterar nuevamente el material.

## Literatura y decisiones

1. [Lucarini–Segurado, DBFFT (2019)](https://arxiv.org/html/1905.12725v2):
   formulación en desplazamientos y condiciones mixtas compatibles con
   precondicionamiento. Apoya la ruta actual; no exige cambiar a FEM.
2. [Eisenstat–Walker, forcing en Newton inexacto](https://petsc.org/release/manualpages/SNES/SNESKSPSetUseEW/):
   adaptar precisión de la solución lineal al progreso no lineal. El
   [código oficial PETSc, variante 3](https://petsc.org/release/src/snes/interface/snes.c.html)
   añade una protección contra sobre-resolver basada en la tolerancia final
   dividida por la norma actual. Es pertinente al Newton final de G09: por
   ejemplo, error 1,50e-5, objetivo 1e-5, pero η=0,008 y 135 CG.
3. [Schneider (2020), CG no lineal sin line search](https://link.springer.com/article/10.1007/s00466-020-01849-7):
   alternativa que evita Newton–CG anidado, interesante si integrar material
   resulta barato. **No implementada ni medida aquí**; exigiría validar otra
   globalización y demostrar menos evaluaciones totales. No asumir que
   reemplazar CG lineal por no lineal siempre acelera.
4. [Ladecký et al. (2023), precondicionamiento FFT–FE](https://arxiv.org/abs/2203.02962):
   útil para estudiar escalado y precondicionadores de referencia; adoptar
   su discretización FE cambiaría el problema actual y queda fuera de esta
   comparación de latencia.
5. [NVIDIA, CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
   accesos contiguos, eliminar lecturas/escrituras redundantes y agrupar
   transferencias pequeñas. Motiva la fusión de recorridos y reducciones,
   no una promesa de aceleración sin medición.
6. [Ghysels–Vanroose, CG pipelined](https://www.sciencedirect.com/science/article/abs/pii/S0167819113000719)
   y [análisis de precisión de Cools et al.](https://arxiv.org/abs/1601.07068):
   solapar comunicaciones globales con aplicación de operador/precondicionador
   es relevante para sistemas distribuidos. No equivale a nuestra fusión
   local de kernels. La [variante de una reducción en PETSc](https://petsc.org/main/manualpages/KSP/KSPCGUseSingleReduction/)
   requiere dos vectores de trabajo adicionales; las recurrencias pipelined
   también requieren atención al residuo verdadero y al error de redondeo.
   **No se ha implementado ni medido aquí**. En esta GPU, aumentar tráfico de
   vectores podría contrarrestar el ahorro de sincronizaciones: hay que medirlo.

### Revisión 2026 y FANS: qué trasladar y qué no

Se volvió a consultar el texto de [UNO-CG, Herb–Fritzen (2026)](https://onlinelibrary.wiley.com/doi/10.1002/nme.70277).
Su §7.3 delimita el estudio a PDE lineales paramétricas; extenderlo a plasticidad
es trabajo futuro. No acredita acelerar esta trayectoria J2. Sus §§4.4–4.5
sí aportan una estrategia interesante para muchas consultas: preprocesar datos
y entrenar un precondicionador SPD con derivadas analíticas, amortizando ese
coste. El entrenamiento O(N) tras preprocesar no convierte la aplicación FFT
del solver en O(N). El prototipo local anterior por bandas no redujo CG;
no es una reproducción completa ni una refutación de UNO-CG.

Se verificaron también el [artículo original FANS](https://link.springer.com/article/10.1007/s00466-017-1501-5)
y su [repositorio oficial](https://github.com/DataAnalyticsEngineering/FANS).
FANS utiliza incógnitas nodales y una discretización que converge a FEM;
su implementación C++ incorpora MPI/FFTW. El cambio directo no sería una
optimización equivalente de nuestro operador espectral. Se retienen las ideas
de referencia precondicionada y trabajo matrix-free; no se sustituye el solver
ni se le atribuyen tiempos medidos en otro código. No se instaló FANS.

## Paralelismo

Ya se paralelizan vóxeles, FFT y reducciones en GPU. Los incrementos de una
curva plástica dependen del estado confirmado anterior: no pueden lanzarse
como problemas independientes conservando esta integración. En una GPU,
múltiples curvas compiten por ancho de banda y memoria. La evaluación previa
de lote 2 mejoró throughput, pero no latencia de una curva: no se activa aquí.
Varias GPU sí permitirían distribuir curvas/materiales independientes; no se
ha medido escalado multi-GPU en esta estación.

La equivalencia entre implementaciones a 40 incrementos no certifica
convergencia espacial/temporal ni calibración experimental. La comparación
histórica G09 40/80 incrementos dio aproximadamente 0,24 % de 15 MPa, superior
al objetivo temporal de 0,1 %; esa limitación no desaparece al acelerar CG.
