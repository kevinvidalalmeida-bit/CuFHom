# Segundo perfil: optimizar el coste de los operadores

Campaña lineal posterior: [reciclaje espectral y cota energética](LINEAR_RECYCLING_TRIALS_20260924.md).
Diez configuraciones ensayadas; ninguna mejora G09. Confirmación: producción
3,168 s / 288 CG; arranque reciclado 3,427 s / 288 CG; corrección espectral
3,842 s / 292 CG. Sin cambios en defaults ni tolerancia física.

Campaña posterior: [pruebas de fusión, referencia tangente y cuFFTDx](FFT_LOWLEVEL_TRIALS_20260924.md).

Campaña espacial: [16 precondicionadores por bloques probados en G09](SPATIAL_PRECONDITIONER_TRIALS_20260924.md).
Ninguno mejora el PCG actual; se conservan los defaults.
La fusión experimental `p.Ap` reduce la curva de 57,801 a 56,362 s en
comparación repetida; todavía no cambia los defaults. El informe también
analiza la distribución actual de los 40 incrementos, sin modificarla.

Alcance: G09 original 240³, RTX 5070 Ti, CuPy float32, curva demostrativa
0→1 %, 40 incrementos uniformes y residuo físico ≤1e-5. No se cambia
discretización, estado plástico, referencia de material ni trayectoria.
Los resultados de esta campaña no calibran la resina real.

**Resultado confirmado:** FFT por componente integrada, mediana no lineal
61,199→57,769 s (5,61 % menos); seis cargas lineales 3,417→3,164 s
(7,40 % menos). Misma curva/C_eff en los valores guardados, mismos CG y
mismas tolerancias. No atribuir estos ahorros al precondicionador J-FFT del
paper: ese método todavía no se ha implementado.

## Integración y reproducibilidad

[Tres repeticiones por variante no lineal](results/g09_hotspot_curves_confirm_20260923/report.json):
referencia 61,197 / 61,219 / 61,199 s; candidato 57,778 / 57,769 / 56,838 s.
Se informa mediana y no el mejor tiempo; la última repetición rápida no se
oculta ni se usa como aceleración representativa. 4.839 CG, 100 Newton,
40 incrementos, cero rechazos y diferencia de curva cero en todas las corridas.
[Seis cargas lineales, tres repeticiones](results/g09_linear_fft_batches_20260923/report.json):
288 CG, error C_eff cero, residuo físico máximo 9,272e-6.

Opción compartida de ambas API: `cupy_rfft_batch_mode='auto'` (predeterminada).
Sólo selecciona componentes en la configuración medida: RTX 5070 Ti,
float32, seis campos contiguos 240³ y planes manuales. `full` permite volver
al lote anterior; `component` sirve para medir explícitamente otras mallas
compatibles. CPU, otros dispositivos, tamaños o precisiones conservan el
lote original en `auto`. Las mallas con volumen impar o punteros sin la
alineación compleja requerida no usan el camino nuevo. La inversa sin escalar
puede consumir su entrada, igual que antes; cada salida tiene almacenamiento
propio. Los planes incluyen el dispositivo en su clave de caché.

Tiempos calientes, sin otro trabajo GPU concurrente. No lineal incluye
preparación de geometría/estado, resolución, auditorías y commits; excluye
exportación de campos, serialización y gráficos. Lineal utiliza geometría
precompilada, incluye escritura de metadatos y excluye exportación de campos.
No sumar estos tiempos como si midieran exactamente el mismo trabajo.

```bash
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_hotspot_curves.py \
  --output /tmp/g09-hotspots-new --variants baseline split_fft --repeats 3
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_linear_fft_batches.py \
  --output /tmp/g09-linear-batches-new --repeats 3
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/profile_nonlinear_stages.py \
  --output /tmp/g09-integrated-new --coordinate-pcg auto \
  --reuse-trial-response --inner-forcing ew_target
```

Para una curva usual: `run_nonlinear_fft.py --case g09 --backend cupy`; para
comparar con el lote viejo, añadir `--cupy-rfft-batch-mode full`.

## Perfil de partida

[Perfil fresco](results/g09_nonlinear_hotspots_baseline_20260923/report.json):
61,2858 s, 4.839 CG y 100 Newton; misma curva que la integración precedente.

| Etapa | Antes, s | Integrado, s |
| --- | ---: | ---: |
| RFFT + inversa sin escalar | 30,195 | 26,760 |
| Aplicación de tangente | 8,024 | 8,004 |
| Conversión de coordenadas | 7,252 | 6,615 |
| Actualización PCG y reducción conjunta | 6,484 | 6,487 |
| Dirección PCG | 3,282 | 3,280 |
| Productos internos y sincronización | 2,367 | 3,109 |
| Integración constitutiva J2 | 0,322 | 0,324 |

Son intervalos exclusivos instrumentados, no tiempo activo puro de kernels.
No sumar los padres inclusivos CG/operador con estas etapas. Las esperas del
host al leer escalares contienen trabajo GPU previo, no son todas transferencias.
El mayor número de lanzamientos FFT cambia también dónde caen los huecos
entre kernels; no se modificaron coordenadas/reducciones en producción.
No atribuir su redistribución de eventos a una optimización de esos kernels.
[Perfil integrado](results/g09_component_fft_integrated_20260923/report.json):
57,738 s sin instrumentar antes, 57,942 s instrumentado y 57,788 s sin
instrumentar después. Las tres curvas coinciden con la referencia guardada.

## Validación

Regresión final: **261 pruebas aprobadas**, 6,14 s; sólo el aviso preexistente
de deprecación `numpy.matlib`. `git diff --check` sin errores. Los hashes de
los archivos de producción coinciden con los metadatos del perfil integrado.

Además de la curva G09 y el tensor lineal, se comprueban FFT32/64,
dimensiones pares/impares/no cúbicas, alineación, propiedad de buffers,
restauración de configuración, medios libres y campos/estado tras descarga.
[Validación independiente de inclusión 12³](results/nonlinear_component_fft_validation_20260923/validation.json),
forzando `component`: error curva CPU/GPU 2,637e-6 sobre escala 15 MPa;
variación al dividir el incremento 8,651e-5; ambos menores de 0,001.
Residuos máximos CPU 9,971e-9 y GPU 9,972e-6.

No es certificado de convergencia temporal de G09. Su estudio previo de
40 frente a 80 pasos daba ≈0,24 % de la escala de 15 MPa; esta optimización
conserva la solución de 40 pasos, no corrige ese error de discretización de
la trayectoria. [Comparación gráfica de la curva optimizada](results/g09_component_fft_curve_comparison_20260923/comparison.png).

## Cribado y piloto

[Microbenchmarks](results/g09_gpu_hotspots_screen_20260923.json) con tablas
reales G09 y campos sintéticos; siete mediciones de diez llamadas calientes.
No confundir estas mediciones con la curva completa.

| Candidato | Observación | Decisión |
| --- | --- | --- |
| Lote cuFFT de un componente | Pareja FFT 5,784→5,061 ms en el cribado | Llevar a curva completa |
| Lotes de dos/tres componentes | ≈5,77 ms | No integrar |
| Tangente con valores en registros y rama por fase | 1,559→1,538 ms | Ventaja pequeña; comprobar combinada |
| Coordenadas con dimensiones constantes | Diferencia pequeña | Comprobar combinada |
| PCG con dimensiones constantes y 2.048–16.384 bloques | No mejora consistente | No integrar |

La pareja FFT del cribado intercala directa/inversa por componente. El solver
no puede hacer eso: necesita el campo completo para la tangente. El piloto
completo usa seis directas o seis inversas consecutivas y valida el beneficio
en la operación real, incluyendo asignación de salidas y lanzamientos.

[Piloto](results/g09_hotspot_curves_pilot_20260923/report.json), una curva por variante:
61,210 s original; 57,789 s FFT por componente; 57,740 s al añadir registros y
coordenadas especializadas. Todas dieron 4.839 CG y diferencia de curva cero.
La diferencia adicional de 0,049 s no justifica más kernels de producción.
[Cribado PCG](results/g09_pcg_geometry_screen_20260923.json).

La sustitución FFT es algebraicamente exacta: la transformada de seis campos
es `I₆ ⊗ F_N`. Ejecutar sus seis bloques por separado no elimina frecuencias
ni mezcla componentes materiales. Se mantienen RFFT, la inversa sin escalar y
el escalado que ya realiza el proyector. No se añade `fast_math`, precisión
reducida ni compartición de estados plásticos. La igualdad bit a bit observada
en G09 no se promete para otras versiones de cuFFT o tamaños de malla.

## Literatura y siguientes prioridades

Revisión del artículo que compartiste:
[`JACOBI_GREEN_REVIEW_20260923.md`](JACOBI_GREEN_REVIEW_20260923.md).
Conclusión: candidato para bajar CG después de plastificar, no una aceleración
universal ni un algoritmo O(N). Hay que adaptar su diagonal de desplazamientos
al espacio espectral y condiciones medias del solver. No se ha integrado ni
se atribuyen a ese paper los tiempos medidos arriba.

- [NVIDIA cuFFT, rendimiento y precisión](https://docs.nvidia.com/cuda/cufft/index.html#accuracy-and-performance):
  recomienda contigüidad, transformadas reales y medición de planes/lotes.
  Aunque recomienda batching en general, no garantiza que el mayor lote
  sea el óptimo de esta GPU. Aquí se decide con curvas completas. 240 ya
  factoriza en 2, 3 y 5; no se modifica la malla para favorecer a cuFFT.
- [AutoMat, Blühdorn–Gauger–Kabel](https://link.springer.com/article/10.1007/s00466-021-02105-2):
  evalúa paralelización constitutiva, coste de registros y compromiso
  memoria/cómputo. En nuestro J2 la integración ya es sólo 0,322 s y está
  paralelizada por vóxel: no sería el primer objetivo para reducir decenas
  de segundos. Tampoco conviene construir una tangente densa 6×6 por vóxel.
- [DBFFT, Lucarini–Segurado, §2.5](https://arxiv.org/html/1905.12725v2):
  fundamenta precondicionamiento en desplazamientos y referencia homogénea.
  Siguiente experimento matemático: referencia SPD que siga la tangente
  plástica, congelada dentro de cada CG. Medir coste de construcción y
  aplicación, además de iteraciones; conservar auditoría del residuo real.
  **Propuesta pendiente, no ahorro demostrado.**
- [Gehrig–Schneider, EBI (2025)](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.70170):
  reduce evaluaciones/variables internas de discretizaciones FE con varios
  puntos de cuadratura por elemento. Nuestro solver ya tiene un punto
  material por vóxel: no autoriza compartir estado entre vóxeles del mismo
  material ni es una optimización directamente aplicable sin cambiar el
  modelo discreto. No se implementa en esta comparación.

Actualización de la revisión de bajo nivel:
[`FFT_LOWLEVEL_REVIEW_20260923.md`](FFT_LOWLEVEL_REVIEW_20260923.md).
El prototipo aislado de mapa adjunto + pᵀAp reduce esa etapa de 1,121 a
0,912 ms, con reducciones FP64 y modos medios. No está integrado en CG ni
se atribuye ese porcentaje a la curva completa. CUDA Graphs sobre el operador
no dieron una mejora clara y empeoraron al incluir la copia de entrada.
El camino de producción y los tiempos confirmados arriba no cambian.
Los incrementos plásticos continúan secuenciales, pues
dependen del estado confirmado anterior. Paralelizar las operaciones por
vóxel/frecuencia no equivale a resolver esos incrementos simultáneamente.
