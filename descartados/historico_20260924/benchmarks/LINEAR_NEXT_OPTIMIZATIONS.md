# Perfil del FFT lineal y siguientes optimizaciones

**Actualización:** las tres primeras estrategias ya se probaron por separado.
Véase [resultados de G09](THREE_LINEAR_STRATEGIES_RESULTS.md): transformación
simétrica 3,526 s; actualización fusionada 3,679 s; anisotrópica 4,231 s,
frente a 3,794 s del PCG actual. Los prototipos no están activados en producción.
Una campaña posterior probó también la combinación de las dos primeras:
[3,419 s, resultados y validación](COMBINED_LINEAR_STRATEGY_RESULTS.md).

## Resultado medido, 2026-09-22

G09 original 240³, seis cargas, RTX 5070 Ti (16 GB), CuPy float32,
residuo físico máximo permitido 1e-5. Geometría compilada y planes calientes;
sin lectura de geometría ni exportación de campos dentro del tiempo.

La API ya selecciona `displacement` automáticamente para esta configuración.
Se ejecutaron tres perfiles, cada uno entre dos resoluciones sin instrumentar:
mediana sin instrumentar **3,8015 s**, mediana instrumentada **3,8003 s**.
La diferencia está dentro de la dispersión; no es una aceleración nueva.
Las nueve resoluciones dieron 288 CG, residuo máximo 9,273e-6 y exactamente
el mismo C_eff entre sí. Esto verifica que el perfil no cambió el resultado;
no sustituye la comparación de precisión frente a CPU64 o frente a `strain`.
Las 114 pruebas de regresión seleccionadas pasaron tras cambiar el default.

Evidencia: [perfil por etapas](results/g09_linear_stage_profile_20260922/report.json),
[kernels CUDA](results/g09_linear_stage_profile_20260922/cuda_summary_cuda_gpu_kern_sum.json),
[transferencias](results/g09_linear_stage_profile_20260922/cuda_summary_cuda_gpu_mem_time_sum.json),
[API CUDA](results/g09_linear_stage_profile_20260922/cuda_summary_cuda_api_sum.json).

## Dónde está el tiempo

Una captura independiente de Nsight Systems, después del calentamiento,
duró 3,7948 s. Sus tiempos de kernels, agrupados, son:

| Trabajo | Tiempo GPU | Aproximación al total de pared |
| --- | ---: | ---: |
| FFT directas e inversas, incluidas preparación/auditorías | 1,774 s | 46,8 % |
| Actualizaciones de x, r y p de CG | 0,526 s | 13,9 % |
| Kernel de productos internos `warp_dot` | 0,432 s | 11,4 % |
| Conversión de coordenadas compatibles | 0,391 s | 10,3 % |
| Aplicación constitutiva indexada | 0,340 s | 9,0 % |
| Aplicación del precondicionador | 0,114 s | 3,0 % |

El resto incluye otros kernels, copias, puesta a cero y huecos de ejecución.
La preparación del campo no debe confundirse con sus 312 aplicaciones durante
la resolución. El campo ya tiene una tabla compacta: no se almacena una matriz
global de rigidez.

El perfil por eventos obtiene 1,706 s en las FFT internas de coordenadas;
Nsight incluye también las FFT de preparación y auditoría. Los tiempos por
eventos incluyen huecos de lanzamiento y los tiempos inclusivos se solapan
con los hijos: no se suman entre sí.

**No interpretar el tiempo de espera como tiempo de transferencia.** Nsight
registra 3,627 s dentro de `cudaMemcpyAsync` en CPU, pero las copias reales
host→device y device→host suman sólo **2,869 ms** en GPU. La CPU espera el
trabajo anterior al extraer escalares. Mantenerlos en GPU no elimina ese
trabajo. Tampoco basta cambiar Python por C++: los kernels principales ya
ejecutan código CUDA compilado.

## Prioridad 1: precondicionamiento simétrico fusionado

Propuesta matemática derivada para nuestro operador, ahora probada en un
prototipo aislado (véase el informe anterior). Sea H = Q* C Q el operador actual y M la referencia
homogénea SPD. Con R = M^(-1/2), resolver

    (R H R) y = R b,    z = R y

con CG ordinario es equivalente al PCG original en aritmética exacta. No cambia
C, la malla ni el subespacio admisible. Para la referencia isotrópica actual,
por frecuencia no nula y admitida:

    R = sqrt(a) I + (sqrt(b) - sqrt(a)) n nᵀ
    a = 1/(2 μ),    b = 1/(K + 4 μ/3),    n = k/|k|

R puede absorberse en las operaciones Q R y R Q*: modificar los coeficientes
del kernel de coordenadas, en lugar de recorrer de nuevo un campo completo.
CG ordinario tampoco necesita el producto adicional rᵀ M⁻¹r del PCG actual.
Es decir, se busca quitar un recorrido de precondicionamiento y una reducción
por iteración. Los bloques candidatos cuestan del orden de 0,11 + 0,14 s en
este perfil; **no es un ahorro garantizado**: hay operaciones adicionales,
redondeo y posibles cambios en las iteraciones o correcciones.

La norma del residuo transformado no es la norma física. Se debe conservar
la auditoría en las ecuaciones originales, los ceros de Nyquist y modo medio,
las reducciones float64 y el mecanismo de corrección. Verificar simetría,
positividad, equivalencia CPU64 en mallas pequeñas y las seis cargas G09.

El precondicionador de referencia diagonal por bloques en Fourier está
fundamentado en [DBFFT, Lucarini y Segurado (2019), §2.5](https://arxiv.org/html/1905.12725v2)
y en [Ladecký et al., formulación bien escalada (2022/2023)](https://arxiv.org/html/2203.02962v1).
La fusión concreta anterior es una propuesta para este código, no un resultado
experimental atribuido a esos artículos. El segundo usa FE; no adoptamos su
discretización como sustitución silenciosa de la espectral.

Puntos de intervención: `pipeline/linear_coordinates.py` para el Krylov
lineal y `DisplacementCoordinates._map` en `pipeline/nonlinear_coordinates.py`
para la conversión. Este último se comparte con el no lineal: cualquier futura
variante debe ser explícita y mantener su comportamiento actual por defecto.
No concentrar la siguiente campaña sólo en `tensors/operators.py`.

## Otras estrategias, por orden

1. **Fusionar actualizaciones y reducciones.** Evitar releer r para calcular
   su norma después de actualizarlo. Preservar la acumulación float64; comparar
   reducción fusionada con el `warp_dot` actual. El objetivo es tráfico de
   memoria, además de lanzamientos. [Rupp et al. (2016)](https://arxiv.org/html/1410.4054v3)
   estudian fusión y solapamiento en GPU. Sus aceleraciones no son predicciones
   para este FFT: las mejoras mayores del paper corresponden a sistemas más
   pequeños. Primero probar fusiones con la recurrencia clásica.
2. **Referencia anisotrópica para bajar iteraciones.** Usar un C₀ homogéneo SPD
   que refleje las orientaciones y bloques 3×3 por frecuencia, transformados a
   nuestras coordenadas. [DBFFT, ecuaciones 39–40](https://arxiv.org/html/1905.12725v2)
   y [Ladecký et al., §4](https://arxiv.org/html/2203.02962v1) motivan la prueba.
   Medir construcción, memoria y las seis cargas completas. El promedio
   isotrópico ya perdió en la campaña no lineal; no asumir que promediar mejora.
3. **Reciclaje/deflación muy pequeño entre seis cargas lineales.** Aquí H no
   cambia, a diferencia de la tangente plástica. [Stathopoulos y Orginos,
   eigCG (2010)](https://arxiv.org/html/0707.0131v2) reutilizan información
   espectral en sucesivos RHS. Sus experimentos amortizan sobre muchos RHS;
   nosotros sólo tenemos seis. Cada vector complejo de coordenadas G09 ocupa
   unos 167 MB decimales: ocho vectores son 1,34 GB, sin contar H aplicado a
   ellos. Ortogonalizar puede costar más que lo ahorrado. No restaurar la
   antigua variante no lineal descartada.
4. **CG pipelined sólo después.** Puede reducir sincronizaciones, pero introduce
   recurrencias auxiliares y riesgo de pérdida de precisión. [Cools et al.
   (2017)](https://arxiv.org/abs/1601.07068) analizan ese problema y proponen
   reemplazar el residuo. No aceptarlo usando sólo el residuo recursivo.

No priorizar J-FFT/Green–Jacobi para esta geometría binaria:
[Ladecký et al., §4.4](https://arxiv.org/html/2508.02613v2) muestran que en su
comparación de interfaces abruptas gana Green; el beneficio de J-FFT se centra
en propiedades suaves. Esto orienta la prioridad, no demuestra imposibilidad
de mejorar G09. Tampoco modificar discretización o eliminar vóxeles en esta
campaña: cambiaría el problema de referencia solicitado.

## Reproducir el perfil

```bash
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/profile_linear_stages.py \
  --repeats 3 --output /tmp/g09-linear-stages-new
```

El script añade instrumentación sólo durante el benchmark y la restaura al
salir. Los JSON guardan selección efectiva, auditorías, hardware, versión
CuPy y hashes del código. Las nuevas estrategias fueron medidas por separado
en el benchmark posterior, no activadas en producción. La ruta no lineal y
sus 40 incrementos permanecen sin cambios.
