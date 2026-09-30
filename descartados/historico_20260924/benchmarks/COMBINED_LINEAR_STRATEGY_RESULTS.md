# Combinación de precondicionamiento simétrico y reducción fusionada

**Actualización posterior:** la combinación ya se integró en producción como
`linear_preconditioning='symmetric_fused'` (default del lineal compatible).
[Verificación de la integración](POLARIZATION_GPU_AND_2026_REVIEW.md).
El resto de este documento conserva los resultados del prototipo original.

## Resultado — G09 240³, 2026-09-22

La combinación es la más rápida de las cuatro opciones medidas en esta sesión.
RTX 5070 Ti, CuPy float32, seis cargas y tolerancia física 1e-5. Cuatro
repeticiones por variante con orden rotativo equilibrado: cada variante ocupó
cada posición una vez. Planes/kernels calientes, geometría compilada en memoria,
sin exportación de campos ni otras pruebas GPU simultáneas. Se incluyen
preparación dentro del solve, auditorías y escritura de metadatos.

| Variante | Mediana | CG totales |
| --- | ---: | ---: |
| Ruta actual: PCG de matriz | 3,7925 s | 288 |
| Precondicionamiento simétrico, por separado | 3,5251 s | 288 |
| Actualización/reducción fusionada, por separado | 3,6746 s | 288 |
| **Combinación** | **3,4188 s** | **288** |

La combinación reduce el tiempo **9,85 % frente a la ruta actual** y **3,02 %
frente a la mejor variante individual**. Su rango fue 3,4124–3,4258 s. No se
sumaron aceleraciones estimadas: estos valores proceden de solves completos.
No es una nueva reducción del número de iteraciones, sino de su coste.

## Qué se combinó

Se utiliza la transformación simétrica R H R, con R = M^(-1/2), fusionada en
las conversiones de coordenadas. Sobre ese sistema se ejecuta CG ordinario con
un kernel que actualiza x/r y acumula la norma de r en float64. No se vuelve
a aplicar M⁻¹ como precondicionador: eso duplicaría el precondicionamiento.
Se conservan el recálculo del residuo verdadero y la auditoría final de las
ecuaciones físicas originales. No cambian malla, materiales ni tolerancias.

## Verificación

- Las 16 resoluciones medidas (96 cargas) cumplieron las comprobaciones físicas.
- Combinación: residuo físico máximo **9,2716e-6**, menor que 1e-5.
- Diferencia relativa de C_eff frente a la ruta actual: **6,2931e-8**, menor
  que el umbral comparativo de 1e-6. Es una comparación entre rutas, no una
  estimación del error de discretización ni una referencia G09 CPU64.
- Ninguna variante requirió CG corrector adicional.
- **134 pruebas superadas**, incluidas la combinación frente a CPU64 en mallas
  pequeñas pares/impares, RHS nulo, fallo por límite de iteraciones, restauración
  de los parches experimentales y regresión del lineal/no lineal.
- Los hashes del código del informe se contrastaron con los archivos usados.

La combinación sigue como **prototipo de benchmark**, sin modificar el default
de producción. Es ahora la candidata recomendada para integrar. La ruta no
lineal permanece intacta.

## Evidencia y reproducción

[Reporte completo](results/g09_combined_linear_strategies_20260922/report.json):
tiempos por repetición, C_eff completos, diagnósticos por carga y hashes.

```bash
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_linear_strategies.py \
  --modes baseline symmetric fused_updates combined --repeats 4 \
  --output /tmp/g09-combined-new

OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_linear_strategy_candidates.py
```

Las pruebas utilizan parches temporales en un proceso dedicado y secuencial.
No son una API concurrente de producción. La integración definitiva deberá
eliminar esos parches y repetir las comprobaciones de tiempo y precisión.
