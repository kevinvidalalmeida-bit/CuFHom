# Prueba de las tres estrategias lineales — G09 240³

Actualización posterior: [la combinación ya fue probada](COMBINED_LINEAR_STRATEGY_RESULTS.md)
y dio 3,419 s; es ahora la candidata más rápida. Este documento conserva
los resultados de la comparación individual anterior.

## Resultado

RTX 5070 Ti, CuPy float32, seis cargas, mismas propiedades, geometría uniforme
original y tolerancia física 1e-5. Tres repeticiones por variante en orden
rotativo, después de calentar cada ruta, sin otras pruebas GPU simultáneas.
Se usa la misma geometría compilada en memoria; se excluyen lectura, compilación
geométrica y exportación de campos. Las preparaciones específicas de las
variantes y las auditorías físicas sí están incluidas en cada resolución.

| Variante | Mediana | Rango de tres repeticiones | CG totales | Cambio frente a actual |
| --- | ---: | ---: | ---: | ---: |
| Actual: PCG con referencia de matriz | 3,7944 s | 3,7921–3,7958 s | 288 | — |
| 1. Precondicionador simétrico fusionado | **3,5259 s** | 3,5252–3,5293 s | 288 | **7,08 % menos** |
| 2. Actualización x/r y norma fusionadas | 3,6792 s | 3,6784–3,6829 s | 288 | 3,03 % menos |
| 3. Referencia anisotrópica promedio | 4,2306 s | 4,2279–4,2347 s | 318 | 11,50 % más |

**Conclusión de esta comparación individual:** la primera fue la mejor.
La segunda también mejoró por separado. En esta campaña no se combinaron;
su combinación se midió después en el informe enlazado arriba. La tercera
no se recomienda para esta G09.

Las tres viven exclusivamente en el módulo de experimentos. **No se cambió
la ruta de producción ni el solver no lineal durante estas pruebas.** El
default sigue siendo la formulación de desplazamiento con PCG de matriz
integrada previamente, no estos nuevos prototipos.

## Qué se probó realmente

1. **Transformación simétrica:** resolver R H R y = R b, con H = Q* C Q,
   R = M^(-1/2), y reconstruir z = R y. La raíz isotrópica se incorpora al
   mismo kernel de coordenadas; se usa CG sin un precondicionador separado.
   Desaparecen el recorrido del precondicionador y el producto rᵀ M⁻¹r por
   iteración. No se eliminan frecuencias ni vóxeles. La auditoría final sigue
   usando las ecuaciones originales, no sólo el residuo transformado.
2. **Fusión de actualización y norma:** se mantiene el PCG de matriz original.
   Un único kernel actualiza x/r y acumula rᵀr en float64, evitando releer r.
   Se conserva el recálculo del residuo verdadero y el reinicio fiable del CG.
   Es una fusión de la recurrencia clásica, no un CG pipelined nuevo.
3. **Referencia anisotrópica:** se calcula el C₀ promedio ponderado por el
   número real de vóxeles de cada orientación. Se construye e invierte
   Q(k)* C₀ Q(k) en cada frecuencia admisible. La construcción usa float64;
   las seis entradas independientes de cada inversa simétrica 3×3 se guardan
   en float32. Se reutilizan dentro de las seis cargas, nunca entre solves.
   Las frecuencias excluidas y modo cero permanecen anulados.

La preparación anisotrópica costó unos **13,4 ms** por solve y añadió
**167.270.400 bytes** (167,3 MB decimales) de inversas espectrales. Su pérdida
no se explica principalmente por esa preparación: necesitó **53 CG por carga**
frente a 48. El promedio no es un mejor precondicionador para esta geometría.
Esto descarta esta referencia concreta, no todas las referencias anisotrópicas.

## Precisión y aceptación

| Variante | Máximo residuo físico relativo | Diferencia relativa de C_eff frente a actual |
| --- | ---: | ---: |
| Actual | 9,2730e-6 | 0 |
| Simétrica | 9,2723e-6 | 8,8325e-8 |
| Actualización fusionada | 9,2723e-6 | 7,3487e-8 |
| Anisotrópica | 8,5434e-6 | 3,9500e-8 |

Las 18 cargas por variante cumplieron residuo ≤ 1e-5. Ninguna requirió
iteraciones correctoras del CG histórico. Las tres pasaron el umbral
comparativo de C_eff de 1e-6 frente al PCG actual. Son diferencias entre
rutas, no errores respecto a la solución continua ni una referencia G09 en
doble precisión. No se cambiaron las tolerancias de aceptación.

Se verificaron además mallas pequeñas pares/impares y no cúbicas frente al
solver CPU64 a 1e-10; error relativo de C_eff < 2e-6. Las pruebas específicas
comprueban adjuntos, equivalencia de R² con la referencia existente, inversión
anisotrópica en el subespacio compatible, positividad, norma fusionada con
acumulación float64, RHS nulo, rechazo por falta de convergencia y restauración
de los puntos de entrada experimentales al salir. La regresión conjunta
incluye el solver no lineal y composite voxels, que permanecen sin cambios:
**130 pruebas pasaron** (una advertencia preexistente de `numpy.matlib`).

## Evidencia y reproducción

- [Reporte de las doce resoluciones medidas](results/g09_three_linear_strategies_20260922/report.json):
  C_eff completos, diagnósticos por carga, preparación anisotrópica y hashes.
- [Prototipos aislados](linear_strategy_candidates.py).
- [Ejecutable del benchmark](benchmark_linear_strategies.py).
- [Pruebas](../../tests/test_linear_strategy_candidates.py).
- [Perfil previo y fundamentos matemáticos/papers](LINEAR_NEXT_OPTIMIZATIONS.md).

```bash
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_linear_strategies.py \
  --repeats 3 --output /tmp/g09-three-strategies-new

OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_linear_strategy_candidates.py
```

Los parches temporales del benchmark requieren ejecución secuencial en un
proceso dedicado; se restauran incluso al fallar. No utilizarlos como API
concurrente de producción. Antes de promover la variante ganadora se deberá
integrar su selección sin parches y repetir la comparación de rendimiento.
