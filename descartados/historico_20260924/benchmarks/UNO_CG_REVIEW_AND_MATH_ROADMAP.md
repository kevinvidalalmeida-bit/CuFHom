# UNO-CG y siguientes desarrollos matemáticos

**Actualización posterior:** ya se implementaron y midieron tres variantes de
polarización GPU en G09; la combinación se integró en el solver lineal.
Véase [campaña de implementación](POLARIZATION_GPU_AND_2026_REVIEW.md).
Este documento conserva el diagnóstico previo y sus propuestas iniciales.

Fecha: 2026-09-22. Alcance: revisión y diagnóstico; no cambia el default,
la geometría, las tolerancias ni el solver no lineal.

## Qué aporta el artículo solicitado

[Herb y Fritzen, 2026, DOI 10.1002/nme.70277](https://onlinelibrary.wiley.com/doi/10.1002/nme.70277)
aprenden un precondicionador lineal simétrico positivo definido (SPD), no la
solución final. Los modos no aprendidos conservan una acción positiva. Su
comparación incluye elasticidad lineal 3D, pero no valida nuestra plasticidad
ni nuestras fibras transversalmente isótropas. Ser competitivo con FANS no
demuestra superar nuestro precondicionamiento actual. La aplicación con FFT
sigue siendo O(N log N); el coste de entrenamiento debe amortizarse.

El [código oficial](https://github.com/DataAnalyticsEngineering/UNOCG) usa
PyTorch y tiene licencia MIT. No se ha instalado ni incorporado al repositorio.

## Evidencia local: atacar las iteraciones

El mejor prototipo combinado medido tarda **3,4188 s de mediana**, seis cargas,
G09 240³, CuPy float32, GPU caliente. Sigue en benchmark, no en producción.
Conserva **288 CG** y residuo físico máximo 9,2716e-6.
[Condiciones, repeticiones y verificaciones](COMBINED_LINEAR_STRATEGY_RESULTS.md).

Una captura adicional Nsight del combinado duró 3,4071 s; los tiempos de
kernels fueron:

| Trabajo | Tiempo GPU |
| --- | ---: |
| FFT, incluidas preparación y auditorías | 1,7741 s |
| Conversión de coordenadas | 0,3938 s |
| Actualización x/r y norma fusionadas | 0,3624 s |
| Aplicación constitutiva indexada | 0,3348 s |
| Actualización de dirección CG | 0,1758 s |
| Productos internos warp_dot | 0,1744 s |

Las FFT representan aproximadamente el **52 % del tiempo de pared**. No sumar
este perfil con otros tiempos inclusivos. Bajar CG evita también operaciones
constitutivas y recorridos de memoria, no sólo FFT.
[Resumen de kernels](results/g09_combined_linear_strategies_20260922/cuda_summary_cuda_gpu_kern_sum.json).

## Propuesta propia A: corrección espectral aprendida y fusionada

Adaptación propuesta para este código, **todavía no implementada ni medida**.
El sistema combinado actual es H = R Q* C Q R, con R = M^(-1/2).
Ya guarda los vectores de Krylov en coordenadas de Fourier de tres componentes.
Por tanto, una corrección por bloques 3×3 puede aplicarse allí, sin añadir un
par de FFT exclusivamente para el precondicionador.

1. Parametrizar pocos bloques por bandas y direcciones: B(k) = L(k)L(k)* + δI,
   δ > 0. Inicializar B = I y mantener I fuera de las bandas aprendidas.
   Limitar también el autovalor máximo para evitar mal escalado en float32.
2. Ajustar pares residuo/error del **operador transformado real**, H e = r;
   no entrenar únicamente contra seis números efectivos. Separar casos de
   entrenamiento y evaluación. Optimizar error de ajuste es sólo un indicador:
   el criterio final son CG y tiempo de solves completos.
3. Mantener B fijo durante cada CG, respetar la simetría conjugada RFFT y
   excluir exactamente los mismos modos medio/Nyquist que el operador actual.
4. Si S = B^(1/2), resolver S H S y = S b, recuperar z = S y.
   Fusionar Q R S y su adjunto S R Q*. **No conmutar R y S** ni aplicar además
   el antiguo precondicionador. Usar el producto interno de nuestras
   coordenadas isométricas, no copiar sin adaptación uno de otra discretización.
5. Conservar auditoría física original, reducciones float64 y fallback.
   Validar positividad, adjunción y solución CPU64 en mallas pares/impares.

La ganancia es incierta: bloques diagonales en frecuencia no representan todo
el acoplamiento entre frecuencias que produce la heterogeneidad. El promedio
anisótropo ya perdió (318 CG y 4,231 s); entrenar bloques más generales no
garantiza corregirlo. Empezar con un ajuste pequeño, no una red grande.

Condición aproximada de rentabilidad, ignorando costes fijos: si el coste por
iteración sube 10 %, deben bajar los CG totales de 288 a menos de 262. Contar
además preparación, entrenamiento, transferencias y almacenamiento. Para
muchas consultas, amortización sólo si

    n_consultas > T_entrenamiento / (T_base - T_nuevo)

con denominador positivo y tiempos completos comparables. No se promete un
tiempo objetivo antes de medirlo.

## Propuesta propia B: eliminar incógnitas de matriz, no material

Diagnóstico algebraico lineal implementado en
[check_polarization_identity.py](check_polarization_identity.py).
Sea S la inserción de valores de fibra en la malla completa, S* su restricción,
y C = Cm + S ΔC S*. Aquí S no es la raíz de la sección anterior.
Para contraste ΔC positivo definido, D = (ΔC)^(1/2). Con el Green discreto
de matriz Γm y deformación media E:

    (I + D S* Γm S D) p = D S* E
    τ = D p
    ε = E - Γm S τ

El operador reducido es SPD porque Γm es semidefinido positivo. Es una
reformulación exacta del mismo problema discreto: la matriz sigue influyendo
por Γm y su campo se reconstruye. No es coarsening ni eliminación física de
la resina. La idea de polarización tiene antecedentes; no se reclama novedad.

**Comprobación CPU64 realizada:** seis cargas, orientaciones aleatorias,
materiales G09 y convenciones espectrales existentes:

| Malla | Diferencia relativa de C_eff | Residuo físico máximo |
| --- | ---: | ---: |
| 5×5×5 | 3,86e-13 | 9,23e-13 |
| 6×4×8 | 3,12e-13 | 9,11e-13 |

[Resultados reproducibles](results/g09_polarization_feasibility_20260922.json).
No es una resolución reducida G09 ni un benchmark de velocidad. Los CG de
estas pruebas usan tolerancia 1e-12 y no se comparan con los 48 por carga GPU.

G09 contiene 3.453.312 vóxeles de fibra, el 24,98 % de los 13.824.000 totales.
Un vector real float32 de polarización sobre fibras ocupa **82,88 MB**, frente
a **167,27 MB** del vector de coordenadas complejo64 actual. Es una reducción
por vector Krylov, **no de toda la memoria GPU**. Se conservan buffers FFT
densos y aparecen costes de restricción/inserción. La malla FFT no disminuye
y la complejidad no se convierte automáticamente en O(N).

El contraste material local tiene seis autovalores positivos. Aproximarlo por
rango uno descarta **10,30 % de su norma Frobenius**: no es una simplificación
exacta ni un error demostrado de C_eff. Considerar aproximaciones sólo dentro
del precondicionador, manteniendo el operador físico completo.

## Orden recomendado y criterios de aceptación

1. Pilotar A sobre el combinado, inicialmente CPU64 pequeño; medir después
   G09 completo con costes de preparación y evaluación fuera del entrenamiento.
2. Prototipar B en GPU por separado; comparar memoria máxima, CG y tiempo.
   No mezclar ambas antes de identificar sus costes individuales.
3. Sólo si hay ganancia, estudiar la combinación y el reciclaje entre cargas.

Cada candidato debe conservar tolerancia física 1e-5 GPU, diferencia relativa
de C_eff ≤ 1e-6 frente a la ruta actual y comprobación independiente CPU64
pequeña. Medir repeticiones calientes en orden rotativo y reportar por separado
arranque frío, preparación y entrenamiento. Rechazar mejoras sólo aparentes
por relajación de tolerancia o por omitir auditorías.

La eliminación mediante una matriz homogénea no se traslada directamente a
una resina plastificada: allí la tangente depende del estado local. Un B
aprendido podría actualizarse entre sistemas Newton, pero necesitaría nuevas
pruebas y permanecer fijo dentro de cada CG. Nada de esta revisión modifica
la ruta no lineal ni su trayectoria de 40 incrementos.

Reproducción del diagnóstico algebraico, usando una salida nueva:

```bash
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/check_polarization_identity.py \
  --output /tmp/polarization-check-new.json
```
