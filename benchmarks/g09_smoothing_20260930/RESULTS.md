# CuFHom: menos iteraciones con suavizado tardío de CG

30 de septiembre de 2026. G09 original **240³**, Vf **24,98 %**, seis cargas
Mandel unitarias, RTX 5070 Ti 16 GB, CuPy 13.6.0, float32, residuo físico
relativo **1e-5**, atol 0. Entradas originales conservadas por SHA256.

## Resultado integrado

| Configuración | Mediana de 5 repeticiones alternadas | CG totales |
| --- | ---: | ---: |
| Optimizaciones anteriores, suavizado desactivado | 2,85595 s | 288 |
| Predeterminados con suavizado tardío | **2,77550 s** | **276** |

**2,82 % menos tiempo y 4,17 % menos iteraciones**: 46 por carga en vez de 48.
Se compara cambiando únicamente `linear_residual_smoothing`. Máximo residuo
físico **9,6143e-6**; diferencia relativa de Ceff **7,26e-10** respecto a la
referencia de esta comparación. [Datos, hashes y configuraciones](timings.json).

Los tiempos incluyen material, seis cargas, auditorías, Ceff y JSON de tiempos,
reutilizando la geometría preparada. Excluyen inicio del proceso, preparación
geométrica inicial y exportación de campos. Rangos medidos: 2,853–2,865 s sin
suavizado y 2,774–2,777 s con suavizado.

La confirmación previa del prototipo dio **3,03643 → 2,94805 s**, también cinco
repeticiones, 288 → 276 CG y un ahorro del 2,91 %. Los tiempos absolutos variaron
entre sesiones; no atribuimos al algoritmo esa variación común a ambas rutas.
El ahorro se calcula dentro de cada comparación alternada.
[Prototipo y evidencia](../../descartados/suavizado_integrado_20260930/README.md).

## Cómo resolvemos ahora

Se conserva el problema lineal y el precondicionamiento simétrico:

```
H z = b,       H = R Q* C Q R
R = raíz inversa de la referencia isotrópica
```

Q transforma tres coordenadas por frecuencia en deformaciones compatibles;
Q* es su adjunto para el producto real RFFT. C aplica el tensor constitutivo
local. Se mantienen modos medios, Nyquist y la misma discretización. H es SPD
en el espacio admisible para materiales SPD. Las reducciones escalares son
float64; los campos y FFT del benchmark son float32.

La recurrencia CG sigue calculando `alpha=(r*r)/(p*Hp)`, actualizando x/r y
usando `beta=(r_nuevo*r_nuevo)/(r*r)`. Sus fusiones anteriores permanecen.
Cuando la norma de r entra en la última década antes de la tolerancia,
inicializamos y=x y s=||r||². Para cada aproximación CG posterior:

```
w = s / (s + ||r_nuevo||²)
y = (1-w) y + w x_nuevo
s = w ||r_nuevo||²
```

En aritmética exacta, la ortogonalidad de los residuos CG hace que este peso
minimice la norma del residuo de la combinación. En float32, **s es solo una
estimación**. Si alcanza el umbral, recalculamos b-Hy. Si falla, reiniciamos
CG y el promedio desde esa aproximación. Al salir se conserva además la
auditoría independiente de las ecuaciones físicas originales y su corrección.

El promedio anterior se fusiona con la siguiente actualización x/r. Solo se
termina por separado antes de una auditoría. Se necesita un vector adicional:
unos **167 MB** en G09, asignado al comenzar el suavizado. No se usa información
de fibras, orientación, volumen ni dimensiones particulares para decidirlo.

La técnica general de suavizado y sus problemas de redondeo se estudian en
[Zhou y Walker, 1994](https://users.wpi.edu/~walker/Papers/residual_smoothing,SISC_15,1994,297-312.pdf).
La activación tardía y la fusión son decisiones de implementación de CuFHom.
La [revisión de artículos](../../docs/PAPERS_LINEAR_ACCELERATION_20260929.md)
documenta también CR, bloques y posibles precondicionadores futuros.

## Casos generales y límites

Cinco repeticiones alternadas por caso, geometría preparada, seis cargas y
las mismas auditorías físicas. Los materiales y geometrías son sintéticos y
se definen en el benchmark, sin adaptar parámetros a G09.

| Caso | Malla | CG antes → después | Tiempo antes → después |
| --- | --- | ---: | ---: |
| Laminado | 64³ | 29 → 29 | 17,811 → 17,843 ms |
| Inclusiones compactas | 64³ | 147 → 142 | 40,207 → 39,651 ms |
| Regiones con distintas orientaciones, contraste longitudinal 100 | 65³ | 290 → 275 | 85,603 → 81,961 ms |
| Inclusiones con contraste longitudinal 1,5 | 64³ | 30 → 30 | 18,054 → 18,190 ms |

[Datos completos](general.json). No hay ganancia en los dos casos fáciles;
la diferencia es de 0,03 y 0,14 ms. No se promete acelerar todas las entradas.
`linear_residual_smoothing=False` permite comparar o desactivar el promedio.
Se aplica en la ruta GPU de coordenadas simétricas, que requiere representación
indexada compatible. CPU, PCG con referencia explícita y `Cfield_array` general
mantienen sus rutas compatibles; este último usa deformaciones.

## Tiempo fuera del bucle CG

Perfil separado, cinco repeticiones por modo con fronteras GPU sincronizadas.
[Informe](pipeline.json). Las medianas de etapas no suman necesariamente la total.

| Etapa | Desde archivos | Geometría preparada |
| --- | ---: | ---: |
| Bucle CG, seis cargas | 2.566,48 ms | 2.570,79 ms |
| RHS, coordenadas, auditorías físicas y campo macroscópico | 163,52 ms | 163,25 ms |
| Preparación constitutiva y transferencia GPU | 346,41 ms | 5,04 ms |
| Lectura de arrays | 14,64 ms | <0,01 ms |
| Ensamblaje de Ceff | 34,50 ms | 34,53 ms |
| Proyección y resto | ≈7,91 ms | ≈6,85 ms |
| **Total** | **3.132,86 ms** | **2.780,45 ms** |

Solo precargar arrays, sin preparar sus grupos, da **3,115 s**. La lectura ya
es pequeña; el principal coste fuera de CG es preparar material y geometría.
La compilación explícita cuesta **344 ms**, más **14,7 ms** de lectura, una vez
por geometría. Reutilizarla ayuda en campañas; una consulta nueva paga ese coste.

Importaciones: **222 ms**; contexto CUDA: **225 ms**; primera llamada desde
archivos: **3,240 s**, adicional a lo anterior. Las cachés OS/JIT ya existían.
Generación, voxelización y exportación de campos no se midieron aquí. Los seis
campos completos representan aproximadamente 1,99 GB antes de comprimir.

## Precisión y comprobaciones

- **312 pruebas y 7 subpruebas aprobadas**. 16.605 avisos de deprecación
  existentes, principalmente NumPy; sin fallos ni omisiones. JUnit local:
  `benchmarks/results/linear_smoothing_tests_20260930.xml`.
- Referencias CPU64 pequeñas: mallas pares/impares, laminados, bloques,
  inclusiones compactas y tensor anisótropo general. Matriz SPD densa aleatoria
  contrastada con `numpy.linalg.solve`, sin usar la recurrencia como referencia.
- Pruebas de estimación falsa antes y después de activar el suavizado,
  reinicio con dirección diferida, RHS nulo, presupuesto agotado, warm start,
  buffers independientes y opción explícita para desactivar la mejora.
- Las seis deformaciones completas G09 pasan el criterio ya existente de
  comparación: diferencia L2 máxima **1,863e-5**; diferencia puntual normalizada
  máxima **1,771e-5**; Ceff **5,51e-8**. Esta comparación separada usa la base
  histórica con fusiones nuevas desactivadas y posprocesado escalar, por eso su
  referencia difiere de la tabla de tiempos. [Campos y hashes](fields.json).

Los campos finales no son idénticos a los de CG anterior; ambas aproximaciones
satisfacen la misma tolerancia física. Esta validación no sustituye una solución
CPU64 de G09 completo ni un estudio de convergencia de malla.

## Ensayos que no quedaron activos

Se archivaron CR directo, CG por bloques y correcciones con 18 modos Fourier.
CR también bajó iteraciones, pero fue más lento; los otros dos no redujeron
el trabajo del operador en G09. El suavizado continuo resultó más caro que
activarlo cerca de convergencia. La búsqueda escalar de K/G no mejoró la cota
material de la referencia actual. [Motivos, código y JSON](../../descartados/README.md).
Estos ensayos no descartan todos los precondicionadores de dos niveles: una
base GenEO adaptada al operador y subdominios sigue pendiente de implementación.

## Reproducción

```bash
source benchmarks/reference_solvers/env_gpu.sh
export OPENBLAS_NUM_THREADS=1
.venv/bin/python benchmarks/profile_g09_linear.py \
  --geometry "$GEOMETRY_DIR" --material "$MATERIAL_JSON" \
  --output benchmarks/results/my_smoothing --repeats 5 --compare-smoothing
.venv/bin/python benchmarks/profile_linear_pipeline.py \
  --geometry "$GEOMETRY_DIR" --material "$MATERIAL_JSON" \
  --output benchmarks/results/my_smoothing_pipeline --repeats 5 --time-cg
.venv/bin/python benchmarks/validate_g09_linear_fields.py \
  --geometry "$GEOMETRY_DIR" --material "$MATERIAL_JSON" \
  --output benchmarks/results/my_smoothing_fields
.venv/bin/python benchmarks/profile_linear_smoothing.py \
  --output benchmarks/results/my_smoothing_general --size 64 --repeats 5
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -I -m pytest -q
```

Cada salida debe ser nueva. Ejecutar pruebas y benchmarks GPU secuencialmente.
Los JSON versionados conservan entradas, versiones, fuente y configuraciones;
los archivos grandes no se añaden al repositorio.
