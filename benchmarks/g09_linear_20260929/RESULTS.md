# CuFHom: optimización lineal de G09

Campaña del 29–30 de septiembre de 2026. GPU RTX 5070 Ti de 16 GB,
Python 3.12, CuPy 13.6.0, CUDA runtime 12.9.

**Informe histórico de las fusiones y el posprocesado.** La revisión posterior
añade suavizado tardío, reduce 288 a 276 CG y documenta sus propios tiempos:
[resultado vigente](../g09_smoothing_20260930/RESULTS.md).

## Resultado del solver completo con geometría preparada

G09 original **240³**, seis cargas Mandel unitarias, float32 y aceptación
por residuo físico relativo **1e-5**. Este G09 tiene Vf **24,98 %**: es
un caso distinto del benchmark pequeño Vf 15 %, AR 10.

| Configuración | Mediana, 5 repeticiones | CG totales |
| --- | ---: | ---: |
| Base optimizada previa: fusiones nuevas desactivadas, ensamblaje escalar | 3,16535 s | 288 |
| Tras las fusiones de coordenadas y el ensamblaje GPU | **3,03720 s** | **288** |

Reducción de tiempo **4,05 %**; 48 iteraciones por carga en ambos casos.
La mejora reduce trabajo y tráfico de memoria por iteración y en posprocesado.
Esta etapa no redujo iteraciones; el informe posterior sí documenta una reducción.

Los tiempos incluyen preparación del material a partir de la geometría ya
compilada, resolución, auditorías, Ceff y escritura del JSON de tiempos. Excluyen
importación, contexto CUDA, carga/compilación inicial de geometría y exportación
de campos completos. Hay calentamiento por variante y orden alternado.
[Datos, configuraciones, Ceff y hashes](complete.json).

## Dónde se va el tiempo además de CG

Medianas de cinco ejecuciones por modo con fronteras GPU sincronizadas.
El perfil de llamadas se hizo aparte y no interviene en estas medianas.
[Datos por etapa](pipeline_final.json).

| Etapa | Desde archivos | Geometría preparada |
| --- | ---: | ---: |
| Bucle CG, seis cargas | 2.814,53 ms | 2.813,46 ms |
| RHS, coordenadas, auditorías y adición macroscópica | 173,69 ms | 175,07 ms |
| Preparación constitutiva y transferencia GPU | 332,79 ms | 4,74 ms |
| Lectura y conversión de arrays | 13,65 ms | <0,01 ms |
| Ensamblaje de Ceff | 37,31 ms | 37,70 ms |
| Proyección | 1,14 ms | 1,13 ms |
| Operador material y salida final | 0,26 ms | 0,24 ms |
| Resto, incluido JSON de tiempos | 6,95 ms | 5,46 ms |
| **Total mediano** | **3.382,92 ms** | **3.038,04 ms** |

Las medianas de etapas no tienen por qué sumar exactamente la mediana total.
Cargar arrays en RAM sin compilar sus grupos dio **3,365 s**: casi todo el
sobrecoste restante está en preparar grupos, índices y rotaciones del material.
El perfil CPU inicial sitúa aproximadamente 191 ms en agrupar orientaciones,
37 ms en `argwhere` y 48 ms en construir rotaciones; estos valores vienen de
una ejecución instrumentada, incluyen llamadas anidadas y no se suman a la
mediana. Son costes de la entrada fase/orientación usada por G09; un Cfield
arbitrario tiene un coste de preparación diferente.

Costes separados, medidos una vez en un proceso nuevo con cachés existentes:

- Importación de módulos: **209 ms**; inicialización del contexto GPU: **244 ms**.
- Primera llamada desde archivos: **3,490 s**, adicional a lo anterior.
  No se vaciaron cachés de sistema ni JIT: no es una prueba de disco frío.
- Precarga de arrays: **13,6 ms**; compilación explícita de geometría: **330 ms**.
  Se pagan una vez por geometría y se amortizan si se reutiliza. El camino
  preparado no ahorra ese coste en una única consulta nueva.
- Hashes para verificar las entradas: **71,6 ms**, instrumentación del benchmark.

Antes del último cambio, la lectura tardaba unos **29 ms** y el ensamblaje
**68 ms**. Tras evitar copias del mismo dtype y reutilizar la reducción CUDA,
son unos **14 ms** y **38 ms**, respectivamente. Los tiempos completos desde
archivos pasaron de **3,429 a 3,383 s** en campañas sucesivas, sin cambiar CG.
[Perfil anterior](pipeline_before.json), [perfil final](pipeline_final.json).

Para reducir tiempo general fuera de CG, las prioridades son mantener vivo el
proceso GPU en campañas, reutilizar preparaciones explícitas cuando los datos
son los mismos y evitar exportar campos que la aplicación no utiliza. La
lectura de G09 ya es pequeña. La generación/voxelización de una microestructura
nueva y la exportación a NPZ/VTK no se ejecutan en este benchmark; no se les
atribuye tiempo cero. Los seis campos float32 completos suman aproximadamente
1,99 GB antes de comprimir y transferir/guardar eso requiere otra medición.

## Cambios integrados

- Los parámetros mínimos GPU activan las optimizaciones compatibles. Las
  opciones explícitas y la precisión solicitada tienen prioridad. El diccionario
  del usuario conserva su contrato de salida y no recibe defaults nuevos.
- `p.Ap` se acumula en float64 al producir el adjunto; la actualización de la
  dirección CG se fusiona con el mapa de coordenadas. Se conserva la comprobación
  independiente del residuo físico y la corrección si no converge.
- El ensamblaje GPU reutiliza la reducción por warps del solver, calcula todas
  las entradas con sumas float64 y transfiere la matriz pequeña una vez. Evita
  36 campos temporales de productos y sus transferencias escalares individuales.
- La carga desde archivos evita copias cuando el dtype ya es el requerido.
- La métrica de referencia experimental en seis componentes salió del núcleo.
  Código, parches, resultados y motivos están en [descartados](../../descartados/README.md).

Las fusiones no dependen de reconocer fibras ni fijan dimensiones de G09.
La ruta rápida en coordenadas requiere la representación indexada compatible;
los campos `Cfield_array` generales conservan la formulación en deformaciones.
El ensamblaje nuevo también funciona en esa ruta. No se afirma que todos los
materiales tengan la misma velocidad ni que todas las rutas usen coordenadas.

## Validación

Suite final: **306 pruebas y 7 subpruebas aprobadas**, sin fallos ni skips.
Aparecen 14.905 avisos, principalmente deprecaciones existentes de `numpy.fix`
y `numpy.matlib`; no son errores de convergencia. Registro JUnit local:
`benchmarks/results/linear_final_tests_20260930.xml` (ignorado por Git).


- Cada medición conserva las seis auditorías físicas, con residuo máximo
  **9,27175e-6**. Los archivos originales mantienen sus SHA256.
- Diferencia relativa de Ceff respecto a la base: **5,52e-8** (norma Frobenius).
- Las seis deformaciones completas se comparan en una ejecución separada:
  diferencia L2 relativa máxima **1,593e-6**; diferencia puntual máxima
  normalizada por el máximo del campo **3,869e-7**.
  [Informe final de campos](fields_final.json).
- Casos pequeños contra CPU64: mallas pares/impares, bloque, laminado,
  inclusiones compactas y campo anisótropo general. Productos de posprocesado
  contra contracción independiente CPU64, incluyendo campos no contiguos y
  un operador no simétrico para comprobar todas las entradas.
- Se comprueban reinicio fiable de CG, falsa convergencia, guess reutilizado,
  conservación de entradas y propiedad independiente de los buffers devueltos.

Esta evidencia comprueba equivalencia con el problema discreto y la
implementación anterior. No sustituye un estudio de convergencia de malla ni
una solución CPU64 de G09 completo.

## Medición del flujo y reproducción

`profile_linear_pipeline.py` distingue ejecución desde archivos, arrays en
RAM y grupos geométricos compilados. Sus fronteras de etapa sincronizan CUDA.
`--time-cg` mide además el bucle CG por separado; `--profile-host` produce un
perfil de llamadas, excluido de las medianas. Los tiempos acumulados de ese
perfil incluyen esperas GPU: no son tiempos exclusivos de kernels y no deben
sumarse entre padres e hijos.

```bash
source benchmarks/reference_solvers/env_gpu.sh
export OPENBLAS_NUM_THREADS=1
.venv/bin/python benchmarks/profile_g09_linear.py \
  --geometry "$GEOMETRY_DIR" --material "$MATERIAL_JSON" \
  --output benchmarks/results/my_linear_run --repeats 5 --defaults
.venv/bin/python benchmarks/profile_linear_pipeline.py \
  --geometry "$GEOMETRY_DIR" --material "$MATERIAL_JSON" \
  --output benchmarks/results/my_pipeline_run --repeats 5 --time-cg --profile-host
.venv/bin/python benchmarks/validate_g09_linear_fields.py \
  --geometry "$GEOMETRY_DIR" --material "$MATERIAL_JSON" \
  --output benchmarks/results/my_fields_run
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -I -m pytest -q
```

Definir las dos rutas de entrada; cada salida debe ser una carpeta nueva.
`phase.npy` y `ori.npy` se leen sin modificar. El benchmark no genera otra
microestructura ni cambia material, tolerancia o cargas. Los JSON de esta
carpeta conservan rutas locales para identificar los datos originales; los
scripts usan argumentos explícitos y se pueden ejecutar con otros datos.

## Artículos y siguientes oportunidades

La [revisión de artículos](../../docs/PAPERS_LINEAR_ACCELERATION_20260929.md)
prioriza estudiar una corrección global basada en el operador y particiones
periódicas de la malla. Debe validarse en varios tipos de microestructura.
La construcción de bases, su memoria y la aplicación del precondicionador
cuentan dentro del tiempo total. No hay una implementación nueva de dos
niveles activada por defecto.

Los JSON `before`, `defaults`, `kernel_trials`, `direction_trials`, `integrated`
y `final` son etapas anteriores de la campaña. `final.json` documenta solo las
fusiones de coordenadas (3,173 → 3,069 s); **`complete.json` incluye también el
posprocesado**. Los pilotos temporales están archivados, fuera de producción.
