# CG por bloques: descartado para producción

30 de septiembre de 2026. Se resolvieron las seis cargas conjuntamente con
bloques de direcciones conjugadas, productos internos float64 y normalización
por descomposición espectral de matrices de Gram pequeñas. La base puede
perder rango sin descartar ecuaciones ni relajar el residuo físico.

La recurrencia usa la forma alternativa de CG por bloques descrita en
[Dubrulle, 2001, figuras 3.2–3.3](https://etna.ricam.oeaw.ac.at/vol.12.2001/pp216-233.dir/pp216-233.pdf).
La normalización espectral/rango y los kernels son decisiones de este prototipo;
no se presenta como una reproducción de su algoritmo BCGrQ con Householder.

## Validación

Productos de Gram y combinaciones GPU frente a CPU64, matrices SPD densas con
cargas dependientes y una carga nula, y casos periódicos completos contra
CPU64. En inclusiones compactas 10³, 19 pasos de bloque equivalen a 114
aplicaciones del operador más seis de auditoría, frente a 149 iteraciones
individuales en el cribado anterior. El laminado usa un bloque de rango 3.

## G09 original 240³

| Ruta | Tiempo seis cargas | Pasos / aplicaciones |
| --- | ---: | --- |
| CG actual | 3,046 s | 288 iteraciones individuales |
| Bloques | 10,343 s | 48 pasos de rango 6; 288 aplicaciones + 6 de auditoría |

Una repetición medida después del calentamiento, que dio 10,335 s: cribado
suficiente para rechazar esta implementación, sin anunciar una mejora.
Residuo físico máximo 9,19e-6; ninguna corrección física omitida. El campo `cg`
del runner contabiliza seis veces los pasos de bloque más correcciones; no
confundirlo con el número de pasos del algoritmo de bloques.

Se mantiene el mismo operador fino y se reconstruyen/verifican los seis RHS.
El coste adicional incluye productos cruzados, ortogonalización, combinaciones
de vectores y una preparación duplicada del RHS dentro del adaptador de prueba.
Esta preparación extra no explica la penalización de más de siete segundos.
No se conservaron seis iteradores independientes disfrazados de block-CG.

`cufhom_blockcg.py` queda solo como evidencia, fuera del núcleo activo.
`validate.py` reproduce las verificaciones; `benchmark.py` acepta rutas y usa
`PYTHONPATH=benchmarks:descartados/cg_bloques_20260930`. Los JSON guardan los
hashes y los recuentos de aplicaciones reales del prototipo.
