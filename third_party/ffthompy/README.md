# FFTHomPy

`core` contains a modified copy of FFTHomPy, originally authored by
Jaroslav Vondřejc. Its MIT notice is preserved in
[`core/LICENSE`](../../core/LICENSE) and in the installed
package. Individual source notices are retained.

The internal namespace is `cufhom_core`. Local modifications include CuPy FFTs,
constitutive fields, projections, CG and GPU kernels. CuFHom's own code has no
project license assigned; the MIT notice applies to the incorporated code.

## Comparación con el original — 2026-10-03

Se descargó el [repositorio oficial de FFTHomPy](https://github.com/vondrejc/FFTHomPy)
y se comparó la revisión
[`2c23c80dd2cab46f1090103e613b4f886b3daac7`](https://github.com/vondrejc/FFTHomPy/tree/2c23c80dd2cab46f1090103e613b4f886b3daac7)
con CuFHom `4b54f49`. La copia original está en
`external/solver_references/FFTHomPy/`, excluida de Git y de la distribución.
Esa revisión es la referencia de esta comparación; no identifica necesariamente
la revisión histórica desde la que se incorporó el núcleo.

La comparación del árbol sintáctico de Python normaliza únicamente el namespace
`ffthompy` a `cufhom_core` e ignora formato, comentarios y docstrings.
Encontró **190 funciones o métodos con la misma implementación** en archivos
correspondientes. Algunos ejemplos:

| Archivo de `core/` | Funciones o métodos iguales |
| --- | ---: |
| `matvecs/objects.py` | 95 |
| `tensors/objects.py` | 34 |
| `general/base.py` | 18 |
| `tensors/operators.py` | 16 |
| `mechanics/matcoef.py` | 10 |
| `general/solver_pp.py` | 6 |

Los módulos `mechanics/matcoef.py` y `general/solver_pp.py` conservan toda su
estructura de código bajo esa normalización. También existen modificaciones
extensas y funciones añadidas en CuFHom. Estos recuentos documentan código
retenido; no representan un porcentaje de autoría o de novedad científica.

En pruebas CPU de una geometría mixta de 4³ vóxeles, realizadas con el wheel de
CuFHom instalado y sin FFTHomPy instalado o importado, ambos solvers funcionaron.
La ruta lineal ejecutó 19 funciones conservadas del original. La ruta no lineal
también utilizó el núcleo incorporado, a través de sus funciones modificadas.
Por tanto, la incorporación sigue activa y no es únicamente código archivado.

### Descripción del proyecto

Una descripción respaldada por esta comprobación es:

> CuFHom es un proyecto autónomo de generación de geometrías y homogenización
> FFT que incorpora un núcleo modificado de FFTHomPy bajo licencia MIT y añade
> funciones y optimizaciones para CPU y GPU.

La autonomía de instalación no implica que toda la implementación tenga una
autoría independiente. Los cambios de nombre, correcciones y optimizaciones
conservan la procedencia del código incorporado. La
[licencia original](https://github.com/vondrejc/FFTHomPy/blob/2c23c80dd2cab46f1090103e613b4f886b3daac7/LICENSE)
permite modificar y distribuir el código con sus avisos de copyright y permiso.
Ambas copias de la licencia conservadas en CuFHom coinciden byte a byte con ella.
