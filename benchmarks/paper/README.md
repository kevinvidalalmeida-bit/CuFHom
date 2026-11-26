# Benchmarks lineales reproducibles

Esta carpeta pública contiene las siete geometrías, protocolos y ejecutores
para una campaña nueva. No requiere manuscritos ni registros históricos. Las
mediciones se vuelven a obtener con el hardware y el solver de cada ejecución;
no se recuperan tiempos del artículo.

## Preparación y comprobación rápida

Para resolver un problema propio sin configurar una campaña, usa
`homogenize(labels, stiffness, precision="fp64")`, como explica la
[guía de uso](../../docs/USAGE.md). Las campañas de esta carpeta fijan sus propias
precisiones (`fp32`, `fp64`, `mixed`), tolerancias y dispositivos en protocolos
JSON; no heredan la detección automática de CPU/GPU de esa API.

Desde la raíz, con CuFHom instalado (`python -m pip install -e '.[test]'`):

```bash
python benchmarks/paper/run.py --help
python benchmarks/paper/run.py --dry-run
python benchmarks/paper/run.py --backend numpy --small --output results/smoke
```

`--small` usa FP64 y mallas 8³/9³. Sirve para comprobar instalación y datos;
no produce un benchmark del artículo. La campaña completa requiere CuPy/CUDA,
una GPU compatible y memoria suficiente. Los ejecutores CPU/GPU externos son
opcionales y se documentan junto con sus adaptadores.

```bash
CUFHOM_FUSED_COMPILE=sync python benchmarks/paper/run.py \
  --protocol benchmarks/paper/protocols/public.json \
  --output results/public_nueva
```

Cada condición y sesión se ejecuta en un proceso independiente, secuencialmente.
Los directorios de salida deben ser nuevos: ningún registro se sobrescribe.
`results/` está excluido de Git. Para comparar rendimiento, reservar la GPU,
fijar afinidad/hilos cuando corresponda y conservar el mismo entorno; el
ejecutor registra afinidad, variables de hilos, versiones, CPU y GPU, pero no
garantiza exclusividad del dispositivo.

## Protocolos

| JSON | Contenido |
| --- | --- |
| `public.json` | Imagen pública: HEX8 192³, GaNi 189³, FP64/mixto, η=10⁻⁶, tres procesos por condición y tres llamadas calientes. |
| `six_cells.json` | Seis celdas, HEX8 64³/128³/192³ y GaNi 63³/125³/189³; referencia tensorial FP64 a 10⁻¹⁰. |
| `fp32.json` | Imagen pública, FP32 a 10⁻⁵ y comparación con referencia nominal FP64. El redondeo material se identifica por separado. |
| `six_cells_fp32.json` | Las seis celdas y sus tres mallas, FP32 a 10⁻⁵, con referencia nominal FP64. |
| `tolerance.json` | Intentos de tolerancia, incluidos los rechazados; un proceso por punto, sin repeticiones temporales. |
| `assembly.json` | η=10⁻²…10⁻⁸, tensores directo/corregido de la misma solución, cinco llamadas calientes y referencia FP64 a 10⁻¹². |
| `contrast.json` | χ=10⁻³…10³, incluido χ=1; E₁=1 y ν₁=ν₂=0.30. |
| `reference.json` | Cruce χ–ν₀ con contraste nominal separado; μ₀ fijo dentro de cada contraste. |
| `scaling.json` | 64³…512³ en ambas formulaciones, registrando la política de residencia real. |

La configuración es JSON editable, no código de campañas con fechas. Una
llamada resuelve las seis deformaciones Mandel unitarias, desde cero, con
η relativa y tolerancia absoluta cero. HEX8 usa fuerza nodal Linf y GaNi
tensión proyectada RMS en CUDA: compartir η no iguala sus normas ni sus
errores frente al continuo. En CPU, GaNi usa su ruta de referencia, identificada
en el resultado. FP64 usa acumulación y Krylov FP64; mixto usa acumulación FP64,
Krylov FP32 y evaluaciones nativas df64; FP32 redondea además los coeficientes.
Una auditoría nativa no se presenta como operador independiente ni como
certificación de aritmética exacta.

Los errores FP32 frente a la referencia nominal incluyen la diferencia
introducida por los coeficientes redondeados, además del error de resolución.
No se interpretan por sí solos como error iterativo para el material redondeado.

En los barridos de contraste, `null` significa **material nominal público**
(E=75/400, ν=0.30/0.20), no χ=1. Los contrastes numéricos sustituyen el material
por E=(1,χ), ν=(0.30,0.30). Se mantiene exactamente la misma imagen remuestreada.
Se calcula μ₀ a partir de la proyección isótropa de la media aritmética de las
matrices de fase nominales FP64, sin ponderar por volumen. Para las dos fases
isótropas es (μ₁+μ₂)/2. Se conserva este μ₀ al variar ν₀ y entre precisiones;
λ₀=2μ₀ν₀/(1−2ν₀). El tensor de referencia numérico para medir error algebraico
es otro concepto, y se guarda junto con sus seis auditorías.

## Llamadas frías en las mallas de escalado

El ejecutor de llamadas frías conserva un proceso independiente por condición
y pasada, con cachés CuPy y CuFHom privadas y vacías al comenzar cada reloj:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 python -B -m benchmarks.paper.cold_scaling \
  --output results/cold_scaling_nueva
```

El protocolo predeterminado mide la imagen bifásica pública en
64/128/192/256/320/384/448/512, HEX8/GaNi y FP64/mixto a η=10⁻⁶:
32 configuraciones y tres pasadas, 96 llamadas frías en total. `--dry-run`
muestra el inventario sin ejecutar la GPU. `--resume` conserva los intentos
previos y sus fallos; solo vuelve a intentar condiciones sin registro aceptado,
usando cachés nuevas. Exige las mismas fuentes y el mismo protocolo.

Este reloj excluye construcción del descriptor host, selección de referencia,
muestreo, importaciones y contexto CUDA, para corresponder a la frontera de
las figuras históricas de escalado. Incluye preparación dentro de `homogenize`,
compilación síncrona, seis cargas desde cero, auditorías, ensamblaje y retorno
sincronizado. La caché del controlador CUDA se conserva. No se ejecuta un
calentamiento del solver y el estado vacío de caché se registra antes del reloj.

Los JSON incluyen tensores, seis residuos, conteos, residencia, entorno y
memoria muestreada. Se guardan las fuentes efectivas, las huellas, los logs y
las cachés de cada intento. El programa no interpreta el sobrecoste frío.

Para medir primeras llamadas y llamadas calientes, usa el ejecutor general
`run.py` con los protocolos públicos. Su frontera temporal se describe a
continuación; los registros de llamadas frías y del ejecutor general conservan
sus propias definiciones y no se combinan automáticamente.

## Qué mide cada registro del ejecutor general

- Primera llamada y llamadas calientes completas separadas; las calientes
  vuelven a resolver las seis cargas desde cero. No reutilizan la solución.
- El reloj incluye la construcción del descriptor material, las seis
  resoluciones, las auditorías frescas nativas y el ensamblaje corregido,
  con sincronización CUDA. Excluye lectura/muestreo, importaciones, contexto,
  selección de referencia y escritura. La primera llamada puede incluir
  compilación/planes de primera utilización; no se identifica automáticamente
  con las fronteras históricas del artículo.
- Seis conteos, seis residuos absolutos/relativos, umbrales y aceptación por
  llamada; las excepciones y puntos no aceptados se conservan. El caso
  homogéneo puede tener norma inicial nula y residuo relativo `null`; sus
  actualizaciones son cero y se excluye de normalizaciones por el mínimo.
- Matrices directa y corregida cuando la ruta las devuelve. Su diferencia
  procede de las mismas soluciones. **El ensamblaje directo no se cronometra
  por separado**: su error comparte el tiempo de la llamada corregida.
- Memoria libre global muestreada cada 20 ms respecto de la base previa a cada
  llamada; valores activos/reservados finales del pool CuPy y residencia real.
  No es un pico total por proceso y otras asignaciones pueden afectarla.
  El muestreo puede modificar el tiempo observado.
- Huellas de índices y coeficientes nominales/empleados, fuentes Python,
  revisión Git y entorno. Metadatos nativos no finitos se guardan como `null`
  con una lista de rutas; las normas físicas aceptadas deben ser finitas.
- Capacidad y disponibilidad de RAM/swap del host, RSS actual y máximo RSS de
  la vida del proceso, antes y después de cada llamada. `VmHWM` también incluye
  preparación y llamadas anteriores; no se presenta como un pico aislado de la
  llamada ni como almacenamiento total del host. Las medidas procfs ausentes
  quedan explícitamente no disponibles.

Las referencias ajustadas se calculan **después** de las llamadas medidas,
para no calentar la primera llamada. No se mezclan tiempos CPU, GPU ni de
CUPTI, ni se inferirán errores de discretización a partir de errores algebraicos.
Las distribuciones entre sesiones son descriptivas; tres sesiones no forman
un intervalo de confianza por sí solas.

## Ablaciones de implementación

Los ejecutores y variantes están agrupados en [ablations/](ablations/README.md).

```bash
python -m benchmarks.paper.ablations --pair hex_stencil --dry-run
CUFHOM_FUSED_COMPILE=sync python -m benchmarks.paper.ablations \
  --pair hex_stencil --precision fp64
```

Las salidas se guardan en carpetas nuevas bajo `results/ablations/`; los
controles se ejecutan en procesos aislados sin modificar el solver. La guía
detalla los pares, tolerancias, cronómetros y límites de interpretación.

## Tablas y diagnóstico de la nueva campaña

```bash
python benchmarks/paper/summarize.py results/public_nueva
```

Se escribe `tables/points.csv`, con estados fallidos conservados, seis residuos
y seis conteos por fila, tiempos de primera llamada y distribución caliente.
Para el barrido χ–ν₀ se añaden `reference_curves.csv` (valores absolutos y
normalizados), `reference_diagnostics.csv` (mínimos muestreados y penalizaciones
de ν₀=0.30/0.48) y `reference_comparisons.csv` (diferencias FP64/mixto y entre
contrastes recíprocos sobre los mismos ν₀). χ=1 y curvas de trabajo nulo no se
normalizan. Estos diagnósticos comparan los puntos medidos y no demuestran un
óptimo continuo ni independencia respecto del contraste. No se sobrescriben
las figuras ni las cifras del artículo; la campaña nueva se revisa antes de
sustituir evidencia publicada.

## Comparadores independientes y perfiles opcionales

`external.py` usa los paquetes originales, nunca `cufhom_core` como sustituto:

```bash
python -m benchmarks.paper.external ffthompy --case public --grid 189 \
  --source external/FFTHomPy --precision fp64 --eta 1e-6 \
  --output results/ffthompy_session1.json
python -m benchmarks.paper.external fans --case public --grid 192 \
  --source external/UNOCG --eta 1e-6 \
  --output results/fans_session1.json
```

La preparación de dependencias y sus revisiones fijas está en
[docs/REPRODUCIBILITY.md](../../docs/REPRODUCIBILITY.md).
FANS aquí es la implementación PyTorch de `FansPreconditioner` distribuida en
UNOCG, la usada en la comparación GPU del artículo; no es el programa FANS C++.
No requiere pesos UNO ni sus grandes archivos de entrenamiento. Las mallas de
este comparador FANS deben ser pares; las comparaciones FFTHomPy/GaNi usan
mallas impares. Cada comando produce una primera llamada y tres calientes,
todas desde cero; para tres sesiones se ejecutan tres procesos distintos con
nombres de salida nuevos. FFTHomPy admite también almacenamiento FP32.

La frontera nueva de los comparadores incluye preparación material, seis
resoluciones, ensamblaje y auditoría física fresca. Importaciones, contexto,
lectura y generación de entradas quedan fuera. La auditoría de FFTHomPy usa
IEEE FP64 y el material efectivamente almacenado; las normas siguen siendo
propias de cada formulación. El comparador conserva la revisión y las huellas
de los fuentes externos. Sus tiempos no sustituyen automáticamente los tiempos
históricos, cuyas fronteras variaban por campaña. No se ejecuta FANS FP32 como
comparación de precisión equiparada.

Para recoger eventos de una llamada caliente, con PyTorch/CUPTI disponibles:

```bash
python -m benchmarks.paper.profile --solver cufhom --discretization hex8 \
  --precision fp64 --grid 192 --output results/profile_hex8.json
python -m benchmarks.paper.profile --solver fans --grid 192 \
  --source external/UNOCG --output results/profile_fans.json
```

El perfil guarda una traza Chrome y un resumen por nombre de evento, además de
las soluciones y auditorías. Su duración está instrumentada: no es un tiempo de
benchmark sin perfilador. Las sumas de eventos pueden solaparse; la unión de
intervalos registrados tampoco certifica utilización física completa.
