# Uso de CuFHom

## Homogenización con las opciones habituales

`homogenize` recibe una imagen 3D de etiquetas enteras y una tabla de matrices
elásticas por fase. Solo necesita esas entradas; la precisión se puede elegir
con un nombre. Este ejemplo funciona con una instalación CPU:

```python
import numpy as np
from cufhom.fft import homogenize

labels = np.indices((4, 4, 4)).sum(axis=0).astype(np.int32) % 2
C = 2 * np.eye(6)
C[:3, :3] += 1
result = homogenize(labels, np.stack([C, 4 * C]), precision="fp64")
print(result["C"])
print(result["backend"], result["precision"])
```

Cada etiqueta selecciona una matriz de la tabla. Se aceptan matrices completas
`(n_fases, 6, 6)` o filas del triángulo superior `(n_fases, 21)`. Deben ser
simétricas y positivas definidas. Las etiquetas deben estar dentro de la tabla.
El orden Mandel es 11, 22, 33, 23, 13, 12; las componentes cortantes llevan
normalización Mandel. Todas las fases usan las mismas unidades de tensión.

## Qué seleccionar y qué dejar predeterminado

| Opción | Selecciones | Valor al omitirla |
| --- | --- | --- |
| `precision` | `"fp32"`, `"fp64"`, `"mixed"` | FP64 en CPU, mixto en CUDA para entradas sin preparar |
| `discretization` | `"gani"`, `"hex8"` | `"gani"` |
| `backend` | `"auto"`, `"numpy"`, `"cupy"` | `"auto"`: CUDA si hay dispositivo disponible, CPU en otro caso |
| `lengths` | Tres longitudes físicas de celda X, Y, Z | `(1.0, 1.0, 1.0)` |
| `rtol` | Tolerancia relativa del residuo nativo | `1e-5` con `precision="fp32"`; `1e-6` en los demás casos |

GaNi es Fourier–Galerkin con integración numérica; HEX8 usa elementos
trilineales completamente integrados. Para comparar formulaciones basta cambiar
`discretization="hex8"`. `lengths` describe la celda física, no el número de
vóxeles. La referencia elástica y la planificación de memoria se eligen
internamente; las tolerancias son ajustables si el problema lo requiere.

### Precisiones

| Modo | Coeficientes y solución acumulada | Vectores Krylov | Dispositivo |
| --- | --- | --- | --- |
| `"fp32"` | FP32 | FP32 | CPU o CUDA |
| `"fp64"` | FP64 | FP64 | CPU o CUDA |
| `"mixed"` | FP64 | FP32 | CUDA |

El modo mixto usa evaluaciones frescas df64 por defecto en sus rutas CUDA;
FP64 mantiene aritmética IEEE FP64. FP32 redondea los coeficientes: cambiar
posteriormente a FP64 no recupera los valores anteriores al redondeo. El tensor
retornado se presenta en un array FP64 también en FP32, pero ese formato no
aumenta la precisión de la solución calculada.

Para forzar CPU usa `backend="numpy"`. Para exigir CUDA usa `backend="cupy"`;
requiere CuPy y un dispositivo compatible. Pedir `"mixed"` sin CUDA produce un
error explícito. La selección automática se hace al empezar: una falta de
memoria o un fallo de convergencia no cambia la precisión ni ejecuta el problema
en otro dispositivo.

## Interpretar la salida

Cada llamada resuelve seis deformaciones Mandel unitarias. El resultado incluye:

- `C`: matriz de rigidez efectiva 6 × 6.
- `backend`, `precision`, `dtype`, `working`: configuración efectivamente usada.
- `iterations`: seis conteos de actualizaciones.
- `audits`: seis residuos frescos, umbrales y estados de aceptación.
- `precision_contract`: aritmética de materiales, iteraciones y evaluaciones.

HEX8 usa fuerza nodal Linf y GaNi tensión proyectada RMS en CUDA. Una misma
`rtol` no iguala las normas ni demuestra el mismo error frente al continuo.
El error algebraico frente a una referencia del mismo problema discreto tampoco
mide el error de discretización. El solver rechaza cargas no convergidas.

## Ejemplo analítico de Hashin

Después de instalar las dependencias GPU, añade Matplotlib y ejecuta:

```bash
python -m pip install -e '.[examples]'
python examples/hashin_convergence.py
```

El comando usa **CUDA y precisión mixta** por defecto, resuelve GaNi y HEX8
y genera una sola figura con tres paneles: corte de la geometría voxelizada,
error del módulo volumétrico y tiempo de ejecución. Conserva las matrices,
auditorías, iteraciones, versiones y huellas de entrada en `results.json`,
y la curva en `convergence.csv`. Cada ejecución crea un directorio nuevo en
`results/hashin/`; `--output` permite elegir uno que todavía no exista.
Para una comprobación pequeña en CPU:

```bash
python examples/hashin_convergence.py --backend numpy --precision fp64 \
  --grids 8 14 --repeats 1
```

### Selección de mallas

Deja `--grid-policy auto` para el uso habitual. `--grids` indica tamaños
solicitados; la política escoge el menor tamaño permitido que sea igual o
mayor. La consola y el JSON muestran los tamaños solicitados y los utilizados.
Si varios tamaños solicitados coinciden tras seleccionar, se resuelven una vez.

| Política | Selección |
| --- | --- |
| `auto` | `odd-13` para GaNi y `even-13` para HEX8; predeterminada |
| `even` | Tamaños pares, sin restricción de factores primos |
| `odd` | Tamaños impares, sin restricción de factores primos |
| `even-13` | Pares con factores primos no mayores que 13 |
| `odd-13` | Impares con factores primos no mayores que 13 |

Por ejemplo:

```bash
python examples/hashin_convergence.py --grid-policy odd-13 --grids 16 32 64
```

Las políticas `-13` emplean los factores **2, 3, 5, 7, 11 y 13**. No exigen
múltiplos de 13: 63 = 3² × 7 cumple la condición. La regla vive en el paquete
y la comparten los chequeos de las FFT CUDA generadas. Los solvers comprueban
además la memoria compartida: pueden utilizar FFT de biblioteca cuando la
ruta generada no admite la malla. Esta selección no garantiza un tiempo óptimo.

Los tamaños solicitados predeterminados son 16, 24, 32, 48, 64, 96, 128 y 192.
Con `auto`, GaNi selecciona 21, 25, 33, 49, 65, 99, 135 y 195; HEX8 conserva
los tamaños solicitados. Las mallas GaNi impares permiten comparaciones con
el adaptador externo de FFTHomPy; este ejemplo no ejecuta ese comparador.
Las FFT generadas de HEX8 requieren el último eje par: elegir `odd` u `odd-13`
para ambos métodos permite estudiar esa paridad usando FFT de biblioteca en
HEX8. Los métodos mantienen la misma geometría continua, con discretizaciones
que pueden diferir; los tiempos no representan una comparación a igual error.

La función general está disponible para preparar cualquier geometría:

```python
from cufhom.fft import select_grid

n = select_grid(128)  # 135: auto para GaNi
n_hex8 = select_grid(128, discretization="hex8")  # 128
n_odd = select_grid(17, "odd")  # 17; odd-13 seleccionaría 21
```

Se selecciona la resolución **antes de voxelizar**. `homogenize` conserva
la imagen recibida y no cambia la resolución de una geometría ya discretizada.
La medición y la gráfica se reutilizan desde el ejecutor existente para otros
casos públicos:

```python
from benchmarks.paper.run import run_convergence, plot_convergence

study, section, output = run_convergence("laminate", grids=(16, 32, 64))
plot_convergence(study, section, output / "convergence.png")
```

Sin `reference_bulk`, la gráfica muestra el cambio de módulo entre mallas.
Una referencia analítica se añade explícitamente, como hace el ejemplo Hashin;
no se asigna una solución analítica a otros materiales.

### Referencia y criterio de convergencia

Se reutilizan los materiales y radios públicos de `benchmarks/paper/inputs/`.
Para una esfera de núcleo y recubrimiento concéntricos, la referencia
hidrostática de [Hashin (1962)](https://doi.org/10.1115/1.3636446) es

$$
K_H=K_s+\frac{f(K_c-K_s)(K_s+4G_s/3)}
 {K_s+4G_s/3+(1-f)(K_c-K_s)}.
$$

Aquí `f = 0.14` es la fracción del núcleo dentro de la esfera recubierta;
`c` indica núcleo y `s`, recubrimiento. El módulo volumétrico de la matriz
exterior es `K_H = 1.237735849056604`, para que la esfera sea neutra bajo carga
hidrostática. Se compara exclusivamente `K = sum(C[:3, :3])/9`; esta solución
no es una referencia analítica para todas las componentes de la rigidez ni
para el módulo de corte. La derivación radial también está en
[Milton, ecuación 7.11 y su desarrollo](https://www.math.utah.edu/~milton/TheoryCompositesNOPRINT.pdf).

La comprobación registra si, en la malla más fina de cada método, tanto el
error relativo frente a `K_H` como el cambio respecto a la malla anterior
son menores o iguales a **0.1 %**. Es una comprobación empírica de esas mallas.
No demuestra independencia para cualquier refinamiento posterior. Los radios
físicos permanecen fijos: la voxelización puede producir errores no monótonos.
Un estudio que termina correctamente puede no alcanzar ese objetivo; el
JSON y la consola indican ambos estados por separado.

### Tiempos

Cada punto mide la llamada completa a `homogenize`: preparación del material,
seis cargas desde cero, auditorías y ensamblaje de la rigidez. Se sincroniza
CUDA antes y después del reloj. Se excluyen generación de vóxeles, imports y
creación del contexto CUDA. La primera llamada se guarda separadamente de
las tres repeticiones calientes predeterminadas; puede incluir compilación
y preparación de cachés. La figura muestra mediana y rango mínimo–máximo de
las repeticiones calientes, no intervalos de confianza. Se alterna el orden
de los métodos entre niveles dentro de un único proceso.

Las normas nativas de residuo de GaNi y HEX8 son distintas. La figura muestra
también el error analítico para interpretar el tiempo junto a la precisión
obtenida. Para campañas controladas usa los protocolos de reproducción.

## Geometrías de fibras

Selecciona la geometría física, la resolución y un directorio nuevo:

```bash
cufhom-generate --output results/geometry --vf .08 --ar 4 --grid 72
cufhom-generate --help
cufhom-generate --help-all
```

Las longitudes `--box` y `--diameter` usan una misma unidad. `--grid` indica
vóxeles por eje; `--vf` es la fracción volumétrica objetivo; `--ar` la relación
longitud/diámetro. La ayuda habitual muestra opciones físicas y sus valores
predeterminados. La ayuda completa añade los presupuestos de relajación,
propuestas y tolerancias; siguen disponibles para comandos existentes.

Los valores predeterminados son fibras rectas, forma cápsula, `vf=0.15`,
`ar=8`, `box=16`, diámetro 1, malla 96³ y dos hilos. `--fiber-model flexible`
selecciona fibras segmentadas. Las fibras rectas usan por defecto A2 y cierre A4;
las flexibles, A2. `--a2 a11 a22` especifica orientación diagonal con
`a33=1-a11-a22`; `--orientation-json` admite tensores A2/A4 completos.

Por ejemplo, para pedir fibras cilíndricas y orientación anisótropa:

```bash
cufhom-generate --output results/oriented --shape cylinder \
  --vf .08 --ar 4 --box 12 --grid 72 --a2 .6 .3
```

La generación y la voxelización se auditan separadamente. El directorio de
salida contiene la configuración, el resumen y la geometría aceptada.
Las tolerancias de orientación o penetración no son tolerancias de equilibrio.
No se reutilizan directorios de salida existentes.

## Controles avanzados y compatibilidad

No es necesario combinar `dtype` y `working` al usar `precision`. Esos parámetros
siguen disponibles para código existente. Una combinación contradictoria se
rechaza. Al omitir `precision`, un `IndexedMaterial` conserva su almacenamiento;
el modo de trabajo y la tolerancia conservan los valores heredados. Por ejemplo,
un descriptor FP32 preparado conserva `rtol=1e-6` salvo indicación explícita.
No se vuelve a redondear un descriptor para hacer coincidir otro modo solicitado.

`atol=0`, `maxiter=3000`, `storage="auto"`, `execution="optimized"` y
`evaluation="df64"` permanecen predeterminados. Los controles de ejecución,
residencia y evaluación sirven para estudios especializados; no hacen falta
para elegir precisión. GaNi exige planificación automática de memoria.

`verify_tensor=True` conserva campos y reevalúa la matriz en CPU FP64, con mayor
coste. `energy_rtol` cambia el criterio de aceptación a una estimación de error
energético en CUDA. Ambas opciones están desactivadas por defecto; sus resultados
no certifican por sí solos el redondeo ni el error respecto del continuo.

La API `solve_homogenization(parameters)` sigue disponible para geometrías de
fibras orientadas, archivos de configuración y exportación de campos. Conserva
sus claves históricas, como `solver_real_dtype` y `fft_backend`. El selector
`precision` descrito aquí corresponde a `homogenize`.

## Campañas del artículo

Las siete microestructuras, materiales, protocolos y comparadores están en
[benchmarks/paper/](../benchmarks/paper/README.md). Las llamadas calientes
resuelven otra vez las seis cargas desde cero. Los relojes y la memoria se
registran por campaña; los perfiles GPU no sustituyen los tiempos sin
instrumentación. Consulta [REPRODUCIBILITY.md](REPRODUCIBILITY.md) para los pasos
CPU/CUDA y la distinción entre nuevas medidas y las cifras del artículo.
