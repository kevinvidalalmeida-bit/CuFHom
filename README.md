# CuFHom

CuFHom resuelve homogenización FFT periódica lineal y no lineal. Incluye rutas
CUDA para la resolución espectral y herramientas para generar y voxelizar
microestructuras de fibra corta. El solver lineal también acepta campos
constitutivos generales mediante `Cfield_array`.

## Inicio rápido CPU

```bash
python3 -m venv .venv
source .venv/bin/activate
python -I -m pip install -c requirements-cpu.lock -e '.[test]'
python -I -m pytest -q
python examples/smoke_cpu.py
```

## Generación de microestructura

```bash
cufhom-generate --output runs/geometry01 \
  --box 12 --ar 4 --vf .08 --grid 72 --a2 .6 .3 \
  --min-gap .02 --initial-count 24 --batch-size 4 --candidates 8 \
  --attempts 2 --max-stages 12 --threads 2
```

El generador produce fibras rectas o flexibles y registra la auditoría de
contactos, volumen voxelizado y momentos de orientación A2/A4.

### Elegir A2 o A4

`--a2` define el tensor objetivo; `--solver` decide qué momentos se imponen.
El modo predeterminado para fibras rectas comprueba **A2 y A4**, incluso si
solo se proporciona `--a2`. En ese caso calcula A4 mediante una distribución
angular central gaussiana. A2 por sí solo no determina un A4 único.

| Objetivo | Parámetros |
| --- | --- |
| Solo A2 | `--solver lbfgs-a2 --a2 .6 .3` |
| A2 y A4 calculado a partir de A2 | `--solver paper-a4 --a2 .6 .3` |
| A4 proporcionado como tensor completo | `--solver paper-a4 --orientation-json orientacion.json` |

`--a2 .6 .3` corresponde a `A2 = diag(0.6, 0.3, 0.1)`: los dos valores son
`a11` y `a22`, y `a33 = 1 - a11 - a22`. Deben ser no negativos y sumar como
máximo uno. Si se omiten, se usa `A2 = I/3`.

Ejemplo completo con **solo A2**, cilindros rectos y AR = 30:

```bash
cufhom-generate --output runs/ar30_a2_01 \
  --solver lbfgs-a2 --a2 .6 .3 --shape cylinder \
  --vf .05 --ar 30 --diameter 1 --box 60 --grid 192 \
  --overlap-tolerance 1e-6 --initial-count 48 --batch-size 92 \
  --seed 20261002 --threads 1 --wall-seconds 30
```

El modo A2 usa por defecto una tolerancia de penetración de `1/120` del
diámetro; el ejemplo la fija en `1e-6`, igual que el modo A4.
La separación adicional `--min-gap` está disponible en el modo A4.

Para imponer **A2 y A4** con la misma orientación media:

```bash
cufhom-generate --output runs/ar30_a4_01 \
  --solver paper-a4 --a2 .6 .3 --shape cylinder \
  --vf .05 --ar 30 --diameter 1 --box 60 --grid 192 \
  --min-gap .001 --initial-count 48 --seed 20261002 --threads 1 --wall-seconds 30
```

También se puede proporcionar A2 como matriz JSON, incluidos términos fuera
de la diagonal:

```json
{"A2": [[0.6, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.1]]}
```

Se carga con `--orientation-json orientacion.json`; para imponer solo A2,
añadir `--solver lbfgs-a2`. A2 debe ser simétrico, positivo semidefinido y
tener traza uno. Un A4 explícito usa la clave `"A4"` y tamaño `3×3×3×3`;
si se proporcionan ambos tensores, su contracción debe coincidir con A2.

| Entrada geométrica | Significado |
| --- | --- |
| `--vf .05` | Fracción de volumen objetivo: 5 % |
| `--ar 30` | Longitud/diámetro; la longitud es `AR × diámetro` |
| `--diameter 1` | Diámetro en µm |
| `--box 60` | Lado del cubo periódico en µm |
| `--grid 192` | Voxelización de `192³`; resolución por diámetro = `grid × diámetro / box` |
| `--shape cylinder` | Cilindros con extremos planos |
| `--seed 20261002` | Semilla para reproducir la generación |
| `--output runs/caso01` | Directorio nuevo para geometría y certificados |

La revisión adicional agrupó la evaluación de candidatas y reutiliza las
evaluaciones idénticas de A2. La comparación de tres ejecuciones completas,
con Numba ya compilado, midió **2,09× en A2 y 3,09× en A4** para 458 fibras.
Ambos casos conservaron exactamente sus geometrías continuas anteriores y
exportaron resultados aceptados. Véanse los
[parámetros, tiempos y límites de esa medición](benchmarks/sam_optimization_review_20261002/REVIEW.md).

### Cilindros rectos, orientación isotrópica y AR alto

`cufhom-generate` usa A4 y L-BFGS por defecto para fibras rectas. La opción
`--shape cylinder` calcula el volumen y voxeliza cilindros con extremos planos;
los contactos se comprueban mediante cápsulas, incluyendo imágenes periódicas.
Sin `--a2`, el objetivo es isotrópico en A2 y A4.
El modo A4 evalúa 128 candidatas por fibra y aplica perturbaciones colectivas
pequeñas en los reintentos; `--retry-shake 0` permite desactivarlas.

Ejemplo breve verificado: AR = 30, Vf = 5 % y cubo de lado 60 diámetros:

```bash
cufhom-generate --output runs/ar30_vf05_01 \
  --shape cylinder --vf .05 --ar 30 --box 60 --min-gap .001 \
  --initial-count 48 --grid 192 --seed 20261002 --threads 1 --wall-seconds 30
```

Este caso generó y exportó 458 cilindros en 1,62 s, con las compilaciones de
Numba ya disponibles. Pasaron las comprobaciones de contactos periódicos,
separación mínima, volumen y orientación A2/A4. El tiempo depende del equipo,
la geometría y la resolución; la primera ejecución incluye la compilación.
Los lotes se reducen tras un rechazo y las relajaciones que mejoran reciben
más iteraciones dentro de su presupuesto. Solo se guarda geometría que pasa
la auditoría independiente.
Se puede continuar en otro directorio, conservando el estado aleatorio:

```bash
cufhom-generate --output runs/ar30_vf05_02 \
  --resume runs/ar30_vf05_01/last_geometrically_valid.npz --wall-seconds 30
```

Una pausa durante una inserción vuelve al último conjunto certificado y repite
esa inserción al reanudar. El código de salida `2` indica un resultado parcial.
`--geometry-only` omite la certificación voxelizada y no exporta `geometry.npz`.
Al alcanzar el número objetivo, reanudar sin esa opción permite comprobar
el volumen y los momentos voxelizados. La aceptación geométrica no certifica
resolución suficiente para FFT.

Estas mejoras corresponden a `cufhom.cli.geometry`; la ruta histórica
`SAMLiteGenerator` conserva el control A2 y corrige su escala de inercia rotacional.
La comparación de algoritmos, las verificaciones y los ensayos históricos
están en [el informe SAM](benchmarks/sam_improvements_20261002/RESULTS.md).

## Solver

```python
from cufhom.pipeline.fft_solver import solve_homogenization
from cufhom.pipeline.nonlinear_solver import solve_nonlinear_homogenization
```

Al elegir `fft_backend='cupy'`, el solver aplica las optimizaciones GPU
compatibles con los datos y la precisión solicitada. En float32 usa RFFT,
almacenamiento simétrico, planes FFT reutilizables y las fusiones verificadas.
El posprocesado GPU acumula los productos en float64. Las opciones explícitas
del usuario tienen prioridad; CPU y los perfiles float64 conservan su precisión.

El CG en coordenadas simétricas combina sus últimas aproximaciones mediante
suavizado de residuos. Se activa cerca de la tolerancia y comprueba el residuo
verdadero antes de aceptar. `linear_residual_smoothing=False` conserva el CG
sin esa combinación para comparar rendimiento y precisión.

En esa ruta, `linear_fft_autotune=True` compara una vez los lotes FFT de 1, 2,
3 y 6 componentes usando el operador completo. Guarda la elección por forma,
precisión, GPU y versiones en `~/.cache/cufhom/fft_tuning_v1.json` (o en
`CUFHOM_FFT_TUNING_CACHE`). Una política `cupy_rfft_batch_mode` explícita tiene
prioridad. El primer uso incluye el coste de selección; los informes registran
ese coste y la procedencia de la elección. `linear_fft_autotune=False` permite
comparar con la política anterior.

`linear_energy_update=True` calcula `p*Hp` en el kernel de material, con suma
float64, y fusiona el adjunto con la actualización del CG. Así elimina la
escritura y lectura de `Hp`. Su auditoría final conserva el operador físico
original. Ambas opciones están activas por defecto en la ruta compatible;
[derivación, validación y tiempos](benchmarks/linear_strategy_20261002/RESULTS.md).

`linear_final_audit_reuse=True` reutiliza la deformación y la tensión de la
última comprobación verdadera de CG. Conserva la proyección física y su
residual; comprueba que los campos pertenezcan al mismo iterado y operador.
También retiene el espectro de tensión para evitar una RFFT final repetida
en la ruta compatible. linear_final_spectrum_reuse=False permite medir
solo ese cambio conservando la reutilización de deformación/tensión.
Los campos iniciales añadidos y las rutas incompatibles usan la evaluación
completa. Está activa por defecto; `False` permite comparar ambas políticas.
[Tiempos, campos y revisión del código](benchmarks/code_review_20261002/RESULTS.md).

La rotación GPU del material calcula directamente Crot = B C Bᵀ en Mandel,
con aritmética float64 y una conversión final a la precisión de la tabla.
Evita los temporales de cuarto orden y los lanzamientos de empaquetado.
[Validación y comparación de los defaults finales](benchmarks/vf_ar_curves_20261002/CODE_REVIEW.md).

La ruta de coordenadas simétricas requiere material indexado y seis cargas
seriales. Los campos constitutivos generales y las configuraciones incompatibles
usan la formulación en deformaciones. Las pruebas cubren ambos caminos; la
velocidad medida en G09 no garantiza la misma aceleración para cualquier campo.

Para campañas sobre una misma geometría, `compile_cfield_geometry(phase, ori)`
prepara los grupos una vez. Requiere arrays de solo lectura y se pasa como
`compiled_cfield_geometry` junto con esos mismos arrays. No hay una caché global
que reutilice silenciosamente materiales o geometrías diferentes.

Para ejecutar el benchmark GPU incluido:

```bash
source benchmarks/reference_solvers/env_gpu.sh
.venv/bin/python benchmarks/reference_solvers/run_cufhom_gpu.py
```

El caso de referencia SFRP de Vf 15 %, AR 10 y orientación aleatoria está en
[`benchmarks/reference_solvers`](benchmarks/reference_solvers/). Sus resultados
comparan CuFHom con DAMASK, AMITEX_FFTP y Mérope.

La campaña [G09 lineal 240³](benchmarks/g09_smoothing_20260930/RESULTS.md)
documenta 288 → 276 iteraciones, tiempos por etapa, validación de campos y
reproducción. La [etapa anterior](benchmarks/g09_linear_20260929/RESULTS.md)
conserva la comparación de las fusiones y el ensamblaje GPU.
La [revisión de artículos](docs/PAPERS_LINEAR_ACCELERATION_20260929.md)
evalúa métodos generales para reducir iteraciones; las propuestas pendientes
están identificadas como investigación.

El [barrido de Vf, AR y vóxeles](benchmarks/vf_ar_curves_20261002/RESULTS.md)
incluye 72 puntos: Vf 5/15/25%, AR 5/10/20, orientación aleatoria y dominio 2L.
Cada punto tiene cinco mediciones de las seis cargas. Compara mallas de
32³ a 256³ y una resolución constante de seis vóxeles por diámetro.

## Estructura

- `src/cufhom`: API, generación y solvers de CuFHom.
- `src/cufhom_core`: núcleo espectral interno.
- `tests`, `examples`, `benchmarks`: validación y casos reproducibles.
- `third_party`: avisos de atribución del código incorporado.
- `descartados`: código retirado y evidencia de los ensayos, fuera del paquete
  instalado y de las pruebas activas.

Los entornos Python, binarios de referencia y resultados grandes están
excluidos del control de versiones.
