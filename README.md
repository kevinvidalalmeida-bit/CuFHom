# CuFHom

Geometrías periódicas de fibras cortas y homogenización FFT lineal y no lineal.
La generación usa CPU/Numba. Los solvers admiten CPU/SciPy y GPU/CuPy.

## Instalación y comprobación

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python examples/smoke_cpu.py
python examples/smoke_cpu.py --geometry
```

Python >= 3.11. `requirements-cpu.lock` contiene las versiones comprobadas en
Linux/Python 3.12 y se puede usar con `pip install -c requirements-cpu.lock`.
Para GPU, instala CuPy compatible con tu CUDA en el mismo entorno. Si las
bibliotecas NVIDIA están instaladas mediante wheels, `source tools/env_gpu.sh`
añade sus directorios al buscador de bibliotecas del entorno activo.

## Geometría: entradas A2 y A4

```bash
cufhom-generate --output runs/geometry01 --shape cylinder \
  --vf .08 --ar 4 --diameter 1 --box 12 --grid 72 \
  --solver a4 --a2 .6 .3 --threads 2
```

| Entrada | Significado |
| --- | --- |
| `--vf .25` | Fracción de volumen objetivo: 25 % |
| `--ar 20` | Longitud/diámetro; longitud = AR × diámetro |
| `--diameter 1`, `--box 60` | Diámetro y lado del cubo periódico en µm |
| `--grid 360` | Malla 360³; resolución por diámetro = grid × diámetro / box |
| `--shape cylinder` | Cilindros rectos con extremos planos |
| `--shape capsule` | Segmentos con extremos hemisféricos |
| `--a2 .6 .3` | A2 = diag(0.6, 0.3, 0.1) |
| `--seed 20261002` | Semilla de la generación |

**Solo A2:** `--solver a2 --a2 .6 .3`.
**A2 y A4:** `--solver a4 --a2 .6 .3`.
Si se omite `--a2`, el objetivo es isotrópico, A2 = I/3.
`auto` usa A4 para fibras rectas y A2 para flexibles.

A2 no determina un A4 único. El modo A4 calcula un cierre mediante una
distribución angular central gaussiana, o recibe tensores completos con
`--orientation-json orientacion.json`:

```json
{"A2": [[0.6, 0, 0], [0, 0.3, 0], [0, 0, 0.1]]}
```

El JSON admite `A2` (3×3) y/o `A4` (3×3×3×3), con simetría y contracción
consistentes. Para imponer solo A2 desde JSON, usa `--solver a2`.

Ejemplo isotrópico de 25 % y AR 20:

```bash
cufhom-generate --output runs/ar20_vf25 --shape cylinder --solver a4 \
  --vf .25 --ar 20 --diameter 1 --box 60 --grid 360 \
  --min-gap .001 --initial-count 48 --seed 20261002 --threads 1
```

Los contactos se comprueban con cápsulas conservadoras, también al exportar
cilindros. Se auditan todas las imágenes periódicas pertinentes. El modo A4
usa L-BFGS con escala de inercia, candidatos por lotes y vecinos con margen de
movimiento. Los reintentos conservan el último estado certificado.
A2 admite por defecto penetraciones de hasta 1/120 del diámetro; para una
certificación más estricta usa `--overlap-tolerance 1e-6`, que es el valor
predeterminado de A4. `--min-gap` exige A4.

`summary.json` registra contactos, Vf medido y errores de orientación continua
y voxelizada. `geometry.npz` se exporta solo cuando pasan todos los controles;
contiene `phase`, `ori` y los tensores objetivo. Las fibras continuas y el
checkpoint se conservan también en ejecuciones parciales. El código de salida
`2` indica un resultado parcial o rechazado. `--geometry-only` omite el raster
y su certificación. La fracción solicitada y la convergencia FFT requieren sus
respectivas comprobaciones; la resolución del raster afecta el volumen medido.

```bash
cufhom-generate --output runs/continued \
  --resume runs/geometry01/last_geometrically_valid.npz --wall-seconds 30
```

Para fibras flexibles: `--fiber-model flexible --solver a2 --segments 4`.
Su AR usa longitud de contorno; el modelo representa cadenas de cápsulas con
curvatura geométrica; cada segmento debe medir al menos un diámetro
(`AR >= segments`). La separación mínima y la reanudación se admiten en el
modo recto.

API de generación:

```python
from pathlib import Path
from cufhom.geometry import GeometryRequest, generate_geometry

summary = generate_geometry(GeometryRequest(
    output=Path("runs/example"), vf=.08, ar=4, box=12, grid=72,
    solver="a2", a2=(.6, .3), shape="cylinder"))
```

## FFT lineal

```python
import numpy as np
from cufhom.fft import solve_homogenization

with np.load("runs/geometry01/geometry.npz", allow_pickle=False) as data:
    phase, ori = data["phase"], data["ori"]

C = solve_homogenization(dict(
    input_dir=".", seed=0, phase_array=phase, ori_array=ori,
    Em=3., nu_m=.3, Ef_L=70., Ef_T=8., nu_LT=.25, nu_TT=.3, G_LT=5.,
    fft_backend="scipy", solver_real_dtype="float64",
    solver_rtol=1e-8, require_convergence=True))
```

Los módulos usan las mismas unidades de tensión para todos los materiales;
los ejemplos usan GPa. `C` es la rigidez 6×6 en Mandel, orden
11, 22, 33, 23, 13, 12. `cufhom.materials` convierte a constantes ingenieriles.
El solver también acepta `Cfield_array` y geometrías constitutivas compiladas.
La ruta GPU (`fft_backend="cupy"`) aplica los defaults compatibles de RFFT,
material indexado, proyección directa, fusiones y auditoría del residuo real.
Las formulaciones de referencia se mantienen para verificación.

### Benchmark común con FANS

`cufhom.fft.linear.hex8` proporciona un operador periódico HEX8 con ocho puntos
de Gauss para comparar el mismo sistema nodal que el ejemplo oficial de FANS.
Admite dos fases isotrópicas y float64; conserva todas las frecuencias salvo
las traslaciones uniformes. Sus arrays espaciales usan el orden Z,Y,X y sus
componentes Mandel el orden 11,22,33,12,13,23.

El stencil reúne exactamente las contribuciones de los elementos incidentes.
Las 256 combinaciones de fases permiten almacenar sus bloques sin construir
una matriz global. El solver recalcula `b - K u` y exige
`max(abs(b - K u)) / max(abs(b)) <= rtol`.

Con CuPy y PyTorch instalados en el mismo entorno, una copia del
[código oficial](https://github.com/DataAnalyticsEngineering/UNOCG) y la muestra
original `test/dset_1` de `3d_microstructures.h5` del
[dataset publicado](https://doi.org/10.18419/DARUS-5686):

```bash
python tools/benchmark_hex8.py --phase runs/paper_fans_192/phase.npy \
  --reference external/solver_references/UNOCG --output runs/hex8_comparison
```

El script valida primero operador, carga y precondicionador en mallas pequeñas.
Después alterna las mediciones 192³, con una carga, float64 y residuo máximo
relativo `1e-6`. El código original de FANS audita ambos resultados; se comparan
también desplazamientos y tensiones por celda. La preparación se registra
por separado; el tiempo de resolución incluye la auditoría común y excluye
las transferencias y la comparación de campos. Los datos externos y resultados
permanecen excluidos de Git. `--small-only` ejecuta la comprobación pequeña en CPU.

En la RTX 5070 Ti, tres pares alternados dieron medianas de **22,101 s para
FANS** y **1,463 s para CuFHom-HEX8**: **15,10×** de aceleración local. Ambos
usaron 19 iteraciones y residuo verdadero máximo relativo `3,80e-7`.
Los errores relativos entre los campos completos fueron `3,31e-13` para
desplazamientos y `8,91e-14` para tensiones. La cifra publicada de 1,3008 s
para FANS en H100 tiene otro hardware y alcance de cronometraje.

## FFT no lineal

```bash
cufhom-nonlinear --case homogeneous --component all --backend scipy \
  --maximum .01 --increment .001 --output runs/nonlinear01
```

La matriz usa plasticidad J2 con endurecimiento isotrópico; las fibras son
elásticas. Las seis curvas incluyen deformaciones normales y cortantes
ingenieriles. Hay checkpoints, reinicio y descarga mediante `--unload`.

El comando selecciona la formulación en desplazamientos y el precondicionador
de matriz utilizados en las mediciones de rendimiento. Las rutas de referencia
siguen disponibles en la API Python para verificación.

Para una geometría propia usa `--case geometry --geometry geometry.npz
--material material.json`. El material se especifica explícitamente:

```json
{
  "matrix": {"E": 3, "nu": 0.3, "yield_stress": 0.015, "hardening": 0.03},
  "fiber": {"E_L": 70, "E_T": 8, "nu_LT": 0.25, "nu_TT": 0.3, "G_LT": 5}
}
```

API: `from cufhom.fft import solve_nonlinear_homogenization`.
El [ejemplo CPU](examples/smoke_cpu.py) comprueba límites analíticos de ambos
solvers. `cufhom-nonlinear --help` muestra los controles de carga y reinicio.

## Estructura y distribución

```text
cufhom/                  # Una sola raíz de proyecto
    src/                 # Paquete Python cufhom
        geometry/        # Rectas, flexibles, contactos, orientación y raster
        fft/
            linear/      # Ensamblaje elástico y CG optimizado
            nonlinear/   # Equilibrio incremental y curvas constitutivas
            operators.py # Espacio admisible y precondicionador
            krylov.py    # CG compartido y auditoría del residuo
        materials/       # Elasticidad, rotaciones y plasticidad J2
        cli/             # Generación y curvas no lineales
    core/                # Paquete cufhom_core, aviso MIT conservado
    tests/
        geometry/        # Incluye referencias geométricas independientes
        fft/linear/
        fft/nonlinear/
        materials/
    examples/            # Comprobaciones ejecutables
    tools/               # Preparación del entorno GPU
    third_party/         # Licencias y atribuciones
```

El empaquetado asigna `src/` a `cufhom` y `core/` a `cufhom_core`;
los nombres físicos de las carpetas no cambian los imports de Python.
Instala el proyecto con `pip install -e .` antes de ejecutar sus comandos.

Las APIs públicas son `cufhom.geometry`, `cufhom.fft` y `cufhom.materials`.
Los antiguos imports `cufhom.pipeline.*` se sustituyen por estas rutas.
Las campañas Sobol y su comando se retiraron del paquete.
Se conservan las formulaciones de referencia que validan los operadores
optimizados, los checkpoints funcionales y los controles de convergencia.
Los resultados y las instalaciones locales externas están excluidos de Git.

Para crear una distribución:

```bash
python -m pip wheel . --no-deps --wheel-dir /tmp/cufhom-wheel
```

### Licencias

El núcleo incorporado mantiene la licencia MIT de FFTHomPy y sus avisos de
autoría: [atribución](third_party/ffthompy/README.md).
La licencia del código propio y el nombre completo del titular están pendientes
de elección del autor; el aviso MIT del núcleo no licencia todo el proyecto.
El archivo [LICENSE](LICENSE) registra este estado.
