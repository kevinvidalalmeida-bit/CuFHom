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
curvatura geométrica. La separación mínima y la reanudación se admiten en el
modo recto.

API de generación:

```python
from pathlib import Path
from cufhom.pipeline.geometry import GeometryRequest, generate_geometry

summary = generate_geometry(GeometryRequest(
    output=Path("runs/example"), vf=.08, ar=4, box=12, grid=72,
    solver="a2", a2=(.6, .3), shape="cylinder"))
```

## FFT lineal

```python
import numpy as np
from cufhom.pipeline.fft_solver import solve_homogenization

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
11, 22, 33, 23, 13, 12. `cufhom.engineering` convierte a constantes ingenieriles.
El solver también acepta `Cfield_array` y geometrías constitutivas compiladas.
La ruta GPU (`fft_backend="cupy"`) aplica los defaults compatibles de RFFT,
material indexado, proyección directa, fusiones y auditoría del residuo real.
Las formulaciones de referencia se mantienen para verificación.

## FFT no lineal

```bash
cufhom-nonlinear --case homogeneous --component all --backend scipy \
  --maximum .01 --increment .001 --output runs/nonlinear01
```

La matriz usa plasticidad J2 con endurecimiento isotrópico; las fibras son
elásticas. Las seis curvas incluyen deformaciones normales y cortantes
ingenieriles. Hay checkpoints, reinicio y descarga mediante `--unload`.

Para una geometría propia usa `--case geometry --geometry geometry.npz
--material material.json`. El material se especifica explícitamente:

```json
{
  "matrix": {"E": 3, "nu": 0.3, "yield_stress": 0.015, "hardening": 0.03},
  "fiber": {"E_L": 70, "E_T": 8, "nu_LT": 0.25, "nu_TT": 0.3, "G_LT": 5}
}
```

API: `cufhom.pipeline.nonlinear_solver.solve_nonlinear_homogenization`.
El [ejemplo CPU](examples/smoke_cpu.py) comprueba límites analíticos de ambos
solvers. `cufhom-nonlinear --help` muestra los controles de carga y reinicio.

## Campañas Sobol

`cufhom-sobol --smoke` ejecuta un diseño y una realización con FFT GPU.
La configuración está en `cufhom.cli.sobol`; `--points`, `--seeds` y
`--no-overlap` permiten acotar una campaña. Todos los diseños usan
el mismo generador certificado, con cilindros y A4. Los resultados quedan en
`results/` y están excluidos de Git. CuPy/CUDA se comprueba antes de iniciar.

## Estructura y distribución

- `src/cufhom`: geometría, solvers y comandos.
- `src/cufhom_core`: núcleo espectral y formulaciones de referencia.
- `tests`: controles físicos, gradientes, CPU/GPU y comandos.
- `examples`: ejemplos independientes de campañas externas.
- `third_party`: atribución del código incorporado.

No se incluyen resultados, binarios de solvers externos ni rutas experimentales.
[Publicación y licencia](docs/PUBLISHING.md).
