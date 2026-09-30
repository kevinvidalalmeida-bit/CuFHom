# CuFHom

CuFHom resuelve homogenización FFT periódica lineal y no lineal de compuestos
de fibra corta. Incluye generación de microestructuras, voxelización y rutas
CUDA para la resolución espectral.

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

## Solver

```python
from cufhom.pipeline.fft_solver import solve_homogenization
from cufhom.pipeline.nonlinear_solver import solve_nonlinear_homogenization
```

Para ejecutar el benchmark GPU incluido:

```bash
source benchmarks/reference_solvers/env_gpu.sh
.venv/bin/python benchmarks/reference_solvers/run_cufhom_gpu.py
```

El caso de referencia SFRP de Vf 15 %, AR 10 y orientación aleatoria está en
[`benchmarks/reference_solvers`](benchmarks/reference_solvers/). Sus resultados
comparan CuFHom con DAMASK, AMITEX_FFTP y Mérope.

## Estructura

- `src/cufhom`: API, generación y solvers de CuFHom.
- `src/cufhom_core`: núcleo espectral interno.
- `tests`, `examples`, `benchmarks`: validación y casos reproducibles.
- `third_party`: avisos de atribución del código incorporado.

Los entornos Python, binarios de referencia y resultados grandes están
excluidos del control de versiones.
