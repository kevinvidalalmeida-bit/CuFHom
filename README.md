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

## Estructura

- `src/cufhom`: API, generación y solvers de CuFHom.
- `src/cufhom_core`: núcleo espectral interno.
- `tests`, `examples`, `benchmarks`: validación y casos reproducibles.
- `third_party`: avisos de atribución del código incorporado.
- `descartados`: código retirado y evidencia de los ensayos, fuera del paquete
  instalado y de las pruebas activas.

Los entornos Python, binarios de referencia y resultados grandes están
excluidos del control de versiones.
