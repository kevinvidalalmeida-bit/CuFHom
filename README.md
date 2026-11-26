<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cufhom-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/cufhom-logo-light.svg">
    <img src="docs/assets/cufhom-logo-light.svg" alt="CuFHom" width="500">
  </picture>
</h1>

<p align="center"><strong>Homogenización FFT de materiales heterogéneos en CPU y GPU.</strong></p>

**CuFHom — CUDA Fourier Homogenization** calcula la rigidez efectiva de sólidos
periódicos voxelizados. Resuelve seis problemas de equilibrio en notación Mandel
mediante Fourier–Galerkin con integración numérica (**GaNi**) o elementos
hexaédricos trilineales completamente integrados (**HEX8**).

[Instalación](#instalación) · [Inicio rápido](#inicio-rápido) ·
[Guía de uso](docs/USAGE.md) · [Reproducción](docs/REPRODUCIBILITY.md)

## Características

- **Dos formulaciones:** GaNi y HEX8 con una misma entrada de etiquetas y
  matrices elásticas por fase, incluidas fases anisótropas.
- **Tres precisiones en GPU:** FP32, FP64 y mixto, con evaluación fresca del
  residuo y aceptación explícita de las seis cargas.
- **Geometrías periódicas:** fibras rectas o flexibles, con controles de
  fracción volumétrica y orientación.
- **Experimentos reproducibles:** siete microestructuras, protocolos y
  registros de entradas, convergencia, tiempos y memoria.

## Instalación

Requiere **Python 3.11 o posterior**. Desde una copia del repositorio:

```bash
git clone https://github.com/kevinvidalalmeida-bit/CuFHom.git
cd CuFHom
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

En Windows, activa el entorno con `.venv\Scripts\Activate.ps1` en PowerShell.
La instalación básica incluye NumPy, SciPy y Numba y funciona en CPU.

### Habilitar GPU

Con una GPU NVIDIA, controlador compatible y **CUDA Toolkit 12.x** instalado:

```bash
python -m pip install -e '.[gpu]'
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount(), 'GPU disponibles')"
```

El extra `gpu` instala CuPy 13 para CUDA 12; la validación local usó CuPy 13.6.
Si usas otro entorno CUDA, consulta la
[instalación de CuPy](https://docs.cupy.dev/en/v13.6.0/install.html).
Si las bibliotecas CUDA están instaladas mediante wheels de NVIDIA, ejecuta
`source tools/env_gpu.sh` en el entorno activo para localizar bibliotecas y cabeceras.

### Comprobar la instalación

```bash
python examples/smoke_cpu.py
cufhom-generate --help
```

El ejemplo compara la rigidez de un material homogéneo con su solución analítica.
Las instrucciones de pruebas CPU/GPU están en [docs/TESTING.md](docs/TESTING.md).

## Inicio rápido

La entrada es una imagen 3D de etiquetas y una tabla de rigideces por fase:

```python
import numpy as np
from cufhom.fft import homogenize

# Dos fases isótropas; módulos expresados en las mismas unidades de tensión.
labels = np.indices((4, 4, 4)).sum(axis=0).astype(np.int32) % 2
C = 2 * np.eye(6)
C[:3, :3] += 1
stiffness = np.stack([C, 4 * C])

result = homogenize(labels, stiffness, precision="fp64")
print(result["C"])
print(result["backend"], result["precision"])
```

`result["C"]` es la rigidez efectiva 6 × 6, con las mismas unidades que la
entrada. El orden Mandel es `(11, 22, 33, 23, 13, 12)`, con normalización √2 en
las componentes cortantes. `result["audits"]` contiene las seis comprobaciones
de equilibrio y `result["iterations"]`, sus conteos de iteraciones.

| Selección | CPU | GPU NVIDIA | Material y solución / iteraciones |
| --- | --- | --- | --- |
| `precision="fp32"` | Sí | Sí | FP32 / FP32 |
| `precision="fp64"` | Sí | Sí | FP64 / FP64 |
| `precision="mixed"` | — | Sí | FP64 / FP32 |

El dispositivo se detecta automáticamente. Para exigir GPU y usar modo mixto:

```python
result = homogenize(labels, stiffness, backend="cupy", precision="mixed")
```

Sin seleccionar precisión, las entradas sin preparar usan FP64 en CPU y mixto
en GPU. GaNi es la formulación predeterminada; `discretization="hex8"` selecciona
HEX8. Las tolerancias, la referencia y la planificación de memoria tienen
valores predeterminados. La [guía de uso](docs/USAGE.md) explica las opciones,
las convenciones y la interpretación de los residuos.

## Ejemplos

Ejecuta los comandos desde la raíz del repositorio:

| Comando | Contenido |
| --- | --- |
| `python examples/smoke_cpu.py` | Validación analítica en CPU |
| `python examples/smoke_cpu.py --geometry` | Validación y celdas de fibras rectas y flexibles |
| `python examples/hashin_convergence.py` | Hashin en GPU y mixto: una figura con corte, convergencia GaNi/HEX8 y tiempos |
| `cufhom-generate --output results/geometry --vf .08 --ar 4 --grid 72` | Generación y auditoría de una geometría periódica |

Para obtener la figura completa de Hashin, con el entorno CUDA indicado arriba:

```bash
python -m pip install -e '.[gpu,examples]'
python examples/hashin_convergence.py
```

Usa GPU, precisión mixta y selección automática de mallas. Guarda una sola figura
con corte, convergencia y tiempos, y los datos CSV/JSON en una carpeta nueva bajo
`results/hashin/`. Puedes elegir `--precision fp64` o cambiar la selección con
`--grid-policy`; `auto` es suficiente para empezar. La
[guía del ejemplo](docs/USAGE.md#ejemplo-analítico-de-hashin) explica las opciones
y el criterio de convergencia.

`cufhom-generate --help` muestra las opciones habituales y `--help-all`, los
controles completos. Los resultados se escriben en directorios nuevos.

## Reproducir los experimentos

Las entradas, protocolos y ejecutores están en
[benchmarks/paper/](benchmarks/paper/README.md). Para comprobar el protocolo y
resolver dos condiciones pequeñas en CPU:

```bash
python benchmarks/paper/run.py --dry-run
python benchmarks/paper/run.py --backend numpy --small --output results/smoke
```

Las campañas completas usan CUDA y guardan las seis auditorías, matrices,
huellas de las entradas y fuentes, entorno y fronteras de medición. Consulta la
[guía de reproducción](docs/REPRODUCIBILITY.md) y los
[protocolos de campaña](benchmarks/paper/README.md) para nuevas mediciones.

## Estructura

```text
src/
  geometry/       Generación, voxelización y auditoría de geometrías
  materials/      Elasticidad y convenciones Mandel
  fft/linear/     Solvers GaNi y HEX8
  cli/            Comando de generación de geometrías
core/             Núcleo incorporado de FFTHomPy
examples/         Ejemplos ejecutables
benchmarks/paper/  Entradas y protocolos de reproducción
tests/            Validación numérica e integración
docs/             Guías y logo del proyecto
```

## Contribuir

Para instalar las herramientas de desarrollo:

```bash
python -m pip install -e '.[dev]'
python -m pytest -m 'not gpu and not slow'
```

[CONTRIBUTING.md](CONTRIBUTING.md) describe las comprobaciones de estilo,
pruebas y empaquetado. Las pruebas CUDA requieren una GPU disponible y
`python -m pytest --gpu -m gpu`.

## Licencia y cita

El código propio usa la [licencia MIT](LICENSE). El núcleo incorporado conserva
su [licencia](core/LICENSE) y la
[atribución a FFTHomPy](third_party/ffthompy/README.md). Los datos externos
mantienen sus [condiciones y procedencia](benchmarks/paper/inputs/README.md).
Para citar el software, consulta [CITATION.cff](CITATION.cff).
