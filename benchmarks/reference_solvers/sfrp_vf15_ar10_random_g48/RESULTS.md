# CuFHom: referencia SFRP, Vf 15 %, AR 10, orientación aleatoria, 48³

Fecha: 2026-09-29. Equipo: AMD Ryzen 9 9950X (16 núcleos físicos) y NVIDIA
GeForce RTX 5070 Ti (16 GB). Carga común: `epsilon_xx = 0.001`, con una sola
componente activa. La preparación previa de CUDA, planes FFT y kernels se
excluye de los tiempos de CuFHom. Los tiempos de CuFHom son la mediana de tres
repeticiones calientes; DAMASK y AMITEX se ejecutaron una vez por variante.

## Geometría común

| Propiedad | Valor |
|---|---:|
| Fibras periódicas rectas | 31 |
| Volumen de fibra voxelizado | 0.1505263 |
| Relación de aspecto | 10 |
| Dominio | 12 x 12 x 12 |
| Malla | 48 x 48 x 48 |
| Semilla | 20260929 |

La geometría y sus orientaciones están en `geometry.npz`; las fibras continuas
que utiliza Mérope están en `continuous_fibers_master.csv`.

## CuFHom lineal en GPU

| Ruta | Medidas de pared (s) | Mediana (s) | Cxxxx (GPa) |
|---|---:|---:|---:|
| Densa, FFT compleja, núcleo histórico | 0.07239, 0.07321, 0.07219 | 0.07239 | 6.46533761 |
| RFFT, campo `sym21` indexado, planes manuales y kernels fusionados | 0.00814, 0.00795, 0.00880 | 0.00814 | 6.46533728 |

La configuración optimizada es 8.90 veces más rápida. La diferencia relativa
entre ambas rigideces es `6.07e-8`.

## DAMASK Grid en CPU

Cada corrida usa `OMP_NUM_THREADS=1` y MPI ligado a núcleos físicos. La malla
no admite 32 rangos en la descomposición MPI 2D de DAMASK; 16 fue el máximo
válido y el mejor valor probado para el esquema básico.

| Ajuste de `spectral_basic` | Tiempo (s) | Estado |
|---|---:|---|
| 1 rango MPI | 50.29 | Convergió, 32 iteraciones |
| 4 rangos MPI | 13.96 | Convergió |
| 8 rangos MPI | 8.42 | Convergió |
| 16 rangos MPI | 6.96 | Convergió |
| 32 rangos MPI | — | No admite la descomposición de 48³ |

| Formulación, 16 rangos MPI | Tiempo (s) | Estado |
|---|---:|---|
| `spectral_basic` | 6.96 | Convergió |
| `spectral_polarization` | 10.46 | Convergió |
| `spectral_Galerkin` | 3.28 | Convergió |

La elección de rendimiento para esta RVE y carga es `spectral_Galerkin` con
16 rangos MPI.

## AMITEX_FFTP en CPU

Se usó `OMP_NUM_THREADS=1`, MPI ligado a núcleo y el esquema básico. Para la
variante acelerada con filtro hexa, cuatro rangos fue el mínimo medido.

| Ajuste acelerado hexa | Tiempo (s) | Estado |
|---|---:|---|
| 1 rango MPI | 2.24 | Convergió, 57 iteraciones totales |
| 4 rangos MPI | 0.94 | Convergió, 57 iteraciones totales |
| 8 rangos MPI | 1.07 | Convergió, 57 iteraciones totales |
| 16 rangos MPI | 1.27 | Convergió, 57 iteraciones totales |

| Algoritmo, 4 rangos MPI | Tiempo (s) | Estado |
|---|---:|---|
| Básico, acelerado, filtro hexa | 0.94 | Convergió, 57 iteraciones totales |
| Básico, acelerado, filtro octa | 1.01 | Convergió, 70 iteraciones totales |
| Básico, sin aceleración, filtro hexa | — | Residuo de equilibrio `NaN`; no es un tiempo válido |

La elección de rendimiento es aceleración activada, filtro hexa y cuatro
rangos MPI.

## Mérope

| Regla de voxelización | Tiempo (s) |
|---|---:|
| `Center` | 0.008 |
| `Average` + Reuss | 0.026 |

## Alcance de la comparación

CuFHom rota una ley elástica transversalmente isótropa con la orientación local
de cada fibra. DAMASK y AMITEX emplean dos fases elásticas isotrópicas como
sustituto de tiempo: comparten la misma máscara de fibras y la misma malla,
pero aún no reciben la ley TI por voxel. Por tanto, estos tiempos comparan
infraestructura FFT y opciones de solver; no validan que los tres programas
predigan la misma rigidez física de SFRP.

Los registros crudos, archivos de entrada y salidas se guardan en
`results/` y `reference_inputs/`. Para repetir CuFHom GPU:

```bash
source benchmarks/reference_solvers/env_gpu.sh
.venv/bin/python benchmarks/reference_solvers/run_cufhom_gpu.py
```
