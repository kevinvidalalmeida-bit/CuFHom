# Optimización adicional y entradas A2/A4 — 2 de octubre de 2026

## Cambios verificados

El perfil inicial de una generación A4 de 458 fibras atribuyó 1,823 s de
2,170 s del generador a la selección de candidatas. Son tiempos bajo el
perfilador, usados para identificar el coste; la comparación de rendimiento
de abajo se ejecutó sin perfilador.

Se aplicaron dos cambios:

1. **Evaluación agrupada de candidatas.** El generador reserva la tabla de
   fibras una vez por propuesta. La evaluación de contactos copia las fibras
   existentes una vez por grupo y reutiliza sus buffers. Cada candidata sigue
   siendo una alternativa independiente. Se conserva el mismo núcleo exacto
   de contactos, todas las imágenes periódicas, los contactos con copias de la
   misma fibra y el orden de selección. La política aleatoria permanece igual.
2. **Reutilización de evaluaciones A2.** L-BFGS y su callback solicitaban la
   energía y los gradientes para las mismas coordenadas. La caché conserva una
   copia de las últimas coordenadas y reutiliza ese resultado. Una modificación
   de las coordenadas invalida la caché. La auditoría geométrica sigue siendo
   independiente.

También se aclararon los mensajes del CLI y los ejemplos del README para
seleccionar A2 o A4.

## Comparación completa

Se ejecutaron ambos métodos en el mismo proceso, con una pasada de calentamiento
por método y tres repeticiones, alternando el orden. La compilación de Numba
queda fuera de la medición. Se usó un hilo de Numba y de BLAS. El tiempo incluye
generación, auditorías, voxelización y exportación, medido por `total_wall_s`.

Ambos casos usaron cilindros rectos, AR = 30, diámetro 1, cubo de lado 60,
Vf objetivo 5 %, 458 fibras, grid 192³ y semilla 20261002. El caso A2 impuso
`diag(0.6, 0.3, 0.1)`, tolerancia de contacto `1e-6`, lotes de 92 y 32
candidatas. El caso A4 impuso isotropía en A2/A4, separación 0,001 y 128
candidatas. Los objetivos y parámetros se mantuvieron entre el método anterior
y el optimizado en cada caso.

| Caso | Mediana anterior | Mediana optimizada | Aceleración |
| --- | ---: | ---: | ---: |
| Solo A2 | 1,0621 s | 0,5093 s | 2,09× |
| A2 y A4 | 3,0746 s | 0,9946 s | 3,09× |

Todas las ejecuciones terminaron con código `0`, `accepted=True` y exportaron
`geometry.npz`. En cada caso, las tablas continuas fueron **idénticas elemento
por elemento** entre métodos y repeticiones. Los certificados, volúmenes y
momentos voxelizados pasaron sus comprobaciones. Estos factores corresponden
a los dos casos medidos; dependen del tamaño, la orientación y el equipo.

Las muestras individuales y hashes de fuentes están en [measurements.json](measurements.json).
Las funciones anteriores se conservaron en [reference_before.py](reference_before.py).
Las geometrías y registros de cada repetición están en
`benchmarks/results/sam_optimization_comparison_20261002`.

## Entradas de orientación

- **Solo A2:** `--solver lbfgs-a2 --a2 .6 .3` fija
  `A2 = diag(0.6, 0.3, 0.1)`. Este modo no impone un A4.
- **A2 y A4 calculado:** `--solver paper-a4 --a2 .6 .3` usa el A4 de una
  distribución angular central gaussiana compatible con A2. Esa distribución
  es una elección: A2 no determina un A4 único.
- **A4 explícito:** `--solver paper-a4 --orientation-json orientacion.json`
  acepta la clave `A4` con tensor completo de tamaño `3×3×3×3`. Puede contener
  también `A2`; su contracción debe coincidir.
- **A2 completo:** un JSON con la clave `A2` admite una matriz simétrica
  positiva semidefinida de traza uno. Añadir `--solver lbfgs-a2` impone solo A2.
- Sin `--a2`, el tensor A2 objetivo es `I/3`. En fibras rectas, `auto` selecciona
  A4 y controla ambos momentos. En fibras flexibles selecciona solo A2.

La [guía de entradas del README](../../README.md#elegir-a2-o-a4) incluye comandos
completos. El modo A2 tiene por defecto tolerancia de contacto `1/120` del
diámetro; los ejemplos y la comparación la fijan en `1e-6`. La separación
adicional `--min-gap` requiere el modo A4.

## Validación y reproducción

La batería SAM terminó con **104 pruebas y 7 subpruebas aprobadas**, incluidos
contactos periódicos, alternativas independientes, invalidación de la caché,
exportación A2/A4, reanudación, gradientes y fibras flexibles. El registro está
en [tests.txt](tests.txt). También se compararon cuatro propuestas iniciales
A2/A4, isotrópicas y anisotrópicas: conservaron exactamente las elecciones
obtenidas con la misma semilla antes de la optimización.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_sam*.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  benchmarks/sam_optimization_review_20261002/measure_optimizations.py \
  --output /tmp/sam_optimization_comparison.json \
  --runs /tmp/sam_optimization_comparison_runs
```

Los directorios de salida deben ser nuevos. Esta revisión usa objetivos de
5 % y conserva las campañas densas ya detenidas.
