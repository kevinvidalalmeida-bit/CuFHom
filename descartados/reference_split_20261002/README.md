# Separación de la referencia: piloto fuera de producción

La identidad `H = I + Q_R* (C-C₀) Q_R` es correcta sobre las coordenadas
admisibles y permite evitar parte del trabajo de material. Se probaron
dos implementaciones GPU: norma directa fusionada con el mapa y norma
recurrente basada en ortogonalidad ideal de CG.

La directa ahorra 1,05% en G09, 3,36% en fibras 192³, 3,60% en inclusiones
compactas 256³ y 2,79% en regiones 128³. G08 no mejora. Todas las mediciones
conservan las iteraciones y el residual físico original `1e-5`.

**Se retira como política general predeterminada.** Una fase homogénea
`C = 1e-6 C₀` produce 7,44% de error relativo en el denominador split de
CG por cancelación en FP32; el denominador original queda en ≈4e-8.
Esto no invalida la identidad matemática ni la ganancia en los casos
anteriores. Para integrarla harían falta una protección numérica y una
elección por coste medido que también sirvan para casos generales.

La recurrencia no supera a la norma directa en G09/192³/256³/128³ y
añade una dependencia de ortogonalidad en precisión finita. No se activa.

[Resultados, campos y evidencia](../../benchmarks/code_review_20261002/RESULTS.md).
[Derivación y condiciones](../../benchmarks/code_review_20261002/DERIVATION.md).
Producción no importa estos parches. Los runners del informe los cargan
explícitamente y solo durante el ensayo.

Controles independientes de tensor anisótropo SPD, contrastes negativos,
dirección diferida, suavizado y tres formas pares/impares:

```bash
source benchmarks/reference_solvers/env_gpu.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest \
  descartados/reference_split_20261002/test_reference_split.py -q
```
