# Prototipo de energía: integrado

Este archivo conserva el piloto que permitió medir la eliminación del vector
Hp. El método aceptado está en
[linear_energy.py](../../src/cufhom/pipeline/linear_energy.py): tiene interfaz
explícita para CG, comparte la fórmula del adjunto con los mapas existentes y
libera el espectro consumido antes de construir el siguiente H.

Producción y las pruebas normales usan esa implementación. Este parche
temporal no se importa desde `src/`.

[Mediciones y validación](../../benchmarks/linear_strategy_20261002/RESULTS.md).
