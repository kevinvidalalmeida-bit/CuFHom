# Prototipo de las fusiones lineales

`linear_kernel_trials.py` conserva el ensayo con parches temporales de clases:
especialización por dimensiones, mapa adjunto con `p.Ap`, dirección CG con
mapa directo y combinaciones. No pertenece al solver instalado.

Resultados de los cribados:

- [Especialización y producto fusionado](../../benchmarks/g09_linear_20260929/kernel_trials.json).
- [Dirección y producto fusionados](../../benchmarks/g09_linear_20260929/direction_trials.json).

La especialización queda descartada. Las dos fusiones se integraron con
argumentos explícitos y salidas propias, sin cachés globales de vectores.
El archivo conserva el último piloto, cuyo `main` mide las cuatro variantes
de dirección/producto. `trial(specialized=True)` permite reconstruir el primer
cribado. Para ejecutar este archivo histórico hay que añadir `benchmarks/`
a `PYTHONPATH`; la reproducción vigente del cambio integrado usa
`profile_g09_linear.py --defaults`.
