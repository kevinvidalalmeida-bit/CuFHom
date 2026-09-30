# Reajuste de la referencia isotrópica: sin ventaja en la cota

Se calculó la cota material `max(lambda(C0^-1/2 C C0^-1/2)) / min(lambda(...))`
sobre las dos fases de G09. C0 es isotrópica, de modo que esta comprobación
es invariante frente a las rotaciones locales. Se varía K/G con escala global
fija: un factor escalar común no cambia esa razón espectral.

El K/G actual es **3,098423**, con cota **80,32009**. La minimización escalar
devuelve **3,098410** y **80,32012**: no mejora la selección actual. La pequeña
diferencia numérica no justifica cambiar el material de referencia.

Esto es una cota basada en tensores constitutivos; no calcula el número de
condición exacto del operador FFT ni prueba que ningún K/G pueda reducir sus
iteraciones. No se le atribuye una medición de tiempos.
`check_bound.py --material RUTA --output RUTA` conserva la comprobación;
`report.json` incluye el barrido y el resultado del optimizador.
