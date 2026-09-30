# Suavizado fusionado desde el comienzo: sustituido

El promedio de aproximaciones se retrasa un paso para fusionarlo con el
kernel que actualiza x/r y reduce r.r. G09: **3,0474 s / 276 CG**, frente a
**3,0338 s / 288 CG**, una medición después del calentamiento. La diferencia
es pequeña y no demuestra una ganancia; no se amplió esta variante porque
la activación tardía dio un ahorro claro y confirmado en cinco repeticiones.

Se conservan la validación CPU64, auditoría física 1e-5 y el prototipo.
La fusión sí se aprovecha en producción, aplicada solo cerca de la tolerancia.
`PYTHONPATH=benchmarks:descartados/suavizado_fusionado_20260930` permite ejecutar
el benchmark. Los hashes del núcleo histórico `4efc2bf` están en el JSON.
