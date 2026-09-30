# Prototipo de suavizado tardío: integrado en producción

Se archiva la implementación temporal duplicada; **el método fue aceptado**.
Se activa cuando la norma del residuo CG baja a diez veces la tolerancia.
El peso del promedio se fusiona en la siguiente actualización x/r; se
completa antes de auditar. No utiliza etiquetas ni parámetros de fibras.

Cribado: 3,0345 → 2,9455 s. Confirmación del prototipo, cinco repeticiones
alternadas: **3,0364 → 2,9480 s**, **288 → 276 CG**. Máximo residuo físico
9,61e-6. La implementación definitiva, sus mediciones y validación están en
[el informe actual](../../benchmarks/g09_smoothing_20260930/RESULTS.md).

La estimación armónica procede de combinar residuos CG ortogonales en
aritmética exacta. En float32 no se considera una cota certificada: se
recalcula b-Hy y se reinician ambas secuencias si falla. Después se mantiene
la auditoría de las ecuaciones físicas originales.

El prototipo no es una ruta alternativa instalada. Para volver a ejecutarlo:
`PYTHONPATH=benchmarks:descartados/suavizado_integrado_20260930`; la referencia
del benchmark desactiva el suavizado actual. Las mediciones originales usan
`4efc2bf` y sus hashes se conservan en `confirm.json`.
