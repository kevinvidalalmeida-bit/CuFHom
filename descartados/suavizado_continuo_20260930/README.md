# Suavizado desde el comienzo: retirado

Cribado G09 240³, seis cargas: **3,0883 s / 276 CG**, frente a
**3,0317 s / 288 CG**. Una medición tras calentamiento; sin relajar 1e-5.
La combinación de aproximaciones reduce iteraciones, pero añade un recorrido
del vector en cada iteración. No se conserva esa variante como opción activa.

Se comprobó frente a matrices SPD CPU64 y dos problemas periódicos pequeños.
La estimación armónica solo dispara una auditoría independiente del residuo.
Referencia conceptual: [Zhou y Walker, 1994](https://users.wpi.edu/~walker/Papers/residual_smoothing,SISC_15,1994,297-312.pdf).
El peso usado aprovecha la ortogonalidad de los residuos CG en aritmética exacta.

Los scripts y JSON conservan el ensayo contra `4efc2bf`. El benchmark usa
`PYTHONPATH=benchmarks:descartados/suavizado_continuo_20260930`; sus rutas de
entrada son explícitas. El CG de referencia desactiva el suavizado nuevo.
