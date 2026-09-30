# Residuo conjugado (CR): menos iteraciones, más tiempo

G09 240³, seis cargas y residuo físico 1e-5. Cribado de una repetición después
del calentamiento: CG **3,0338 s / 288 iteraciones**; CR **3,1687 s / 276**.
El calentamiento de CR dio 3,1796 s. Se retira este prototipo: su actualización
adicional de vectores cuesta más que las 12 iteraciones evitadas.

Se usa la recurrencia clásica en las mismas coordenadas simétricas; conserva
el residuo verdadero y las auditorías originales. Referencia:
[Saad, segunda edición, §6.8, algoritmo 6.20](https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf).
La validación incluye una matriz SPD densa frente a CPU64 y dos problemas
periódicos pequeños con seis cargas y mallas pares/impares.

`report.json` y `validation.json` preservan la evidencia. `benchmark.py`
requiere `PYTHONPATH=benchmarks:descartados/residuo_conjugado_20260930`.
Se fija `linear_residual_smoothing=False` en la referencia para reproducir
el CG anterior. El núcleo medido corresponde al commit `4efc2bf`; sus hashes
están en el informe. Producción no importa este directorio.
