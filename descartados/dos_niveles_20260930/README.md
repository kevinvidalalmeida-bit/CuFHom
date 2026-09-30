# Corrección de Galerkin con modos Fourier: descartada

30 de septiembre de 2026. Prueba general, sin detectar fibras, orientación ni
relación de aspecto. Se conserva exactamente el operador fino de CuFHom.

Se construyó una base real ortonormal U en coordenadas RFFT, respetando pares
conjugados, modo medio y Nyquist. Con E = U* H U se probaron:

- Bloque: B = I - U U* + U E^-1 U*.
- Balanceada: B = (I - U E^-1 (HU)*) (I - HU E^-1 U*) + U E^-1 U*.

Ambas son SPD para U de rango completo y H SPD. La fórmula balanceada sigue
[Nabben y Vuik, ecuación 4](https://diamhomes.ewi.tudelft.nl/~kvuik/papers/Nab04V.pdf),
con precondicionador base identidad porque nuestras coordenadas ya están
precondicionadas. La elección de modos Fourier bajos es un ensayo propio.
No reproduce GenEO ni Schwarz con autoproblemas locales.

## Comprobaciones

La base pasó reconstrucción de RFFT real e isometría CPU64. Ambas fórmulas se
compararon con matrices SPD densas independientes, con tolerancia 2e-12 y
positividad comprobada. Laminado e inclusiones compactas 10³ coincidieron con
CPU64; no mejoraron sus iteraciones (17→54 y 149→157 para la balanceada).
`small_validation.json` conserva los casos y las auditorías.

## G09 240³, seis cargas, rtol físico 1e-5

| Ruta | Tiempo total | Iteraciones totales |
| --- | ---: | ---: |
| Base actual | 3,037 s | 288 |
| Bloque de 18 modos | 4,557 s | 387 |
| Balanceada de 18 modos | 6,410 s | 294 |

Cribado de una repetición medida después de un calentamiento por variante,
con resultados coherentes en ambos. Se descarta sin ampliar repeticiones,
pues ninguna reduce las iteraciones y ambas empeoran ampliamente el tiempo.
Incluye construir la base una vez por material y reutilizarla en las seis cargas.
No son medianas de cinco repeticiones ni una comparación exhaustiva de bases.

E se construyó con el operador real; su defecto relativo de simetría fue
1,45e-7 y sus autovalores estuvieron entre 3,59 y 6,19. Construcción: 140 ms
(bloque) y 262 ms (balanceada). La matriz HU retenida por la balanceada ocupa
3,011 GB, además de memoria temporal y del solver. La base U se representa
mediante índices dispersos y no necesita 18 vectores densos.

Interpretación: esta selección de frecuencias bajas no captura lo que
necesitamos corregir en G09. No descarta bases adaptadas al operador/material,
pero sí su activación indiscriminada como ruta más rápida.

`cufhom_twolevel.py` es el prototipo. `validate.py` ejecuta las comprobaciones
pequeñas; `benchmark.py` usa el runner activo con rutas explícitas y requiere
`PYTHONPATH=benchmarks:descartados/dos_niveles_20260930`. Los parches solo viven
dentro de su contexto de prueba; producción no importa este directorio.
