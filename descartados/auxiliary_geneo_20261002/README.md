# Espacio auxiliar y GenEO: pilotos retirados

Se implementaron transferencia y adjunto periódicos, elasticidad Q1 con
cuadratura completa, V-cycle simétrico, autoproblemas locales y PCG con
suavizado en la métrica del precondicionador. También se construyó el
Galerkin fino real de los modos seleccionados, sin guardar bases densas
en la malla fina.

Las variantes pasan la validación de campos y residual físico. En G09 todas
mantienen 276 iteraciones y aumentan el tiempo completo:

| Variante | Mediana, seis cargas | Preparación observada |
| --- | ---: | ---: |
| Referencia de auxiliar Q1/GenEO auxiliar | 2,967 s | Incluida |
| Auxiliar Q1 + V-cycle | 5,249 s | 0,247 s en la campaña |
| GenEO auxiliar, tau=0,15 | 5,540 s | 0,533 s en la campaña |
| GenEO, energía fina, tau=0,15 | 7,685 s | 2,722 s en la campaña |
| GenEO, energía fina, tau=0,5 | 9,401 s | 4,431 s en la campaña |

Cada variante se compara con su referencia en la misma campaña; las últimas
dos tienen referencias de 2,963 y 2,965 s. Son tres repeticiones alternadas tras
calentamiento, con preparación reutilizada en las seis cargas e incluida en
el tiempo total. Tau=0,5 selecciona 285 columnas: 162 modos rígidos y 123
deformables; quedan 282 direcciones de energía independientes. No reduce las
iteraciones. Construir su Galerkin fino añade 285 aplicaciones de H.

La derivación SPD garantiza un precondicionador válido, pero el espectro fino
CPU64 y las mediciones muestran que estas bases/transferencias no aceleran
los casos. El promedio constitutivo en la malla auxiliar pierde contraste
local. La energía fina corrige la aproximación de rigidez, pero no consigue
una base suficientemente útil para amortizar su coste. Se retiran estos
pilotos; este resultado no descarta toda construcción futura GenEO.

[Derivación propia y límites](../../benchmarks/linear_strategy_20261002/DERIVATION.md),
[evidencia y resultados](../../benchmarks/linear_strategy_20261002/RESULTS.md).

Repetir sus controles aislados:

```bash
source benchmarks/reference_solvers/env_gpu.sh
OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  descartados/auxiliary_geneo_20261002/test_auxiliary_space.py
```

Los runners de la campaña añaden explícitamente este directorio a su ruta
de importación. Producción y los comandos normales no lo importan.
