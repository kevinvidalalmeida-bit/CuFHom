# Eliminación de trabajo redundante del CG — G09 240³

## Cambio aplicado

El núcleo acepta una estimación inicial implícita (`x0=None`) y usa `R=B`
mediante una copia independiente, sin calcular `A(0)`. Las aplicaciones escalar
y elástica transmiten esa estimación implícita. Los arranques explícitos,
incluidos campos previos, conservan `R=B-A(x0)`. Si existe un callback inicial,
se materializa el vector y se mantiene la aplicación del operador porque el
callback puede modificarlo.

CG comprueba la parada antes de actualizar la dirección siguiente: no genera
una dirección que ya no utilizará por convergencia o por alcanzar `maxiter`.
Se mantiene la frecuencia de comprobación de residuo, el registro de normas y
la secuencia de callbacks. Los dos cambios cubren CG individual y CG de cargas
independientes agrupadas. Además, ya no se retiene una variable local `Ax`
durante todo el solve: un campo de 316.4 MiB en este caso. Esto no es una
medición de reducción del pico de VRAM; el pool CUDA puede conservar memoria.

No se cambiaron `operators.py`, la discretización, la tabla constitutiva,
los criterios de tolerancia, los kernels de reducción ni sus valores por
defecto en esta intervención. No se modificó la geometría G09.

## Comparación reproducible

`profile_g09_cg_cleanup.py` carga, sin checkout, el módulo CG y el inicializador
de carga elástica de la revisión `a42304a7ecf4a638053413f2eef103ec79589305`.
El resto del código FFT/material/proyección es el mismo para todas las
variantes, incluidas las optimizaciones locales anteriores a este cambio.
No se compara contra todo el repositorio antiguo.

Se usaron G09 240³, seis cargas, material monitor 0, RTX 5070 Ti, float32,
`rtol=1e-5`, cinco repeticiones por variante tras calentamiento. El orden se
alterna para reducir sesgo temporal. Los tiempos excluyen lectura/compilación
de geometría, inicialización CUDA y auditorías. Incluyen la construcción
material y el postproceso habitual del solve.

| Variante | Mediana | Iteraciones | Aplicaciones A dentro de CG |
|---|---:|---:|---:|
| Anterior | 4.830 s | 360 | 366 |
| Cambio aplicado, opciones originales | 4.767 s | 360 | 360 |
| Cambio aplicado + reducciones estables existentes | 4.742 s | 360 | 360 |

La mejora aplicada es aproximadamente 1.3%, unos 64 ms por homogenización.
La tercera fila es un experimento optativo, no el nuevo valor predeterminado:
reduce aproximadamente 1.8% frente al anterior, pero cambia la acumulación
de reducciones y sólo se ha contrastado aquí en un material. No sumar los
porcentajes de experimentos independientes.

El conteo excluye formar los seis RHS. Las auditorías añaden otras seis
aplicaciones, fuera de las mediciones. Las actualizaciones de dirección bajan
de 360 a 354 sin reducir el número de iteraciones.

## Verificación de solución

- Todas las cargas de todas las variantes pasan el residuo verdadero,
  aproximadamente `8.58e-6` como máximo, frente al umbral `1e-5`.
- El JSON registra Ceff de cada repetición y su diferencia relativa frente
  a la primera repetición anterior: máximo `1.1764e-7` con el cambio aplicado,
  frente a `9.4928e-8` entre repeticiones anteriores. Se exige menos de `1e-6`.
- Se comparan los campos fluctuantes completos, antes de añadir la carga
  macroscópica, no sólo Ceff. La diferencia L2 máxima del cambio aplicado es
  `1.1375e-5`; al repetir el anterior ya se observa `1.1350e-5`.
- La diferencia máxima puntual normalizada por el máximo del campo de
  referencia es `5.4532e-6` con el cambio y `6.1747e-6` entre dos ejecuciones
  anteriores. No es un error relativo punto a punto donde el campo se anula.
- La equivalencia observada no es identidad binaria ni una certificación de
  tensiones máximas para todos los materiales. Se conserva el problema discreto.

Las pruebas unitarias cubren cero RHS, solución inicial exacta, arranque
no nulo, límite de iteraciones, callback que modifica el vector inicial,
conteos de llamadas, preservación del RHS, NumPy/CuPy y cargas agrupadas con
una RHS ya convergida. Las pruebas existentes de RFFT y composite voxels
también se ejecutaron: 26 pruebas aprobadas.

```bash
OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_cg_redundant_work.py tests/test_rfft_unscaled_cuda.py \
  tests/test_composite_voxels.py
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/profile_g09_cg_cleanup.py --repeats 5
```

Datos: `results/g09_cg_cleanup.json`, con tiempos individuales, Ceff, conteos,
auditorías, material y hashes del código actual utilizado.

## Dónde seguir

El ahorro es acotado: sólo se elimina trabajo de inicio y de salida, no las
360 aplicaciones iterativas. El siguiente objetivo es el coste por iteración
(FFT y tráfico de campos CG), no la construcción material. Antes de activar
otra implementación de reducciones o fusionar más kernels, exigir A/B de
tiempo total, residuo verdadero del operador original y comparación de campos.
`check_true_residual` sigue siendo una auditoría optativa; este benchmark exige
explícitamente su aprobación, pero no modifica la política de producción.
