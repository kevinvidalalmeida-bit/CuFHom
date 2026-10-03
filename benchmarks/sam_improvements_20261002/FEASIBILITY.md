# Viabilidad del cubo para Vf = 60 %, AR = 30 e isotropía

El cubo periódico tiene capacidad para superar 60 % con cilindros rectos de
las dimensiones requeridas. Se demostró mediante una construcción paralela
y una auditoría independiente. **Todavía no se ha demostrado esa fracción
junto con isotropía global A4 en nuestro cubo exacto.** El ejemplo publicado
para un cubo cercano respalda investigar ese objetivo; no sustituye una
geometría certificada con nuestros parámetros.

El último checkpoint usa diámetro `d = 1`, longitud `L = 30`, lado `B = 60`
y separación mínima `g = 0,001`. Las unidades pueden ser µm: las conclusiones
son invariantes al escalado uniforme. La campaña previa comprueba `B = 2L`
en [`run_curves.py`](../vf_ar_curves_20261002/run_curves.py). Las fibras pueden
cruzar las caras del cubo: las caras opuestas se identifican periódicamente.

| Comprobación | Resultado | Alcance |
| --- | ---: | --- |
| Volumen del cubo | 216.000 d³ | Geometría continua |
| Volumen de un cilindro | 23,56194 d³ | `π d² L / 4` |
| Conteo más cercano a 60 % | 5.500 | Vf = 59,99569 % |
| Conteo que supera exactamente 60 % | 5.501 | Vf = 60,00660 % |
| Volumen de 5.500 cápsulas con diámetro `d + g` | 61,45299 % | Condición necesaria; no prueba empaquetamiento |
| Distancia garantizada del eje a sus propias imágenes | ≥30 d | Para cualquier orientación; supera 1,001 d |
| Configuración paralela auditada | 5.501 fibras, 60,00660 % | Sin solapes; incumple isotropía |
| A2/A4 realizables con 5.500 direcciones sin posiciones | Errores 4,45e-16 / 1,98e-14 | No prueba empaquetamiento simultáneo |

Para la cota de autoimagen, toda traslación periódica no nula tiene norma
al menos B. Dos puntos del mismo segmento difieren como máximo L. La
desigualdad triangular da `distancia entre segmentos ≥ B − L = 30`.
El tamaño no elimina ninguna orientación individual por autoimagen.
Aumentar el grid no cambia esta geometría continua; en 256³ el diámetro
ocupa 4,267 vóxeles.

La construcción de capacidad utiliza la dirección `(1,1,1)/√3`. La proyección
de `B Z³` sobre el plano perpendicular genera una red triangular con separación
básica `B√(2/3)`. Su subdivisión en 48 partes produce 48² líneas periódicas
distintas, separadas al menos 1,02062 d. Cada línea tiene periodo `B√3`
y admite tres segmentos separados axialmente 34,64102 d, mayor que la
longitud total de cápsula con separación, 31,001 d. La red completa admite
6.912 fibras y 75,39822 % cilíndrico. Se extrajo un subconjunto de 5.501
y se auditaron todos los pares e imágenes, incluidas autoimágenes:
**cero violaciones incluso con tolerancia cero**.

[`capacity_only_parallel.npz`](capacity_only_parallel.npz) lleva explícitamente
`isotropic_packing_accepted=False`; sus errores A2/A4 son √2 y 2. No es un
resultado del objetivo isotrópico. La comprobación separada de A2/A4 utiliza
cuatro anillos con `E[z²] = 1/3` y `E[z⁴] = 1/5`. No asigna posiciones ni
representa aleatoriedad uniforme de todos los órdenes. Dos comprobaciones
separadas no prueban su compatibilidad conjunta.

La diferencia central con el benchmark es la definición de isotropía.
[Schneider (2017)](https://doi.org/10.1007/s00466-016-1350-7), introducción,
ecuación (1), describe la escala asintótica `Vf × AR ≈ c`, con c entre
4 y 5,6, para empaquetamientos aleatorios de varillas. Para AR30 da
13,33–18,67 %. **Es una escala de modelos y observaciones, no un teorema
que prohíba superar 18,67 %.** El autor explica que controlar A4 global
permite salir de ese régimen. En §4.2 describe haces locales paralelos
que, combinados, conservan los momentos globales prescritos.

La [ficha primaria de Fraunhofer](https://publica.fraunhofer.de/entities/publication/5554dfd0-672f-41a9-aac7-812f542d24cb)
confirma 60 % para orientación isotrópica y AR33. El texto completo local
`/home/user/Downloads/schneider2016 (1).pdf`, §3.4.2 y figura 7, precisa:

| Parámetro | Nuestro ensayo | Caso publicado de 60 % |
| --- | ---: | ---: |
| AR | 30 | 33 |
| B/d | 60 | 70 |
| L/B | 0,50000 | 0,47143 |
| Separación/d | 0,001 | 0,001 |
| Cilindros objetivo | 5.500 | 7.940 |
| Orientación comprobada | A2 y A4 global | A4 global |

El lado relativo a la longitud es 6,06 % mayor en el ejemplo publicado.
No se puede trasladar el resultado por escalado uniforme: AR y B/L difieren.
El 60 % con haces locales es plausible en nuestro régimen; su cumplimiento
en B/d = 60 permanece abierto. El caso publicado con separación/d = 0,2
alcanzó 47,46 % tras 24 horas. No se da un tiempo separado para el 60 %.

Nuestro checkpoint tiene 1.559 fibras, Vf nominal = 17,00605 % y error A4
relativo = 2,17e-8. Para distancia entre ejes ≤1,01 d, sus 6.619 pares tienen
media `P2 = (3 cos²θ − 1)/2 = −0,08407`; solo 2,93 % están separados
por menos de 20° de orientación. Esta medida no muestra alineación local
predominante. La selección por proximidad introduce sesgo angular; no debe
compararse directamente con pares independientes ni tomarse como prueba de
ausencia de todos los haces. El 17 % es compatible con atasco de varillas
desordenadas, pero no certifica que el algoritmo esté implementado correctamente.

Se encontró y corrigió un error concreto en
[`SAMLiteGenerator.rotational_scale`](../../src/cufhom/pipeline/sam_generator.py):

```
epsilon = d² [AR³/12 + AR²/6 + 3AR/16 + 1/15] / (AR + 2/3)
```

La expresión anterior multiplicaba por AR y sumaba 2/3 fuera del cociente.
Para AR30/d1 daba 72.171,4167 en lugar de 78,44647: factor 920,0085.
Las rutas con `tau_rotate/epsilon` fijo rotaban demasiado poco. FIRE y parte
de la compactación usan ganancias proporcionales a epsilon que cancelan
esa diferencia cuando domina esa rama. No corresponde anunciar una mejora
global de velocidad o empaquetamiento de 920 veces.

El ensayo A4 de 17 % **ya usaba la fórmula correcta**, así que este error
no explica por sí solo su atasco. La ruta histórica también exige A2;
A2 isotrópico no garantiza A4 isotrópico. Las cápsulas añaden 2,222 % por
fibra respecto al cilindro a AR30, diferencia insuficiente para explicar
la brecha de 17 a 60 %.

Se verificó visualmente una inconsistencia en el artículo: ecuación (8)
usa `epsilon/8 ||A4 − A4_N||²`, con A4_N normalizado por N en (3), pero
el campo de orientación de (9) omite el factor 1/N de esa derivada.
Nuestra ruta A4 usa `N epsilon/8 ||A4 − A4_N||²`: su derivada y métrica
de inercia reproducen el campo de orientación de (9). Las diferencias
finitas validan esa elección consistente; no prueban qué normalización
usó el C++ original ni hacen de L-BFGS una reproducción literal del descenso.
No se cambió el peso basándose únicamente en esa inconsistencia editorial.

El procedimiento publicado permite 500.000 pasos antes de reducir en una
fibra el lote de adición. Nuestros ensayos generalmente usan límites menores
y reducciones más rápidas. El intento con límite 500.000 se interrumpió
por tiempo antes de completar esa relajación. Reducir lotes pronto cambia
la trayectoria de reorganización. El control pendiente debe reproducir
AR33/B70 y esa política, y después comparar AR30/B63,636 y AR30/B60.
El segundo cubo conserva B/L publicado; el tercero es nuestro objetivo.
Eso permitirá distinguir fidelidad del algoritmo y efecto del tamaño.
Ninguno de esos controles está certificado al 60 % aquí.

La corrección pasó seis pruebas nuevas: integración volumétrica independiente
para cuatro tamaños, derivadas de las fuerzas de contacto a AR30 y conservación
de sus rotaciones con ganancia FIRE compensada. La batería SAM pasó
**94 pruebas y 7 subpruebas**. Datos, certificados y correlaciones están en
[`feasibility.json`](feasibility.json); los resultados históricos se conservan.

Para reproducir:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  benchmarks/sam_improvements_20261002/check_feasibility.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_sam*.py
```
