# Reciclaje lineal y corrección energética: pruebas del 24-09-2026

## Decisión

**Conservar el solver lineal actual.** Ninguna de las diez configuraciones de
reciclaje ensayadas acelera las seis cargas de G09 240³. La señal favorable
observada anteriormente en geometrías CPU diminutas no se reproduce aquí.
No se modificó producción, su tolerancia, la geometría, materiales, malla ni
el solver no lineal. Los prototipos están aislados en `benchmarks/`.

La cota energética pasa las comprobaciones CPU64, pero su criterio de parada
para obtener sólo el tensor NO cumple el residuo físico actual. No se activa
como default ni se anuncia como aceleración equivalente de G09.

## Confirmación G09: tres repeticiones por variante

RTX 5070 Ti 16 GB, CuPy 13.6.0, campos float32 y productos/reducciones FP64;
`displacement`, `symmetric_fused`, RFFT por componente automática, seis cargas
secuenciales, `rtol=1e-5`. Orden rotativo: cada variante ocupa cada posición
una vez. Geometría precompilada y planes/kernels calientes.
El tiempo incluye captura y construcción de la base, productos AU, solución,
correcciones, auditorías y escritura de metadatos de tiempos. No incluye
lectura/compilación de geometría, exportación de campos ni el informe comparativo.

| Variante | Mediana, s | CG totales | Cambio de tiempo |
| --- | ---: | ---: | ---: |
| Producción actual | **3,168433** | **288** | Referencia |
| Arranque Galerkin, ventana 12 / rango 4 | 3,427202 | 288 | +8,17 % |
| Corrección espectral, ventana 20 / rango 1 | 3,842230 | 292 | +21,27 % |

Tiempos individuales de referencia: 3,168433 / 3,168974 / 3,166670 s;
arranque: 3,439947 / 3,427202 / 3,424998 s; espectral:
3,842230 / 3,847246 / 3,841503 s. No se compara un tiempo nuevo contra
el mínimo de una sesión anterior para atribuir una aceleración.

Las nueve resoluciones medidas cumplen las auditorías originales. Diferencia
relativa máxima del tensor frente a producción: **6,929e-8**; residuo físico
máximo: **9,340e-6**. No es una comparación G09 CPU64 ni error de discretización.
[Informe completo](results/g09_linear_recycling_confirm_20260924/report.json).

## Formulación implementada

En las coordenadas simétricas de producción se conserva exactamente la acción
`A = T* C T`. Se aprende U sólo del primer problema y se fija para las cinco
cargas restantes; no se reutiliza entre operadores/materiales distintos.
Con G = U*U y H = U*AU se prueban:

1. Arranque Galerkin: `z0 = U H^-1 U* b`, seguido de CG para la corrección.
   El umbral pertenece al RHS original, nunca al RHS reducido del arranque.
2. Corrección espectral SPD: `B = I + U (H^-1 - G^-1) U*`.
   Una proyección y una combinación de campos por aplicación.
3. Corrección balanceada SPD:
   `B = U H^-1 U* + (I-U H^-1 (AU)*)(I-AU H^-1 U*)`.
   Dos proyecciones y dos combinaciones; AU se conserva en memoria.

Las fórmulas usan el producto REAL de las coordenadas RFFT complejas, no
coeficientes complejos arbitrarios que saldrían del subespacio admisible.
Los kernels de productos y combinaciones acumulan en doble precisión.
Cholesky comprueba las matrices pequeñas SPD. La base se reduce por rango
numérico cuando aparecen dependencias; su dimensión no altera los DOF físicos.

Se capturan ventanas de 8, 12, 16 y 20 vectores sin almacenar todo Krylov.
La variante `rolling` comprime y conserva direcciones Ritz, admitiendo nuevas
direcciones durante toda la primera carga. No es una reproducción completa
de eigCG ni garantiza autovectores convergidos. El AU final se recalcula con
el operador real, no se acepta sólo una aproximación proyectada.
Fundamento de reutilización espectral: [Stathopoulos y Orginos](https://arxiv.org/abs/0707.0131).

## Cribados: una repetición medida por configuración

Estos valores son pilotos, separados de las medianas confirmadas anteriores.
En el primer grupo la referencia fue 3,164078 s / 288 CG; en el segundo,
3,160109 s / 288 CG. Cada grupo tuvo también una ejecución de calentamiento.

| Configuración | Tiempo, s | CG |
| --- | ---: | ---: |
| Espectral, ventana 8 / rango 2 | 4,027 | 307 |
| Espectral, ventana 12 / rango 4 | 4,481 | 312 |
| Espectral, ventana 16 / rango 4 | 4,407 | 301 |
| Balanceada, ventana 12 / rango 4 | 5,001 | 292 |
| Arranque, ventana 12 / rango 4 | 3,411 | 288 |
| Espectral, ventana 20 / rango 1 | 3,830 | 292 |
| Espectral rolling, ventana 12 / rango 2 | 4,413 | 301 |
| Espectral rolling, ventana 12 / rango 4 | 5,287 | 324 |
| Balanceada rolling, ventana 12 / rango 4 | 5,578 | 288 |
| Arranque rolling, ventana 12 / rango 4 | 4,056 | 288 |

[Cribado inicial](results/g09_linear_recycling_screen_20260924/report.json),
[segundo cribado](results/g09_linear_recycling_refined_20260924/report.json).
Todos pasaron el criterio físico, incluido el CG corrector cuando fue necesario.
En total hay **16 solves candidatos y 5 referencias medidos**, más 15 solves
de calentamiento; cada solve contiene las seis cargas completas.

## Dónde se pierde el tiempo

- Arranque confirmado: captura + construcción cuesta aproximadamente **0,246 s**
  y las seis cargas siguen necesitando 288 CG. El coste no se amortiza.
- Espectral de rango 1: aproximadamente **0,381 s** de captura + construcción,
  además del coste de aplicar la corrección y cuatro CG adicionales.
- Rolling de rango 4: alrededor de **1,17 s** sólo en captura + construcción.
  La variante balanceada mantiene 288 CG y añade trabajo en cada iteración.

Son intervalos host con sincronizaciones, no un perfil de tiempo activo CUDA.
El defecto relativo del subespacio invariante es aproximadamente 2,16 % para
el modo de ventana 20, y 3,51–13,20 % para la base rolling de rango 4. Son
modos aproximados; no demuestran que se hayan eliminado los autovalores
extremos de todo el operador. Más refinamiento no produjo una base suficiente
para reducir CG en estas pruebas. No se generaliza a todo método de deflación.

Cada vector complejo G09 ocupa 167.270.400 bytes. La ventana 20 almacena
3,345 GB y la pareja U/AU de rango 4, 1,338 GB. Reserva máxima observada
del pool tras un solve en los cribados: 6,993 GB; **no es el pico de memoria
viva**. Los costes de leer vectores grandes importan aunque H sea diminuta.

## Cota energética: prueba separada, no cambio de aceptación

Para soluciones compatibles y el residuo verdadero R = B-AZ, en aritmética
exacta y restringiendo A al subespacio admisible:

    C_hat - C_eff,h = R* A^-1 R / N >= 0.
    C_hat - C_eff,h <= R*R / (N alpha),  si A >= alpha I.

El código ya calcula C_hat mediante productos energéticos. Aquí se añade la
comprobación de la identidad y una cota escalar por la traza. Alpha se obtiene
de las rigideces LOCALES generalizadas respecto a la matriz, con margen
numérico; nunca se usa el mínimo Ritz de la base como cota inferior global.
Esto es una cota algebraica del error iterativo DISCRETO. La verificación CPU
no es un certificado formal de redondeo por intervalos ni una cota de malla.
Antecedente variacional: [Vondřejc, Zeman y Marek](https://arxiv.org/abs/1404.3614).

Se compararon geometrías sintéticas 9³ y 10×8×12, CPU64, semilla 248,
con los parámetros del material G09. La referencia independiente se resuelve
con CG a 1e-12. El criterio tensor-only exige una cota de error en norma
operador <= 1e-4 E_m: escala E_m=4,338421, umbral absoluto 0,0004338421,
en las mismas unidades de rigidez de entrada. No se cambia ningún parámetro
material para ajustar la respuesta.

| Caso | CG con auditoría física 1e-5 | CG tensor-only | Residuo físico tensor-only | Error medido del tensor / cota por traza |
| --- | ---: | ---: | ---: | ---: |
| 9³ | 225 | 145 | 3,995e-4 | 9,110e-6 / 2,669e-4 |
| 10×8×12 | 210 | 129 | 4,021e-4 | 8,170e-6 / 2,574e-4 |

Se reduce CG, pero se relaja expresamente el criterio físico: **no equivale
al solver estricto ni autoriza sustituirlo**. No se midió este modo en G09/GPU.
Las diferencias de la identidad energética son menores de 4e-15 en estos
casos. La corrección `Zc=Z+U H^-1 U*R` reproduce
`C_hat,c=C_hat-(U*R)* H^-1 (U*R)/N` a menos de 2e-15 y mantiene la cota
superior, salvo redondeo. Su mejora del error del tensor es pequeña; también
se recalcula la cota del residuo corregido en lugar de asumir que disminuye.
[Informe energético completo](results/linear_energy_certificate_20260924.json).

## Validación y reproducción

**390 pruebas aprobadas**, 8,64 s, sin skips ni fallos; sólo el aviso previo
de `numpy.matlib`. Hay 39 pruebas nuevas: SPD, rango deficiente, ventanas
acotadas/reinicio, productos GPU frente a CPU64, no mutación, energía y cotas,
compatibilidad y medias, mallas pares/impares/no cúbicas, seis cargas CPU/GPU,
RHS nulo, límites de iteraciones y restauración de parches ante excepciones.
La suite también mantiene las pruebas lineales, no lineales y composite_voxels.
[JUnit](results/g09_linear_recycling_validation_20260924/pytest.xml).

Los ocho archivos de producción registrados conservan sus hashes; también
se contrastaron al terminar los hashes de phase.npy, ori.npy y material.json.
Los informes identifican las revisiones de prototipo de cada cribado; el primero
precede la adición del refinamiento rolling y del diagnóstico espectral.

```bash
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_linear_recycling.py \
  --output /tmp/g09-recycling-new --repeats 3 \
  --variants baseline seed_w12_r4 spectral_w20_r1

OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_linear_energy_certificate.py \
  --output /tmp/linear-energy-new.json

OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

No se borró código ni resultados preexistentes. No hay una aceleración nueva
que integrar: el resultado útil es descartar estas variantes para G09 y dejar
verificada, por separado, la alternativa de precisión específica del tensor.
