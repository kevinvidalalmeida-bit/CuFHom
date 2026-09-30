# Precondicionamiento espacial: pruebas G09 240³

## Decisión

**No integrar la candidata por bloques.** Ninguna de las 16 configuraciones
ensayadas mejora el tiempo del PCG actual. Algunas reducen CG en el estado
elástico, pero su aplicación cuesta demasiado; en plasticidad mantienen o
aumentan las iteraciones. Se conservan todos los defaults de producción.

No se cambió la geometría, la malla física, las propiedades, el retorno J2,
las condiciones mixtas ni los 40 incrementos uniformes de 0 a 1 %.
Material demostrativo, no calibrado experimentalmente. RTX 5070 Ti 16 GB,
CuPy float32 con productos/acumulaciones de reducción en float64.

## Formulación realmente probada

Para el operador fino `A = Q* C_alg Q + shift I`, se ensayó

    B = B0 + weight Q* R* D R Q.

`B0` es el precondicionador homogéneo de producción. `R` restringe campos
Mandel a bloques con normalización por la raíz del número de puntos incluidos.
La pareja implementación media/prolongación constante equivale a `R* D R`;
se verifica también para bloques incompletos y soportes vacíos.
`D` es una matriz simétrica positiva 6×6 por bloque:

- Voigt: inversa de la tangente media sobre su soporte.
- Reuss: media de la inversa de la tangente sobre su soporte.

La construcción usa `a`, `b` y la dirección plástica locales, conserva la
anisotropía de las fibras cuando están incluidas y añade correctamente PTC.
La inversa J2 usa Sherman–Morrison, incluyendo los pequeños errores de norma
y traza de la dirección almacenada. Cholesky en doble precisión comprueba
la positividad de cada bloque. No se almacena una tangente 6×6 por vóxel.

Se probaron bloques de 4³, 8³ y 16³; pesos 0,1 y 1 sobre matriz; además,
bloques de 8³ y peso 0,1 sobre todas las fases o sólo matriz plastificada,
con ambas medias. Son **bloques del precondicionador**, no coarsening físico.

Esta candidata no es un Schwarz de rigideces locales completas ni la inversa
de un operador Galerkin acoplado. Aproxima el bloque constitutivo y después
lo proyecta al espacio compatible. No se debe generalizar su resultado a
todos los métodos de dos niveles o espacios auxiliares.
La familia matemática ya existe: [Tang et al., 2009](https://doi.org/10.1007/s10915-009-9272-6).

## Protocolo y resultados

Se congelaron operador, RHS y tangente **dentro de Newton reales** de una
trayectoria original G09, en los pasos 1, 12 y 36: deformaciones axiales
0,025 %, 0,30 % y 0,90 %. Cada candidata empieza con corrección cero.
Después del cribado, únicamente el PCG original avanza el estado físico.

Las tolerancias internas son las capturadas del Newton correspondiente,
no el `rtol` exterior. El primer Newton usa 0,1; los posteriores muestreados
usan 0,0253181 y 0,0537813. **La aceptación no lineal permanece en 1e-5.**
Los residuos verdaderos y la compatibilidad se comprueban independientemente;
RHS y todas las matrices/campos de respuesta se contrastan exactamente antes
y después. El mayor defecto relativo de compatibilidad observado fue 1,16e-6.

### Confirmación con reducción rápida: tres repeticiones

Referencia antes/después y orden rotado de seis candidatas. Ejemplo fijo:
matriz, Voigt, bloques 8³, peso 1. Medianas de segundos por **sistema congelado**:

| Estado, primer Newton | PCG actual: CG / s | Bloques: CG / s |
| --- | ---: | ---: |
| Elástico, paso 1 | 10 / 0,11565 | 8 / 0,19825 |
| Transición, paso 12 | 37 / 0,40146 | 38 / 0,82313 |
| Plástico avanzado, paso 36 | 38 / 0,41283 | 38 / 0,82189 |

La mejor candidata entre las seis en transición/plasticidad fue Voigt matriz,
16³, peso 0,1: 0,76076 / 0,78111 s, sin reducir CG. Incluso eliminando
**todo** su coste de construcción, el solve tarda 0,75385 / 0,77448 s:
reutilizar los bloques entre Newton no puede compensar esta diferencia.

### Newton posteriores: las 16 variantes con reducción rápida

Una corrida por variante, referencias antes/después; no se presenta como
otra mediana de tres repeticiones. Para la misma candidata fija de 8³:

| Sistema | PCG actual: CG / s | Bloques: CG / s |
| --- | ---: | ---: |
| Paso 12, Newton 3 | 90 / 0,96355 | 92 / 1,94981 |
| Paso 36, Newton 2 | 63 / 0,67626 | 62 / 1,32357 |

Ninguna gana. Las mejores por tiempo entre las 16 tardan 1,81454 y 1,27633 s.
Algunas elecciones fuertes resultan mucho peores: 339 CG frente a 90 en el
Newton 3 del paso 12. Todos los solves terminaron dentro del límite de 15 s;
no hubo fallos de positividad ni aceptación sólo por tamaño de incremento.

En total: **150 solves candidatos medidos y 30 referencias**, aparte de
calentamiento, auditorías y la trayectoria original que genera los sistemas.
Se recorrió tres veces la trayectoria de 40 pasos sin rechazos. Sus seis
esfuerzos medios coinciden con la referencia previa, pero esas trayectorias
**no son curvas resueltas por la candidata**. Sus tiempos incluyen cribados
y no son nuevas mediciones de latencia de producción.

## Perfil del sobrecoste y corrección de bajo nivel

El prototipo inicial usaba una reducción genérica de CuPy para `r·corrección`:
30,24 ms por llamada, frente a 0,47–0,53 ms del kernel warp ya utilizado por
producción. Diferencia entre resultados escalares: 2,63e-15 relativa.
Se sustituyó en la candidata, conservando ambas opciones reproducibles.
Esto acelera **el prototipo**, no supone una mejora nueva del solver actual.

Corrección espacial de bloque 8³, por aplicación, cinco llamadas perfiladas:

| Componente adicional | Intervalo aproximado, ms |
| --- | ---: |
| FFT directa/inversa | 5,006 |
| Mapas de coordenadas | 1,404 |
| Restricción y matriz de bloque | 1,668 |
| Prolongación | 0,515 |
| Total sin instrumentar, antes/después | 9,268 / 9,131 |

Son intervalos CUDA con posibles huecos de lanzamiento, no contadores de
tiempo activo puro. La corrección cuesta aproximadamente tanto como otra
aplicación del operador. El descenso de CG observado no lo compensa.
Reserva máxima observada del pool: 7,001 GB; **no es una medición de pico de
memoria viva** y el harness conserva copias adicionales para la auditoría.

La explicación matemática plausible es que invertir tangentes locales y
proyectar no equivale a invertir `Q* C_alg Q`: siguen faltando los acoplamientos
compatibles entre bloques. Es una interpretación del resultado, no una
demostración espectral. No se implementó actualización/reutilización entre
Newton ni se ejecutó una curva G09 con la candidata porque falla el filtro
de tiempo incluso suponiendo construcción gratuita.

## Validación, fuentes y reproducción

**351 pruebas aprobadas**, 7,85 s, sin skips; único aviso previo de `numpy.matlib`.
Las 38 pruebas nuevas cubren ensamblado diminuto independiente, inversas,
bloques parciales, soportes, simetría, positividad, compatibilidad, PTC,
inmutabilidad, carga/descarga, campos y estado plástico, ambos reductores
y restauración ante excepciones. Los ocho hashes de producción registrados
coinciden con los archivos actuales. Sin cambios en defaults ni código del solver.

- [Cribado inicial, 16 variantes y reducción genérica](results/g09_spatial_screen_20260924/report.json).
- [Confirmación, seis variantes, tres repeticiones y reducción warp](results/g09_spatial_warp_confirm_20260924/report.json).
- [Newton posteriores, 16 variantes y reducción warp](results/g09_spatial_late_screen_20260924/report.json).
- [Perfil de componentes y comparación de reducciones](results/g09_spatial_components_20260924.json).
- [Regresión completa JUnit](results/g09_spatial_validation_20260924/pytest.xml).

Los primeros dos harnesses contaron la construcción de bloques, pero no la
pequeña preparación del motor PCG experimental; el cribado posterior sí
incluye ambas. Esta omisión favorecía la candidata y no cambia su rechazo.
Los hashes en cada informe identifican las revisiones usadas. El código final
permite reproducir las dos rutas mediante `--reduction generic|warp`.

```bash
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_spatial_preconditioner.py \
  --output /tmp/g09-spatial-new --repeats 3 \
  --variants matrix_voigt_b8_w1 matrix_voigt_b16_w0.1 matrix_reuss_b8_w0.1 \
  --reference FFT/benchmarks/results/g09_lowlevel_curves_confirm_20260924/baseline_2/metadata.json

OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_spatial_preconditioner.py \
  --output /tmp/g09-spatial-late-new --late-newton

OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

Conclusión limitada a lo medido: descartar **esta corrección aditiva de
compliance por bloques** como aceleración de G09. La curva de producción
conserva su camino anterior; no atribuirle los ahorros internos del prototipo.
