# Rutas retiradas de CuFHom

Estas carpetas conservan evidencia y código histórico. No se instalan como
paquetes ni se importan desde `src/`, los comandos o los benchmarks activos.
Las pruebas normales se limitan a `tests/`.

## Qué se retiró el 29 de septiembre de 2026

- **Métrica de referencia en seis componentes:** se movieron
  `reference_elasticity.py` y `cupy_reference_metric.py` desde el núcleo, y se
  eliminaron sus conexiones al proyector, CG, aplicación y Sobol. El
  [parche](metrica_referencia/retirada.patch) conserva las conexiones retiradas.
  En G09 había bajado 360 a 300 CG, pero subido **4,825 a 5,188 s**.
  `cupy_matrix_reference=True` y `cupy_reference_eta != 1` ahora dan un error
  explícito. Sustitución: configuración GPU predeterminada,
  `linear_formulation='auto'`, `linear_preconditioning='symmetric_fused'`.
- **Especialización del kernel por dimensiones:** en la campaña actual dio
  **3,163 frente a 3,167 s**, una diferencia demasiado pequeña para conservar
  otra implementación. La combinación con `p.Ap` tampoco aportó una mejora
  suficiente sobre la fusión sola.
- **Parches temporales de las fusiones:** quedan en
  [lineal_20260929](lineal_20260929/). Las dos fusiones aceptadas tienen ahora
  interfaces explícitas en `linear_krylov.py`; el solver no ejecuta estos parches.
- Los **82 archivos del antiguo `archive/`** se trasladaron, sin cambiar sus
  bytes, a [historico_20260924](historico_20260924/). El
  [manifiesto SHA256](manifest.json) registra cada origen y destino. Incluyen
  prototipos fallidos y versiones antiguas de mejoras que sí se integraron;
  retirar una implementación duplicada no descarta su método matemático.

## Decisiones anteriores que se mantienen

Los tiempos siguientes pertenecen a sus campañas históricas; cada comparación
usa la referencia de esa campaña, no los tiempos de la implementación actual.

| Ensayo | Evidencia | Decisión |
| --- | --- | --- |
| Polarización con eliminación de matriz y corrección local | 3,594 s frente a 3,418 s del combinado | Retirado; las FFT densas y correcciones anulan el ahorro de incógnitas |
| Referencia anisótropa promedio | 4,231 s frente a 3,794 s | Retirado para G09; no generalizar a toda referencia anisótropa |
| Bloques espectrales aprendidos | Mejor mezcla 3,432 s frente a 3,424 s; entrenamiento adicional 2,496 s | Retirado; este ensayo no reproduce todo UNO-CG |
| Reciclaje Galerkin entre seis cargas | 3,427 s frente a 3,168 s; ambos 288 CG | Retirado; construir la base no se amortiza |
| Reciclaje espectral | 3,842 s y 292 CG frente a 3,168 s y 288 CG | Retirado |
| Bloques locales Voigt/Reuss | Cribado de 16 configuraciones sin ganar tiempo | Retirado; fue una aproximación constitutiva local, no un Schwarz completo |
| CUDA Graphs con copia de entrada | 8,528 frente a 8,068 ms por operador | Retirado; medición de operador congelado, no de seis cargas |
| Prototipo híbrido cuFFTDx | ≈21,2 frente a ≈8,1 ms por operador | Retirado; transformadas XY y presión de registros demasiado caras |
| Parada basada solo en energía efectiva | Residuo físico ≈4e-4 en la prueba CPU | Retirado del solver estricto que exige 1e-5 |

Los [JSON de evidencia](evidencia/) son copias pequeñas del proyecto original,
con [hashes y procedencia](evidencia/manifest.json). Los informes completos están
en `historico_20260924/benchmarks/`; conservan sus referencias históricas y no
todos los artefactos grandes citados se copiaron.

## Ensayos del posprocesado, 30 de septiembre

La reducción genérica CuPy con productos float64 elevó el total **3,071 a
5,635 s** y se retiró. La reducción CUDA por warps ya disponible en el núcleo
sí mejoró el tiempo y se integró; su prototipo temporal también se archivó.
[Ensayos y evidencia](posprocesado_20260930/README.md).

## Formulación e iteraciones, 30 de septiembre

Ensayos sobre el mismo G09 240³ y seis cargas, siempre con residuo físico 1e-5.
Los cribados tienen una medición tras calentamiento; la variante aceptada se
confirmó con cinco repeticiones alternadas y pruebas adicionales.

| Ensayo | Resultado frente a su base | Estado |
| --- | --- | --- |
| [Galerkin con 18 modos Fourier](dos_niveles_20260930/README.md) | Bloque: 4,557 s / 387 CG; balanceado: 6,410 s / 294; base 3,037 s / 288 | Descartados; no equivale a GenEO |
| [CG por bloques](cg_bloques_20260930/README.md) | 10,343 frente a 3,046 s; 288 aplicaciones en ambos | Descartado |
| [Residuo conjugado](residuo_conjugado_20260930/README.md) | 276 iteraciones, pero 3,169 frente a 3,034 s | Descartado |
| [Suavizado continuo](suavizado_continuo_20260930/README.md) | 3,088 frente a 3,032 s | Descartado |
| [Suavizado fusionado continuo](suavizado_fusionado_20260930/README.md) | 3,047 frente a 3,034 s; cribado sin ganancia clara | Sustituido por activación tardía |
| [Suavizado tardío](suavizado_integrado_20260930/README.md) | Prototipo: 3,036 → 2,948 s; 288 → 276 CG | Método integrado; solo se archiva el prototipo duplicado |
| [Reajuste isotrópico](referencia_isotropica_20260930/README.md) | K/G actual ya minimiza aproximadamente la cota material examinada | Sin cambio de referencia |

[Validación y tiempos del código integrado](../benchmarks/g09_smoothing_20260930/RESULTS.md).
El [manifiesto de esta campaña](manifest_formulaciones_20260930.json) registra
los prototipos, scripts e informes conservados. Los benchmarks archivados
fijan el suavizado nuevo a `False` para conservar su referencia histórica.

## Rutas que siguen activas

CPU/float64, formulación en deformaciones, comprobación del residuo verdadero,
corrección de falsa convergencia y el PCG de referencia en desplazamientos
siguen siendo útiles para precisión, compatibilidad y validación.
`linear_preconditioning='matrix'` es ese PCG de referencia; es distinto de
la métrica retirada `cupy_matrix_reference`.

Los scripts históricos usan nombres/rutas antiguos y no se presentan como
portados. Para medir el solver actual, usar
[`benchmarks/profile_g09_linear.py`](../benchmarks/profile_g09_linear.py).
