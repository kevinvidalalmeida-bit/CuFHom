# Artículos para reducir iteraciones de CuFHom

Revisión: 29–30 de septiembre de 2026. Alcance: elasticidad lineal periódica,
GPU, G09 original 240³, seis cargas, aceptación física relativa `1e-5`.

## Conclusión para nuestro caso

Conservar CG en coordenadas simétricas como base. La siguiente investigación
con más interés es una **corrección global de Galerkin con una base construida
desde el operador y subdominios de la malla**, acompañada de un precondicionador
local. Debe servir para materiales y microestructuras generales: no requiere
identificar fibras, su eje ni su relación de aspecto. Esa adaptación sigue
pendiente. Los modos Fourier bajos ya se probaron y no mejoraron G09.

Las fusiones iniciales redujeron el tiempo por iteración con 288 CG. Después,
el **suavizado tardío de residuos** redujo las seis cargas a **276 CG** y el
tiempo un **2,82 % adicional** en una comparación alternada de la implementación
final. Se conserva la misma aceptación física 1e-5.
[Mediciones y derivación](../benchmarks/g09_smoothing_20260930/RESULTS.md).
No confundir mejoras contra CG sin precondicionar con mejoras contra CuFHom.

## Qué aportan las fuentes consultadas

### 1. Schwarz de dos niveles con agrupaciones de nodos — prioridad de investigación

Krysl combina subdominios solapados con un problema global reducido construido
por Galerkin a partir de bases polinomiales locales. Sus ejemplos incluyen
fibras en matrices compresibles y casi incompresibles. Consulté la ficha
editorial, el preprint del autor y el código asociado; la publicación usa
elementos finitos, por lo que requiere una adaptación a nuestra discretización.
[Artículo, 2025](https://doi.org/10.1016/j.cma.2024.117558),
[preprint del autor](https://ssrn.com/abstract=4902156).

El código de `TwoLevelPreConditioner` forma y factoriza el operador reducido
`Phi' K Phi`, combina su solución con resoluciones locales solapadas y está
implementado en Julia/FinEtools. No ofrece una sustitución directa para CuPy.
[Implementación del autor](https://github.com/PetrKryslUCSD/FinEtoolsDDMethods.jl/blob/main/src/DDCoNCSeqModule.jl).

**Mi valoración:** esta vía introduce acoplamiento global entre regiones.
El ensayo local Voigt/Reuss que descartamos no construía ese problema global;
su fracaso no demuestra que Schwarz de dos niveles vaya a fracasar. Aun así,
construir/factorizar bases y transferir campos puede consumir todo el ahorro.

### 2. UNO-CG — útil para muchas consultas, ventaja pendiente frente a nuestra base

Herb y Fritzen construyen un precondicionador aprendido lineal, simétrico y
positivo definido. Revisé la parametrización, entrenamiento y tiempos del
preprint. En su tabla 6, mecánica 3D periódica sobre H100 a residuo `1e-6`,
UNO-CG tarda **3.222,4 ms**, FANS **1.300,8 ms** y CG sin precondicionar
**68.129,1 ms**. La tabla 5 emplea 1.000 microestructuras 192³ de entrenamiento
y 400 de prueba en 3D. Estos tiempos no son comparables directamente con G09.
[Preprint, §§4 y 6, apéndices D–E](https://arxiv.org/html/2508.02681v2),
[publicación de 2026](https://doi.org/10.1002/nme.70277).

El repositorio se basa en PyTorch y FEM regular; está orientado a muchas
consultas. [Código oficial](https://github.com/DataAnalyticsEngineering/UNOCG).
Mi valoración: interesa para campañas de materiales y geometrías, con coste
de entrenamiento amortizado. El ajuste espectral pequeño que ya probamos
no aceleró G09 y no equivale a reproducir UNO-CG completo.

### 3. J-FFT / Green–Jacobi — baja prioridad para G09 binario

Ladecký y colaboradores combinan escalado Jacobi y Green. En §4.4 comparan
interfaces suaves y abruptas: Green–Jacobi gana en las suaves, mientras que
Green gana en las abruptas del experimento. La formulación es FEM regular.
Mi valoración: tiene más interés con propiedades que varían suavemente,
algunas tangentes no lineales o campos de densidad. Nuestro salto resina/fibra
y anisotropía no permiten extrapolar una aceleración.
[Preprint actualizado, §§4.4 y 6](https://arxiv.org/html/2508.02613v2).

### 4. A²EM — reservar para poros y contrastes extremos

Dolbeau, Bleyer y Sab actualizan el material de referencia en el esquema
Adaptive Eyre–Milton. El trabajo aborda medios con poros e inclusiones rígidas
y muestra casos de matriz/fibra/poro. Revisé el artículo completo.
Mi valoración: puede ser valioso si añadimos vacíos o fases casi rígidas;
el artículo no constituye una medición contra nuestro CG simétrico en G09.
No se debe cambiar el precondicionador dentro de CG clásico copiando esa
actualización: A²EM utiliza su propia iteración.
[Artículo, 2024](https://comptes-rendus.academie-sciences.fr/mecanique/item/10.5802/crmeca.269.pdf).

### 5. RPM — no priorizar para los seis RHS actuales

Peng, Nepal y Dayal identifican un subespacio de iteraciones inestables o
lentas y aplican Newton sobre él, conservando iteraciones de punto fijo en
su complemento. El estudio se orienta especialmente a alto contraste y
vacíos. Mi valoración: sus comparaciones no prueban superar nuestra base CG;
almacenar y construir subespacios ya resultó costoso en nuestros ensayos.
[Artículo/preprint, 2020](https://arxiv.org/abs/2003.07720).

### 6. Referencia homogénea y CG — fundamento que ya aprovechamos

Ladecký y colaboradores explican la referencia homogénea por bloques en
Fourier y las cotas espectrales basadas en propiedades materiales. Su
formulación FEM difiere de la espectral de CuFHom. Mi valoración: respalda
estudiar el espectro y precondicionar simétricamente; llamar “óptima” a la
referencia respecto a la malla no significa eliminar la dependencia del
contraste. [Artículo/preprint](https://arxiv.org/html/2203.02962v1).

El estudio comparativo de Vondřejc y colaboradores encontró CG más eficiente
que Chebyshev, Eyre–Milton y Richardson en sus problemas elípticos 2D.
Mi valoración: sustituir CG por otro nombre de algoritmo, sin mejorar el
operador precondicionado, no tiene una ventaja demostrada para G09.
[Comparación de solvers](https://arxiv.org/html/1508.02045).

### 7. Espacios espectrales GenEO — criterio general para construir la base

Spillane y colaboradores construyen el espacio global reducido mediante
problemas de autovalores generalizados locales, buscando los modos que frenan
la convergencia en medios heterogéneos. El artículo de 2011 presenta un
resultado variacional y ejemplos de elasticidad; la construcción depende del
operador, las particiones y el solapamiento. Revisé el artículo completo.
[Artículo original](https://comptes-rendus.academie-sciences.fr/mathematique/item/10.1016/j.crma.2011.10.021.pdf).

Se implementó una adaptación propia al espacio espectral de CuFHom y se
midieron sus costes, campos y espectro en problemas pequeños. Las variantes
ensayadas no aceleran el solve: **GenEO queda descartado de la ruta activa**.
El artículo se conserva como referencia; sus cotas FEM no se atribuyen al
operador FFT. La derivación y el rechazo se documentan al final de esta revisión.

### 8. CG por bloques, CR y suavizado — ensayos ejecutados

El CG por bloques sigue la recurrencia alternativa de
[Dubrulle, 2001](https://etna.ricam.oeaw.ac.at/vol.12.2001/pp216-233.dir/pp216-233.pdf),
con normalización espectral propia. En G09 no redujo las aplicaciones del
operador y elevó el tiempo a 10,343 s: prototipo retirado.

CR usa la recurrencia de
[Saad, §6.8](https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf).
Bajó 288 a 276 iteraciones, pero subió 3,034 a 3,169 s por el coste de vectores.
La minimización del residuo por sí sola no garantizó una mejora de tiempo.

[Zhou y Walker, 1994](https://users.wpi.edu/~walker/Papers/residual_smoothing,SISC_15,1994,297-312.pdf)
estudian combinar aproximaciones para suavizar residuos y discuten los errores
de las recurrencias. Aprovechamos la ortogonalidad de CG para calcular el peso
mediante normas ya disponibles. Activarlo cerca de la tolerancia y fusionar
su actualización son decisiones de implementación de CuFHom. La aceptación
recalcula el residuo; la estimación armónica no se toma como cota en float32.

La corrección de Galerkin balanceada ensayada sigue
[Nabben y Vuik, ecuación 4](https://diamhomes.ewi.tudelft.nl/~kvuik/papers/Nab04V.pdf).
Los 18 modos Fourier bajos elegidos para el piloto aumentaron tiempo e
iteraciones. Esta base no reproduce la selección de autovectores locales GenEO.
[Código y evidencia de los pilotos](../descartados/README.md).

## Adaptación con subdominios: implementada y descartada

Actualización: 2 de octubre de 2026. Se implementaron un espacio auxiliar
elástico Q1, transferencia periódica con adjunto real, V-cycle simétrico,
selección local GenEO y PCG con suavizado en la métrica del precondicionador.
Los bloques se construyen por malla y operador, sin reglas para fibras.
También se ensayó la energía fina real: para `U = T Z`,

```
Hc = U* H U
B  = I + U Hc^+ U*
```

Aquí `*` es el adjunto para el producto real de las coordenadas RFFT;
`Hc^+` elimina las dependencias de la base mediante su descomposición espectral.
El operador sigue siendo `H = R Q* C Q R`. La construcción garantiza B SPD,
pero esa propiedad no garantiza reducir el número de iteraciones.
Se evita almacenar una base densa en la malla fina y se conserva el criterio
de aceptación con residual físico verdadero `1e-5`.

### Resultado en G09

Tres repeticiones alternadas por variante, seis cargas y preparación incluida
una vez por material. Cada fila usa su propia referencia contemporánea.

| Variante | Referencia s | Variante s | Preparación s | Iteraciones CG |
| --- | ---: | ---: | ---: | ---: |
| Auxiliar Q1 + V-cycle | 2,967 | 5,249 | 0,247 | 276 |
| GenEO auxiliar, τ=0,15 | 2,967 | 5,540 | 0,533 | 276 |
| GenEO con energía fina, τ=0,15 | 2,963 | 7,685 | 2,722 | 276 |
| GenEO con energía fina, τ=0,5 | 2,965 | 9,401 | 4,431 | 276 |

Ampliar el umbral a 0,5 selecciona 285 columnas, incluidas 123 deformables;
quedan 282 direcciones independientes en la energía fina. Su preparación
requiere 285 aplicaciones de H y tampoco reduce las 276 iteraciones.
En los problemas CPU64 pequeños, el auxiliar empeora el condicionamiento;
usar la energía fina lo aproxima al de la referencia, sin una mejora útil
de CG. Estos resultados corresponden a las bases y transferencias ensayadas.

**Decisión: retirar auxiliar Q1 y GenEO de producción.** El código y sus
controles quedan en [descartados](../descartados/auxiliary_geneo_20261002/README.md).
La [derivación propia](../benchmarks/linear_strategy_20261002/DERIVATION.md)
documenta transferencia, autoproblemas locales, Galerkin fino y suavizado PCG.
El [informe reproducible](../benchmarks/linear_strategy_20261002/RESULTS.md)
incluye tiempos, campos, residual, espectro CPU64 y procedencia del código.

Se integran por defecto la selección persistente de planes FFT y la fusión
de energía/adjunto/update: reducen el coste de aplicar H y conservan las
iteraciones. GenEO deja de figurar como una implementación pendiente.
