# Artículos para reducir iteraciones de CuFHom

Revisión: 29–30 de septiembre de 2026. Alcance: elasticidad lineal periódica,
GPU, G09 original 240³, seis cargas, aceptación física relativa `1e-5`.

## Conclusión para nuestro caso

Conservar CG en coordenadas simétricas como base. La siguiente investigación
con más interés es una **corrección global de Galerkin con una base construida
desde el operador y subdominios de la malla**, acompañada de un precondicionador
local. Debe servir para materiales y microestructuras generales: no requiere
identificar fibras, su eje ni su relación de aspecto. Es una propuesta de
adaptación pendiente: no hay una reducción nueva de CG demostrada en G09.

Las fusiones integradas en esta sesión reducen el tiempo por iteración; las
seis cargas siguen sumando 288 CG. No atribuir ese ahorro a los artículos ni
confundir una mejora contra CG sin precondicionar con una mejora contra CuFHom.

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

Mi valoración: encaja mejor con un solver general que una base elegida por
tipo de inclusión. Es una referencia de diseño, no una implementación lista
para CuFHom: nuestra representación espectral tiene acoplamiento global y
requiere estudiar operadores locales, coste de autovalores y memoria. No
trasladar automáticamente las cotas FEM al operador FFT.

## Adaptación propuesta, todavía sin implementar

Nuestro operador en coordenadas es `H = R Q* C Q R`, SPD sobre los modos
admisibles; `R` es la raíz inversa de la referencia isotrópica. Para una base
`U` de rango completo, construida desde subdominios periódicos y el operador:

```
Hc = U* H U
B  = I + U Hc^-1 U*
```

Aquí `*` es el adjunto para el producto real de las coordenadas RFFT. `Hc`
debe ser SPD y el segundo término es semidefinido positivo, por lo que `B`
es SPD. Es un primer ensayo aditivo de dos niveles propuesto para CuFHom;
no reproduce todavía todos los resolutores locales de Schwarz del artículo.
El operador fino `H` y la aceptación física permanecen completos.

Pasos y criterios concretos para una futura implementación:

1. Partir de bloques de la malla con solapamiento, independientes del tipo de
   inclusión. Comparar funciones locales de orden bajo con modos espectrales
   seleccionados por el operador. Convertirlos al espacio de coordenadas
   existente, retirando dependencias,
   modos medios y planos de Nyquist. No confundir un campo de desplazamiento
   real con el vector de coordenadas blanqueadas del solver.
2. Usar `H` real para formar `Hc`; comprobar simetría, rango y Cholesky en
   doble precisión. Una matriz constitutiva media por bloque no reemplaza esto.
3. Medir coste de construcción, aplicación y memoria antes de G09. Una base
   densa de 32 vectores complejos G09 ocupa aproximadamente **5,35 GB**;
   `HU` duplica ese almacenamiento si se retiene. Hace falta una representación
   compacta o una base mucho más pequeña para esta GPU de 16 GB.
4. Exigir un coste total menor en las seis cargas, incluidos construcción y
   auditorías. Con un sobrecoste ilustrativo de 20 % por iteración y sin
   preparación, hay que bajar de 288 a menos de **240 CG** para ganar tiempo.
   Es una cuenta de rentabilidad, no una predicción.
5. Mantener el mismo residuo físico `1e-5`, comparar CPU64 en casos pequeños,
   campos completos y `Ceff`, y medir al menos cinco repeticiones alternadas.
   Validar laminados, inclusiones compactas, campos anisótropos y distintos
   contrastes, además de G09. No fijar parámetros por nombre de caso.

Los Ritz históricos de G09 abarcan aproximadamente 1,07–71,2 y sugieren un
espectro amplio; no son cotas certificadas. El reciclaje de solo 1–4 modos
no redujo el coste en sus ensayos. Eso favorece estudiar bases que dependan
del operador completo, aunque no garantiza éxito.
[Diagnóstico histórico](../descartados/evidencia/g09_krylov_spectrum_20260922.json),
[reciclaje medido](../descartados/evidencia/g09_linear_recycling_confirm_20260924/report.json).

Ninguna de estas propuestas pendientes se añadió como ruta del solver.
