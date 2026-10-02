# Derivación de las cuatro rutas

## Operador y producto interno

Sea (Q_R=QR) el mapa compatible de CuFHom, con la raíz inversa de la
referencia isotrópica incorporada. El operador es

\[
H=Q_R^*C Q_R.
\]

El producto de coordenadas es `Re(vdot(p,z))`. La representación RFFT lleva
peso √2 en las frecuencias interiores del último eje y normalización
`1/sqrt(V)`. Se conservan las convenciones de media y planos Nyquist del
solver. Los campos son FP32 y los productos escalares de Krylov se suman
en FP64. La ecuación física original se comprueba después de resolver.

## A. Selección de FFT

Los lotes 1, 2, 3 y 6 ejecutan el mismo operador `I₆ ⊗ F`; cambia el tamaño del
lote y su espacio de trabajo. Se mide H completo para incluir ambas FFT y sus
interacciones con los mapas/material. CuFHom valida primero la salida de cada
candidato contra la política previa y mide cinco rondas con orden rotado.
La caché incluye forma, dtype, disposición, dispositivo, versión de CuPy/cuFFT,
runtime y controlador. El cambio debe mejorar al menos 3% la mediana del H
medido frente a la política anterior. Este margen es una decisión práctica,
no una prueba estadística de optimalidad.

Esto aproxima la elección por coste del solve: no ejecuta cuatro solves ni
certifica un mínimo global. Las comparaciones alternadas de seis cargas
comprueban el resultado a escala de la aplicación. La política explícita del
usuario tiene prioridad. La selección inicial se cobra una vez y se registra;
la caché permite reutilizarla en otro proceso.

[NVIDIA documenta cuFFT, lotes, planes y espacios de trabajo](https://docs.nvidia.com/cuda/cufft/index.html).

## B. Eliminar el campo Hp

Para una dirección admisible p, sean εp = Q_R p y σp = C εp. Por definición
del adjunto y simetría de C,

\[
\operatorname{Re}(p^*Hp)=\langle\varepsilon_p,\sigma_p\rangle.
\]

El kernel de material ya tiene las seis componentes de εp y σp en registros.
Ahora suma su producto en FP64 mientras escribe σp. Tras reducir ese escalar,
CG conoce `alpha = rho/(p*Hp)`. Se transforma σp y se aplican Q_R*, la
actualización de x/r y la norma residual en un único kernel. Hp solo existe
en registros. El adjunto se genera desde la misma fórmula que usan los mapas
anteriores, incluyendo media/Nyquist y la raíz de referencia.

La recurrencia de CG y su criterio de aceptación se conservan. Cambia el
orden de redondeo del denominador, por lo que la identidad CPU64 por sí sola
no basta: se contrastan campos completos FP32, Ceff y residuales físicos.
Se conserva la combinación tardía de aproximaciones.

En G09, Hp tendría `3 × 240 × 240 × 121` complejos FP32: 167.270.400 bytes.
Se evitan una escritura y una lectura lógicas de ese vector por iteración:
334.540.800 bytes. Esta cuenta no equivale a bytes DRAM medidos ni a tiempo
recuperable exacto. La reducción de energía añade trabajo FP64 al material.
El espectro de σp se libera al terminar su actualización, antes del siguiente H.

## C. Transferencia a un espacio auxiliar

Para k distinto de cero, con n = k/|k|, el mapa compatible puede escribirse

\[
Q_R(k)=iB(n)\left[G_0^{-1/2}P_T+
(K_0+4G_0/3)^{-1/2}P_L\right].
\]

El gradiente simétrico de un desplazamiento u tiene símbolo
`i 2π|k| B(n)`. Por tanto el desplazamiento se eleva a coordenadas mediante

\[
F_T(k)=2\pi|k|\left[\sqrt{G_0}P_T+
\sqrt{K_0+4G_0/3}P_L\right].
\]

Con P una interpolación trilineal periódica, U la RFFT real unitaria y Π la
proyección admisible, se define `T = Π F_T U P`. Su adjunto es
`T* = P* U* F_T Π`. P* se implementa con sumas separables de los mismos pesos,
sin atomics. Las pruebas usan también una matriz P explícita y transformadas
NumPy independientes. Cubren mallas pares/impares y longitudes no cúbicas.

El auxiliar Q1 usa tensores SPD promediados por celdas y ocho puntos de
cuadratura; cada elemento tiene sus seis modos rígidos. Su matriz Kaux se
escala por `V/dominio_volumen`, porque H usa sumas por vóxel y Q1 integrales
físicas. La transferencia no elimina frecuencias del problema fino: solo
construye una corrección precondicionadora.

### V-cycle y positividad

Se quitan las tres traslaciones. El suavizador es
`R = Πmean omega diag(Kaux)^-1 Πmean`, con omega elegido desde una cota de
Gershgorin para mantener `lambda_max(R^(1/2) Kaux R^(1/2)) < 2`.
Un barrido previo, corrección Galerkin y barrido posterior producen

\[
S=(2R-RK_{aux}R)+(I-RK_{aux})P_cS_cP_c^*(I-K_{aux}R).
\]

El primer término es positivo en el espacio sin traslaciones; el segundo es
semidefinido positivo. El solve inferior es una pseudoinversa simétrica con
los modos nulos excluidos. Se usa un número fijo de barridos, sin paradas
adaptativas que harían no lineal el precondicionador.

El ensayo fino usa `Baux = I + T S T*`. Para r distinto de cero,
`r*Baux r = ||r||² + (T*r)*S(T*r) > 0`. Esto demuestra positividad en
aritmética exacta; no demuestra que el espectro de Baux H mejore.
La evidencia CPU64 muestra precisamente que la aproximación Q1 puede
empeorarlo. El coste de transferir campos y ejecutar FFT adicionales también
se mide en GPU.

## D. Adaptación local GenEO

[GenEO selecciona espacios espectrales locales bajo hipótesis explícitas de
descomposición de dominio](https://arxiv.org/html/2104.00280v2). Nuestra
adaptación se realiza sobre el auxiliar Q1 y se eleva a H; sus cotas de
condición no se atribuyen al operador FFT.

Para cada parche periódico:

- KN suma solo los elementos completos dentro del parche y tiene frontera
  natural; incluye los modos rígidos.
- KD es la restricción principal del operador global auxiliar.
- D es la partición de unidad por multiplicidad de nodos. Los parches caben
  estrictamente dentro de cada eje periódico, de modo que KD es SPD.

Se resuelve el problema generalizado

\[
K_Nv=\lambda(DK_DD)v,
\quad
\lambda=\frac{v^*K_Nv}{(Dv)^*K_D(Dv)}.
\]

Los λ pequeños identifican desplazamientos con poca energía interna en
comparación con su extensión ponderada al auxiliar global. Se retienen
`lambda <= tau`, incluidos los rígidos. Se ensayan tau = 0,15 y 0,5.
La base global es `Z = [R_s* D_s v_s]`, representada mediante una matriz
dispersa. Se descartan dependencias y traslaciones en la pseudoinversa de
`Eaux = Z* Kaux Z`. El enriquecimiento `Z Eaux+ Z*` es semidefinido positivo.

También se probó una segunda formulación con energía fina real:

\[
U_f=T Z,\qquad E_f=U_f^*H U_f,\qquad
B_{fine}=I+U_fE_f^+U_f^*.
\]

Cada columna de Ef se obtiene aplicando el H original. Solo se guarda Z
dispersa y la pequeña inversa; no se conserva una colección de campos finos.
Esta variante evita atribuir a H la energía Q1 aproximada, pero añade
aplicaciones de H a la preparación. Se mide su coste en las seis cargas.
Los autoproblemas y la pseudoinversa se resuelven en CPU64; las aplicaciones
auxiliares/transferencias son FP32. Ef se ensambla desde H y productos dispersos
FP32, se comprueba su simetría y se simetriza antes de factorizar.

### Suavizado en PCG

Con B fijo SPD, los residuos de PCG son B-ortogonales en aritmética exacta.
Para combinar el residuo previo suavizado y el nuevo, el peso mínimo en
esa métrica es `omega = estimate/(estimate + rho_new)`, con
`rho_new = r_new* B r_new`; la estimación resultante es `omega*rho_new`.
Se aplica solo cerca de la tolerancia y se comprueba el residuo coordenado
verdadero. La auditoría física posterior mantiene su tolerancia 1e-5.
La estimación armónica no se trata como cota garantizada en FP32.

## Qué permiten concluir las pruebas

A y B mejoran el tiempo completo en las configuraciones medidas y se integran.
C/D conservan el problema físico y pasan los controles de campos, pero los
ensayos no reducen el tiempo; se archivan. La selección inicial tau = 0,15
retiene solo modos rígidos en G09. El umbral ampliado añade modos deformables,
y también se contrasta con la energía fina real. Esto evalúa esas bases y
transferencias concretas, sin descartar toda futura construcción GenEO.
La media por celdas pierde contraste local y la representación de baja
resolución no captura todos los modos que limitan el solve fino.
