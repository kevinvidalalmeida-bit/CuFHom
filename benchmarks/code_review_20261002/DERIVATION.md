# Separar la referencia homogénea de H

## Identidad en el espacio que usa CuFHom

Sean `Q₀` el mapa compatible isométrico y `M₀ = Q₀* C₀ Q₀` la rigidez de
la referencia isotrópica. El mapa actual es `Q_R = Q₀ M₀^(-1/2)`. Por tanto,
en las coordenadas admisibles:

\[
Q_R^*C_0Q_R=I,\qquad
H=Q_R^*CQ_R=I+Q_R^*(C-C_0)Q_R.
\]

El cero de frecuencia y los planos Nyquist excluidos siguen excluidos.
Sobre todo el almacenamiento, el primer término sería el proyector sobre
esas coordenadas, no una identidad sin restricciones. La norma incorpora
la normalización y los pesos de la transformada real que ya usa el mapa.
La prueba CPU64 independiente construye los 372 grados reales sin media
de una malla impar 5³ y comprueba la igualdad con error relativo <4e-16.

La referencia debe coincidir con la que blanquea `Q_R`. No basta con que
`material_table[0]` contenga cualquier tensor isotrópico. El piloto comprueba
sus 21 coeficientes contra los módulos derivados de `Q_R`.

## Qué se ahorra y qué permanece

Para una dirección admisible `p`, con `ε = Q_R p`:

\[
p^*Hp=\|p\|^2+\langle\varepsilon,(C-C_0)\varepsilon\rangle.
\]

El piloto calcula la norma en FP64 dentro del mapa `p → ε`. En cada voxel
de referencia, escribe seis ceros de tensión de contraste y evita leer las
seis deformaciones y multiplicarlas por el tensor. En otros voxels aplica
el tensor completo de contraste, incluyendo acoplamientos anisótropos.
El adjunto/update añade `p` analíticamente al resultado del contraste.

Las seis transformadas de cada sentido, los buffers densos de tensión y
el número de aplicaciones de H permanecen. La norma añadida aumenta
instrucciones y parciales de reducción en el mapa. No se mide una reducción
de bytes DRAM en este ensayo: los contadores anteriores no corresponden a
estos kernels nuevos.

**La matriz matemática, su espectro y su condición no cambian.** Esta
identidad puede reducir coste por aplicación; no constituye un nuevo
precondicionador ni explica una reducción de iteraciones. Los tiempos
completos y los campos están en [RESULTS.md](RESULTS.md).

## Precisión cuando hay fases blandas

La resta conserva la identidad en aritmética exacta aunque `C-C₀` sea
indefinida. H sigue siendo SPD cuando C lo es. Sin embargo, para una fase
homogénea `C = η C₀`:

\[
p^*Hp=\eta\|p\|^2
=\|p\|^2+(\eta-1)\|p\|^2.
\]

Si `η` es pequeño, dos términos grandes se cancelan. Sumar sus productos
en FP64 no recupera los bits perdidos al formar `C-C₀`, transformar ε o
aplicar el tensor en FP32. El error relativo puede crecer aproximadamente
como `u/η`, donde `u` describe el redondeo de esas operaciones. El cribado
GPU compara ambos denominadores contra `η ||p||²`, calculado de forma
independiente en CPU64. Esto exige una protección adicional antes de
ofrecer el split como política general de producción. Las auditorías
verdaderas permanecen activas, pero no justifican introducir esa pérdida
de precisión cuando existe una ruta estable.

## Variante de norma recurrente

En CG ordinario exacto, `r_k` es ortogonal a `p_(k-1)`. Luego:

\[
\|p_k\|^2=\|r_k+\beta_kp_{k-1}\|^2
=\|r_k\|^2+\beta_k^2\|p_{k-1}\|^2.
\]

Esto permite evitar el producto añadido al mapa. El piloto reinicia la
norma cuando el CG reemplaza el residuo por `b-Hx`; conserva el suavizado
tardío y el H original para todos los controles verdaderos. En FP32 se
omiten el término cruzado de la ortogonalidad perdida y el redondeo del
update de `p`. Es una aproximación de implementación, aunque la fórmula
sea exacta en CG ideal. La variante no supera al cálculo directo en las
dos mallas grandes originales; queda retirada como opción predeterminada.
No se atribuye a PIPECG: aquí no se implementó ese algoritmo.

## Cribado de un precondicionador distinto

También se evaluó, por separado, la compliance fina proyectada:

\[
B_c=M_0^{1/2}Q_0^*C^{-1}Q_0M_0^{1/2}.
\]

Sea el bloque de C en coordenadas compatibles/incompatibles
`[[A₀, E], [E*, D]]`. Su inversa restringida es
`Q₀* C⁻¹ Q₀ = (A₀ - E D⁻¹ E*)⁻¹`. En cambio, resolver el problema requiere
`A₀⁻¹`. **Invertir C antes de proyectarlo no invierte la restricción de C.**
La compliance proyectada es SPD y domina `A₀⁻¹`; de ahí que el espectro
precondicionado tenga mínimo ≥1. Ello no garantiza agrupamiento favorable
ni menos CG para las seis cargas concretas.

El cribado CPU64 obtiene el espectro completo de cinco casos 5³ y resuelve
sus seis cargas a `1e-8`. La condición mejora, pero las iteraciones apenas
bajan o suben. Cada aplicación del precondicionador necesita otra pareja
de transformadas y un tensor inverso. El cociente `2 n_PCG / n_CG` sirve
solo como aproximación con costes H/B iguales, no como tiempo GPU. Por
esa evidencia esta implementación no tiene prioridad sobre la base actual.

## Reutilización de la auditoría final

La comprobación verdadera dentro de CG calcula `Q_R z` y `C Q_R z` antes
de evaluar `b-Hz`. El adaptador vuelve a calcular ambas cantidades para
construir el campo final y comprobar `B-GN(C campo)`. Se añadió una acción
específica para esos controles verdaderos; retiene ambos campos y el
iterado al que pertenecen. La acción normal y las acciones de dirección
invalidan el registro antes de continuar.

Al devolver CG su iterado, la reutilización requiere identidad con el
iterado retenido y ausencia de campo inicial añadido. La proyección
preparada comprueba además que material y GN sean exactamente los factores
del operador físico. Si no coinciden, aplica el operador completo.
Una corrección por falsa convergencia vuelve a evaluar la ecuación original.

La ruta preparada comparte las operaciones FFT, la proyección y su escala
con el operador original. Conserva el residual físico, su suma FP64, la
tolerancia y el máximo de iteraciones. Evita una transformación inversa
de seis componentes y un producto constitutivo por carga reutilizada.
Todavía vuelve a transformar la tensión para proyectarla: no se conserva
su espectro y no se afirma haber eliminado esa transformada.

La nueva opción es `linear_final_audit_reuse`; se comparan ambas políticas
con el mismo código y se comprueba igualdad bit a bit de los campos y de
los residuales. Los tiempos y la decisión están en [RESULTS.md](RESULTS.md).

## Otros cambios posibles, aún sin medir

1. **CG con escalares FP64 residentes en GPU.** Mantener las recurrencias
   actuales y leer estado/convergencia por grupos de iteraciones. Resulta
   especialmente relevante en mallas pequeñas, según la traza nueva.
   Los controles de positividad y los residuales verdaderos deben seguir
   siendo parte del algoritmo.
2. **Reutilizar el espectro de la tensión final.** Podría evitar otra
   transformada R2C, pero requiere extender las interfaces y respetar los
   buffers que cuFFT puede consumir. No se implementó esa segunda extensión.
3. **Ceff a partir de tensiones medias.** En equilibrio,
   `Ceff_ij = <(C ε_j)_i> = <ε_i : C ε_j>`. El residuo finito puede hacer
   diferir ambas expresiones. Se comprobó solo en los cinco problemas
   CPU64; no se sustituyó la matriz de energía de producción.

El [CG de una reducción de PETSc](https://petsc.org/main/manualpages/KSP/KSPCGUseSingleReduction/)
emplea dos vectores extra en esa implementación. No se presupone que
compense en una sola GPU. Cools, Cornelis y Vanroose analizan las pérdidas
de estabilidad de recurrencias encadenadas y una variante estable de
[CG con comunicación solapada](https://arxiv.org/abs/1902.03100); sus ventajas
en memoria distribuida no demuestran una ganancia para CuFHom.

Los callbacks FFT podrían evitar buffers intermedios, pero necesitan un
diseño que respete acoplamientos de seis componentes y su orden de ejecución.
La [documentación cuFFT 12.8](https://docs.nvidia.com/cuda/archive/12.8.1/cufft/index.html)
advierte que un load callback C2R puede invocarse más de una vez por elemento.
Un update de dirección o una suma con efectos laterales dentro de ese callback
no sería válido sin resolver esa multiplicidad. Esta ruta no se implementó.
