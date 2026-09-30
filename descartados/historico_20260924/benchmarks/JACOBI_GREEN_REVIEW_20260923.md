# J-FFT / Green–Jacobi: utilidad para nuestro solver

Artículo solicitado: Ladecký et al., **Jacobi-accelerated FFT-based solver for
smooth high-contrast data**, DOI
[10.1016/j.cma.2026.119269](https://www.sciencedirect.com/science/article/pii/S0045782526005426).
La revista lo asigna al volumen 461, artículo 119269, noviembre de 2026.
Se consultaron el contenido editorial indexado (§§3.2–3.3, 4.5, 5–6) y el
[preprint abierto](https://arxiv.org/html/2508.02613v1).
La URL con `via=ihub` devolvió 403; la búsqueda por DOI sí expuso texto.
El preprint inicial muestra ensayos lineales; la versión editorial añade
elasticidad no lineal de tipo power-law. No confundirla con validación J2
incremental con variables internas y descarga.

## Qué propone el paper

Para `K δu = f`, sea `D = diag(K)` y `G` la pseudoinversa de una referencia
homogénea, aplicada mediante FFT. El precondicionador es:

    M⁻¹ = D⁻¹ᐟ² G D⁻¹ᐟ²

El escalado simétrico combina información local y global. El artículo encuentra
ventajas con coeficientes suaves y contraste alto, pero Green suele ganar con
pocas fases e interfaces abruptas. J-FFT pierde independencia de malla y
conserva complejidad O(N log N), no O(N). Su extracción de la diagonal en 24
acciones en 3D depende de la localidad FEM. Véanse §§3 y 5–6 del
[artículo](https://doi.org/10.1016/j.cma.2026.119269).

## Evaluación específica del repositorio (inferencia propia)

No es correcto insertar `1/sqrt(C_ii(x))` en el CG actual y llamarlo J-FFT:

- Aquí `H = Q* C_alg Q` opera sobre coordenadas Fourier compatibles y cinco
  modos medios libres. La diagonal constitutiva local `diag(C_alg)` no es
  `diag(K)` del sistema de desplazamientos del artículo.
- La base espectral tiene soporte global. El sondeo por colores basado en
  independencia de nodos FEM no recupera su diagonal mediante los mismos
  24 productos. No debemos copiar ese coste de construcción.
- La multiplicación por un coeficiente variable en espacio real mezcla
  frecuencias. No equivale a dividir componente a componente las coordenadas
  Fourier; puede reintroducir modos medios/Nyquist que nuestra formulación
  elimina. El precondicionador debe definirse sobre el subespacio admisible.
- G09 lineal tiene saltos matriz–fibra y orientaciones anisótropas. No hay
  evidencia para reemplazar por defecto la referencia actual. En el J2,
  `C_alg(x)` cambia con plastificación: la idea puede ser más interesante,
  pero las fronteras de plastificación y las fibras siguen introduciendo
  discontinuidades. No supone que aparezca el régimen suave del paper.
- Un precondicionador SPD puede cambiar entre soluciones Newton, pero debe
  permanecer fijo durante un CG estándar. Si cambia dentro, hace falta otro
  método y validación. No se toca el estado plástico confirmado ni se filtra
  el campo material físico para hacer que el benchmark se parezca al paper.

## Cómo lo probaría sin cambiar la solución

1. Congelar tres sistemas reales de G09: elástico, inicio de plastificación y
   plástico avanzado; guardar operador tangente/RHS del mismo Newton.
2. Derivar el cambio de variables entre desplazamientos físicos, coordenadas
   compatibles y modos medios. Construir sólo la diagonal/precondicionador
   sin ensamblar la rigidez global. Alternativa inicial: un aproximante
   positivo obtenido de la tangente; etiquetarlo como **inspirado en J-FFT**,
   no réplica exacta si no usa la diagonal de K.
3. Auditar simetría, positividad, periodicidad, los modos eliminados y los
   esfuerzos medios. La norma para parar sigue siendo el residuo del sistema
   original, no una norma transformada más permisiva.
4. Comparar referencia actual, referencia homogénea que siga la tangente y
   Green–Jacobi. Medir construcción + aplicaciones + CG + auditoría.
5. Sólo si gana en esos sistemas, ejecutar la curva completa y las seis cargas
   lineales, sin cambiar 240³, trayectoria, tolerancias o materiales.

No ejecutado aún: esta revisión no afirma un ahorro de CG por Green–Jacobi.
El ahorro FFT confirmado en [el perfil](FFT_HOTSPOTS_20260923.md) es independiente.

## Umbral económico y orden de ataque

Para un sistema congelado:

    T_nuevo = S + k_nuevo (t_operador + t_PC_nuevo + t_vectores)
    T_actual = k_actual (t_operador + t_PC_actual + t_vectores)

`S` incluye construir el precondicionador. Por ejemplo, con **25 % de coste
adicional por iteración**, haría falta **más de 20 % menos CG**, incluso antes
de amortizar `S`. Es un ejemplo algebraico, no una medición del candidato.
En una curva habría que sumar el coste de cada reconstrucción.

Prioridades a partir de nuestras mediciones:

| Prioridad | Intervención | Estado |
| --- | --- | --- |
| 1 | Coste de FFT, unos 30,2 s antes del cambio | Integrado: 61,20→57,77 s de curva; 3,417→3,164 s lineal |
| 2 | Precondicionador SPD que siga la tangente, con actualización amortizada | Próximo experimento para bajar CG; no promete ganancia |
| 3 | Green–Jacobi adaptado a nuestras coordenadas y condiciones mixtas | Candidato fundamentado por el paper; requiere derivación y auditoría |
| 4 | Fusionar salida de operador con producto pᵀAp; estudiar raíz simétrica no lineal | Puede reducir tráfico/reducciones; preservar norma física y PTC |
| Baja | Integración local J2, 0,322 s en el perfil anterior | Ya paralelizada y compacta; techo de ahorro pequeño |

Los kernels ya paralelizan vóxeles/frecuencias en GPU. Los incrementos
plásticos dependen del estado anterior: no se lanzan como simulaciones
independientes. Más streams tampoco garantiza menor latencia en una GPU
con el ancho de banda ocupado; se justificaría con un perfil y planes sin
workspace concurrente compartido.
