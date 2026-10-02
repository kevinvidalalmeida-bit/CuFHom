# Implementación de las estrategias lineales

## Alcance acordado

Implementar y medir las cuatro rutas propuestas, incluida una adaptación propia
de GenEO al operador de CuFHom. La referencia inicial es el commit `0cc0d47`.
Los experimentos conservan campos y FFT FP32, reducciones FP64, las seis cargas,
el residual físico verdadero `rtol=1e-5` y las geometrías originales.

- [x] A: selección persistente de política FFT por configuración; comparar lotes 1, 2, 3 y 6.
- [x] B: energía en material y adjunto/update fusionados, eliminando el vector intermedio Hp.
- [x] C: espacio auxiliar elástico con aplicación multigrid lineal, simétrica y positiva.
- [x] D: selección espectral local GenEO en el espacio auxiliar y transferencia al espacio admisible fino.
- [x] Validación CPU64 independiente, residual, seis campos y Ceff de cada ruta.
- [x] Comparaciones alternadas en G08/G09 y casos generales, con preparación/memoria incluidas.
- [x] Integrar lo que mejore el tiempo; archivar lo descartado con su evidencia.
- [x] Informe reproducible y actualización de la [PR privada #1](https://github.com/kevinvidalalmeida-bit/CuFHom/pull/1).

## Controles

Una reducción de iteraciones no basta: se mide preparación, aplicación del
precondicionador, auditorías y correcciones. Se mide la selección inicial del
autotuner y el solve con caché; la amortización incluye ese coste. No se afirma
haber medido un proceso completamente frío con contexto y JIT. Los candidatos se eligen
por forma, material/operador y coste, sin reglas por nombre de geometría ni fibras.
Los contadores anteriores orientan el trabajo; los tiempos nuevos se miden
sin Nsight Compute y dentro de la misma sesión que su referencia.

## Derivaciones y controles aplicados

La fusión B usa la isometría RFFT existente:
`Re(p* H p) = sum_voxels (QRp) : C(QRp)`.
Se mueve el denominador de alpha al kernel de material. Después del R2C se
actualizan x/r al aplicar RQ*, sin materializar Hp. No se altera la recurrencia
de CG; el orden de redondeo sí puede variar y se verifica en FP32.

Para C/D se construye una transferencia T con adjunto real exacto entre el
espacio auxiliar y las coordenadas RFFT admisibles. Con S fijo y SPD:
`B = eta I + T S T*`, eta positivo, es SPD. El operador fino H se conserva.
Las cotas de GenEO para matrices dispersas no se atribuyen automáticamente
a H: se prueba la selección local y se mide su efecto en el espectro fino.
El suavizado necesita una derivación en la métrica B de PCG y aceptación con
el residual verdadero original.

Fuentes: [GenEO](https://arxiv.org/html/2104.00280v2),
[cuFFT](https://docs.nvidia.com/cuda/cufft/index.html),
[espacio auxiliar en elasticidad](https://www.math.uci.edu/~chenlong/Papers/ChenHuHuang2017SolverElasticity.pdf).

## Resultado

A/B quedan por defecto en la ruta compatible. C/D se implementaron y validaron,
pero mantienen las iteraciones y aumentan el tiempo; quedan retirados en
`descartados/auxiliary_geneo_20261002`. GenEO no queda como tarea pendiente.

[Resultados y evidencia reproducible](RESULTS.md), [derivación](DERIVATION.md).
La suite de producción pasa 338 pruebas y 7 subpruebas; los controles aislados
del auxiliar pasan 4 pruebas. Los tiempos finales cubren diez casos y cinco
repeticiones alternadas para A/B; C/D usan tres repeticiones alternadas.
