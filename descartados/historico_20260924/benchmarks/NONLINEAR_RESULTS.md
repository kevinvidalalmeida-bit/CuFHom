# Verificación no lineal y curva G09 — 2026-09-19

## Actualización 2026-09-21

La fusión espectral y la tolerancia interna adaptativa de Newton reducen la
curva G09 240³ a **104.659 s**, frente a 234.907 s al repetir la ruta anterior
(**55.45 % menos**), con 40 incrementos sin rechazos. Malla,
materiales, trayectoria y tolerancias finales intactos; pasan 94 pruebas.
Las mediciones históricas y las variantes retiradas se consolidan en
[`../ESTRATEGIAS_DESCARTADAS.md`](../ESTRATEGIAS_DESCARTADAS.md).
Las secciones siguientes conservan las mediciones históricas.

## Actualización: CG variacional y curva recalculada

La nueva [curva G09 240³](results/nonlinear_g09_galerkin_loading/curve.png),
carga monotónica 0 → 1 %, se calculó en **237.015 s**, frente a 404.777 s al
repetir la ruta anterior: **41.45 % menos tiempo**. Ambas tienen 40 incrementos
y cero rechazos. Se conserva malla, material e incremento; CG evita una proyección
FFT redundante por iteración y fusiona actualizaciones GPU. Las repeticiones
históricas y la verificación de equivalencia se registran en
[`../ESTRATEGIAS_DESCARTADAS.md`](../ESTRATEGIAS_DESCARTADAS.md).
La batería actual pasa **75 pruebas**; la nueva ruta también supera la
validación sintética CPU/GPU y de medio incremento, registrada en ese informe.

Lo que sigue documenta los resultados anteriores conservados; los 765 s del
ciclo completo no son un tiempo comparable a 237 s de sólo carga.

## Curva anterior conservada: extracción sin descarga

La API y el runner ahora usan carga monotónica por defecto. La descarga queda
disponible con `--unload` únicamente cuando se pide. Se conserva el ciclo anterior
como verificación y no se alteran sus resultados.

La [curva elastoplástica G09](results/g09_elastoplastic_loading/curve.png) y su
[CSV](results/g09_elastoplastic_loading/curve.csv) contienen exactamente los
primeros 41 puntos del ciclo original (estado cero y 40 incrementos de carga).
Es una extracción de resultados ya convergidos, no otra simulación. La procedencia
y hash del archivo fuente están en los metadatos; no se incluye el checkpoint de
fin de descarga como si fuera un estado al terminar la carga.

La rama llega a 48.113 MPa al 1 % de deformación prescrita, con residuo máximo
8.864e-6. El 1 % sigue siendo el límite demostrativo del ensayo, no la rotura.
Los parámetros plásticos todavía requieren calibración para representar un
material real. La batería anterior pasó 60 pruebas, incluida la equivalencia
de la carga por defecto con la rama inicial de un ciclo carga–descarga.

## Qué se resolvió

Pequeñas deformaciones, matriz J2 con endurecimiento isotrópico positivo y
fibras TI elásticas, sobre la geometría binaria G09 original 240³. Control axial
0 → 1 % → 0 con otros cinco esfuerzos medios nulos. Fluencia=15 MPa y H=30 MPa
son **demostrativos**, no calibrados. Se conservan los parámetros elásticos G09.

La curva incluye elasticidad, plastificación progresiva, endurecimiento y
descarga. **No incluye estricción ni rotura.** La bajada es una descarga impuesta,
no una rama de falla. Para predecir estricción se requiere una extensión a
grandes deformaciones y un problema de probeta con pérdida de uniformidad;
la curva macroscópica de esta celda periódica no sustituye ese análisis.

## G09 240³: ejecución completa Newton

GPU RTX 5070 Ti, CuPy float32, incremento axial 0.00025, rtol=1e-5.

| Medida | Resultado |
|---|---:|
| Incrementos confirmados | 80 |
| Incrementos rechazados | 0 |
| Tiempo global | 765.306 s (12 min 45 s) |
| Tiempo sumado de intentos exitosos | 757.705 s |
| Iteraciones CG totales | 25,447 |
| Residuo físico normalizado máximo | 8.864e-6 |
| Residuo macroscópico máximo | 5.489e-8 |
| Error Hill–Mandel máximo | 4.603e-8 |
| Esfuerzo axial al 1 % | 48.113 MPa |
| Esfuerzo axial al terminar en deformación cero | -25.514 MPa |
| Deformación al cruzar esfuerzo cero en descarga | ≈0.5614 % |

El último valor se interpola entre puntos de la descarga, no es otro solve.
La reserva del pool CUDA al final fue 6,649,650,176 bytes; no es un pico exacto
de memoria exclusiva. El tiempo incluye preparación y checkpoints, excluye
la exportación de la gráfica. Es una ejecución, no una mediana ni un speedup
comparativo; durante parte de ella se ejecutaron pruebas pequeñas CPU/GPU.

Resultados: [curva](results/nonlinear_g09_optimized/curve.png),
[CSV](results/nonlinear_g09_optimized/curve.csv),
[metadatos](results/nonlinear_g09_optimized/metadata.json).
El NPZ adyacente permite reiniciar desde el estado confirmado final.

Esta corrida comenzó antes de incorporar PTC/Euler opcionales y no capturó un
hash de fuentes al inicio. No atribuir sus tiempos a PTC. Una reexportación
posterior sólo añadió el reloj de trayectoria y colores de carga/descarga;
no cambió puntos de la solución. Las ejecuciones nuevas registran hashes.

## Verificación independiente

Inclusión elipsoidal sintética 12³, misma geometría en todas las variantes,
80 incrementos base y 160 al dividirlos por dos. Escala de comparación 15 MPa.

| Comparación | Diferencia máxima / escala, en % | Límite |
|---|---:|---:|
| Curva GPU32 frente a CPU64 | 0.000160 % | 0.1 % |
| Curva CPU64 con medio incremento | 0.008651 % | 0.1 % |

Residuos máximos: CPU64 8.748e-9; GPU32 9.940e-6;
medio incremento CPU64 7.950e-9. Estos resultados pasan los umbrales fijados.
Registro completo: `results/nonlinear_validation/validation.json`.

**Pendiente:** convergencia de incremento sobre G09 240³ y comparación completa
PTC/Newton en esa geometría. La verificación sintética no certifica esos errores
en G09. Tampoco se ha certificado precisión de campos máximos o convergencia espacial.

## Convergencia y pruebas

La batería inicial incluyó 59 pruebas: material/tangente, proyección, régimen elástico frente al
solver existente, descarga, CPU/GPU, reinicio, rechazo/rollback, PTC y Euler
explícito opcional. Incluyen pérdida de curvatura de CG, fallo de Newton seguido
de PTC y rechazo de falsa convergencia cuando el pseudopaso es diminuto.

PTC se contrasta con Newton en casos pequeños tanto CPU como GPU y conserva
la curva dentro del umbral 0.1 %. El arranque Euler explícito también pasa esa
comparación, pero no se afirma que sea más rápido. Newton permanece como primer
intento en `globalization='auto'`; PTC se activa si ese intento falla.

Un diagnóstico inicial G09 detectó deriva de CG al espacio nulo y reducciones
escalares costosas. Reproyectar residuos y usar reducciones por warps corrigió
la convergencia sin relajar la tolerancia. El diagnóstico de un primer incremento
completo terminó sin subdivisión con residuo 1.15e-7. No confundir sus tiempos
con los de una curva completa.

## Limpieza realizada

Se eliminaron definitivamente, sin respaldo solicitado: `superbody/` (77
archivos, aproximadamente 2.5 MiB), los módulos octree/coarsening y sus pruebas,
benchmarks, informes, seis JSON de coarsening y cachés identificadas. Se conservaron
la geometría/materiales G09, la ruta lineal, composite voxels y sus pruebas,
los resultados de rendimiento FFT y todos los cambios ajenos preexistentes.
