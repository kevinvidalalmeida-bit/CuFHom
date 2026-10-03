# Revisión del código de CuFHom: cambios y prioridades

2 de octubre de 2026. Elasticidad lineal; seis cargas; FP32 en campos/FFT,
productos FP64 y residual físico verdadero ≤1e-5. Las
[curvas](RESULTS.md) corresponden al commit 9300f93; la implementación final
se compara por separado contra sus mismas políticas de rotación y auditoría.

## Dos cambios integrados

### 1. Reutilizar el espectro final de tensión

En el operador H = Qᵣ* C Qᵣ, la comprobación verdadera de CG ya calcula
ε = Qᵣ z, σ = C ε y h = Fσ. La auditoría física posterior evalúa la proyección
original Gσ. Reutilizar h elimina otra R2C de seis componentes por carga.
No sustituimos la proyección por una norma aproximada de coordenadas.

La caché pertenece a un único solve y a su iterado exacto. Solo se llena en
comprobaciones verdaderas; las aplicaciones a direcciones, el autotuning y
los reinicios invalidan su contenido. Los campos iniciales añadidos siguen
evaluando el operador completo. El operador exige sus mismos factores A/GN,
RFFT sin escalar, forma, dtype, ejes, contigüidad y dispositivo.
Una entrada incompatible mantiene la evaluación completa.

Qᵣ* lee h. La proyección física genera un nuevo espectro; la C2R consume ese
resultado, preservando h. La referencia a h se libera antes de una eventual
corrección. En 240³ ese espectro ocupa 319 MiB y se retiene durante el cierre
del solve; no se almacena una secuencia de espectros ni una caché global.
No se midió el cambio de pico total de memoria en esta campaña.

La opción linear_final_spectrum_reuse está activa por defecto en la ruta
compatible. False permite comparar manteniendo la reutilización de ε/σ.
linear_final_audit_reuse=False desactiva toda la reutilización final.

Código: [coordenadas](../../src/cufhom/pipeline/linear_krylov.py),
[ciclo del solve](../../src/cufhom/pipeline/linear_coordinates.py) y
[proyección física](../../src/cufhom_core/tensors/operators.py).

### 2. Rotar el material directamente en Mandel

Sea Mα la base ortonormal de tensores simétricos, en orden 00, 11, 22, 12,
02, 01. Las entradas fuera de la diagonal tienen factor 1/√2. Definimos

    Bαβ(R) = Mα : (R Mβ Rᵀ)
    Crot = B(R) C B(R)ᵀ

Para R ortogonal, B también es ortogonal: se conservan simetría,
autovalores y positividad de C en aritmética exacta. La fórmula admite una
matriz anisótropa general; no depende de Vf, AR ni del tipo de inclusión.

Un kernel escribe directamente los 21 coeficientes empaquetados o los 36
de la tabla completa. Sustituye las cuatro contracciones del tensor de cuarto
orden y los lanzamientos individuales de empaquetado. Las operaciones de
rotación usan FP64 y la tabla se convierte una vez al dtype solicitado.
Se conserva el procesamiento por grupos y por chunks. La ruta CPU mantiene
su formulación independiente con tensores de cuarto orden.

Código: [material_rotation.py](../../src/cufhom/pipeline/material_rotation.py).
La rotación anterior vive únicamente en el control del benchmark, obtenida
del commit exacto; no queda una ruta experimental de rotación en producción.

## Resultado y límites

La comparación final incluye setup, seis solves, auditorías, posprocesado y
JSON; cinco pares alternados después de calentamiento. G09 conserva 276 CG.
Las mejoras de medianas son 0,56% en G09, 0,28% en F192 y 1,00% en N256,
con cinco pares favorables en esos tres casos. G08 no demuestra una mejora
total estable. La reducción de setup ronda 0,8 ms en los pilotos pequeños.

La matriz rotada cambia ligeramente por el orden y precisión del cálculo.
Los campos finales se comparan completos; no se presentan como idénticos.
La diferencia L2 máxima en los seis casos grandes es inferior a 5e-6 y la
diferencia relativa máxima de Ceff es inferior a 4e-8. En 8³, comparado con
CPU64 a residual 1e-10, la diferencia L2 es inferior a 4,5e-5.
[Datos finales](production.csv), [campos](production_fields.csv).

**357 pruebas y 7 subpruebas de producción pasan.** Los controles nuevos
comprueban rotación anisótropa completa/sym21 en FP32/FP64; eliminación de seis
llamadas RFFT con full/component/batch2/batch3; formas pares/impares;
rechazo de espectro incompatible y contabilidad de la ruta perfilada.
Los controles previos cubren campos iniciales, materiales cambiados,
iterados distintos, corrección y operador físico incompatible.

## Dónde queda tiempo fuera de H

En Vf 25%, AR 20, 256³, cargar y compilar la geometría consume 0,416 s;
el solve warm completo consume 3,408 s. Esa preparación es independiente del
material y se amortiza con compile_cfield_geometry, que ya está implementado.
No forma parte de las medianas de solve del barrido. La primera llamada de
cada punto se conserva en curves.csv como first_prepared_call_s; ocurre dentro
de una campaña secuencial, no desde un proceso nuevo.

_build_cfield_gpu sigue reconstruyendo el mapa de IDs y transfiriendo
offsets/grupos. Una caché de ese mapa dentro del layout explícito, con
propietario y dispositivo, podría ahorrar parte de los milisegundos de setup.
Debe conservar una tabla nueva para cada material. Es una mejora pendiente
para muchas consultas sobre la misma geometría, con techo pequeño en G09.

compile_cfield_geometry todavía construye índices XYZ y offsets planos.
Evitar o retrasar XYZ en consumidores exclusivamente indexados reduciría
memoria y trabajo de preparación. Requiere mantener compatibilidad con
consumidores densos y layouts antiguos. Tampoco se ha implementado aquí.

## Siguiente ataque para mallas pequeñas: escalares de CG en GPU

material_energy y DeferredHP.update terminan en float(cp.sum(...)).
Cada llamada trae un escalar y espera a la GPU. Son dos sincronizaciones
por iteración de la ruta fusionada. El volumen de datos es pequeño; pesa
la frecuencia de espera. Las opciones lazy/check_every de otras rutas no
eliminan estas conversiones del CG de coordenadas.

La propuesta es mantener ρ, p*Hp, α y β en un estado FP64 del dispositivo:

    αk = ρk / (pk* H pk)
    xk+1 = xk + αk pk
    rk+1 = rk − αk H pk
    βk = ρk+1 / ρk
    pk+1 = rk+1 + βk pk

Sería el mismo CG estándar, consumiendo escalares GPU en las fusiones
existentes y comprobando el estado desde CPU por bloques pequeños.
El dispositivo debe registrar positividad, no finitos y petición de
auditoría; el host debe hacer las comprobaciones verdaderas y los reinicios.
El suavizado tardío también depende de esos escalares.

Está **pendiente de implementación y medición**. No es una nueva prueba del
CG de una reducción de Chronopoulos/Gear: aquel piloto añadió un vector y
empeoró G09. Las capturas de H tampoco sustituyen un ciclo completo de CG
con decisiones y buffers estables. La [traza anterior](../code_review_20261002/RESULTS.md)
motiva esta prioridad; pertenece al código 97939c6, no a esta campaña.

## Siguiente ataque para mallas grandes

1. **Reutilizar modos entre las seis cargas.** H es el mismo para todos los
   RHS. Una base obtenida del Krylov de las primeras cargas podría servir
   para las siguientes. Es distinta de los modos Fourier fijados o las
   bases GenEO/Q1 ensayadas. Hay que medir construcción, aplicaciones de H,
   productos con la base y combinaciones. En 240³ un vector de coordenadas
   ocupa unos 160 MiB; guardar U y HU cuesta unos 319 MiB por columna.
   Un espacio grande puede perder todo el ahorro por tráfico y memoria.
   No se afirma que esta propuesta reduzca CG antes de probarla.
2. **Fusionar energías de posprocesado por bloques.** assembly_matrix en
   modo fused calcula seis tensiones y todavía recorre campos para 36
   productos FP64. Un kernel con pocas filas por bloque podría compartir
   lecturas y conservar la definición variacional de Ceff. Hay que medir
   registros, spills y coste FP64. El posprocesado de las fibras 256³ es
   unos 45 ms frente a 3–4 s totales: incluso eliminarlo completo daría
   solo alrededor de 1,3–1,5% en esos casos.
3. **Reducir H sin repetir fusiones existentes.** Ya se eliminó el buffer
   Hp mediante energía/adjunto/update. Los mapas, material y updates están
   cerca del techo de memoria en la campaña de contadores anterior.
   Un nuevo kernel tiene que reducir bytes o aplicaciones, y ganar en
   pared completa a igualdad de precisión.

Se conservan retirados el split C−C₀ para fases muy blandas, las normas
recurrentes inseguras, los cambios de aceptación sin precisión comparable,
el precondicionador acústico, Q1 y GenEO. Sus resultados están en
[descartados](../../descartados/README.md); no se repiten como novedades.

## Procedencia

Los pilotos aislados candidates.py y sus runners se midieron sobre las
fuentes exactas de 9300f93. El bundle guarda esas fuentes y las finales.
No son rutas importadas por el paquete. La reproducción de las tablas y
gráficas es offline; run_production.py compara las políticas con el código
final y extrae del historial únicamente el constructor anterior.
Las capturas de depuración sin activación del espectro se excluyen.
