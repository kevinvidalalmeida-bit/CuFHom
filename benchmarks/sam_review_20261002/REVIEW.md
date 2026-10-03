# SAM: revisión de Vf = 60 %, AR = 30 y orientación isotrópica

Fecha: 2026-10-02. Alcance: código actual, referencia de 2017, mediciones de
contactos y pilotos acotados. **El 60 % aún no está demostrado en este repositorio.**

Este documento conserva la revisión inicial. Las mejoras posteriores están en
[`RESULTS.md`](../sam_improvements_20261002/RESULTS.md); la comprobación de
capacidad del cubo y la corrección de la inercia histórica están en
[`FEASIBILITY.md`](../sam_improvements_20261002/FEASIBILITY.md).

## Conclusión

El objetivo tiene respaldo publicado para AR = 33. Para aproximarnos conviene
trabajar sobre la ruta `paper-a4`, fijar la convención de cilindro físico y
mejorar la migración colectiva y sus presupuestos. La velocidad del FFT no
resuelve el atasco de inserción de fibras.

Los problemas concretos son: A4 no controlado en la ruta histórica, presupuestos
pequeños, reconstrucción cuadrática de vecinos, auditorías con memoria cuadrática
y descenso que puede consumir miles de evaluaciones antes de resolver contactos.
Las prioridades propuestas abajo son hipótesis de mejora; las aceleraciones
medidas corresponden únicamente a los experimentos descritos.

## Qué consiguió el artículo

[Schneider (2017), §2.2, §3.4.2 y Fig. 7](https://doi.org/10.1007/s00466-016-1350-7):

| Parámetro | Caso denso publicado |
|---|---:|
| Longitud / diámetro | 330 / 10 µm; AR = 33 |
| Lado de caja | 700 µm |
| Separación mínima | 0,01 µm = 0,001 diámetros |
| Fibras / Vf | 7940 / 60 % |
| Control de orientación | A4 global isotrópico |

Con separación 2 µm y caja 710 µm alcanzó 47,46 % tras 24 h. El texto no da
un tiempo separado para el caso de 60 %. Usa incrementos de 2,5 puntos
porcentuales y permite hasta 500 000 pasos antes de reducir la adición.
Los contactos se calculan como cápsulas; después se eliminan los casquetes
para representar cilindros. En las geometrías densas aparecen haces alineados
localmente compatibles con A4 global isotrópico. El caso de 60 % no se empleó
para calcular elasticidad por problemas de resolución de la separación.

Se verificaron las páginas completas del PDF local
`/home/user/Downloads/schneider2016 (1).pdf`, además de la
[ficha de Fraunhofer](https://publica.fraunhofer.de/entities/publication/5554dfd0-672f-41a9-aac7-812f542d24cb).

**Interpretación para nuestro objetivo:** isotropía global no garantiza ausencia
de agrupaciones o correlaciones locales. Además de A2/A4, una futura campaña
debe medir A2 por subvolúmenes y correlaciones de orientación/distancia si se
quiere limitar la formación de haces. Eso añade requisitos al objetivo publicado.

## Hallazgos en el código

### 1. Dos rutas con controles distintos

[`generate_rve_main`](../../src/cufhom/pipeline/rve_generator.py) usa
`SAMLiteGenerator`, con migración colectiva/FIRE cuando corresponde y corrección
de A2. Ya tiene inserción con solapes temporales: no hace falta inventar esa
capacidad. La ruta
[`geometry.py`](../../src/cufhom/cli/geometry.py) selecciona `paper-a4` para
fibras rectas y comprueba A2, A4 y contactos independientemente.

Recalculé los momentos de los masters existentes, verificando sus SHA-256:

| Master | Fibras | Error A2 relativo (%) | Error A4 relativo (%) |
|---|---:|---:|---:|
| Vf25 / AR20 | 985 | 0,00336 | 5,30864 |
| Vf25 / AR10 | 239 | 0,12867 | 8,59177 |
| Vf15 / AR20 | 590 | 0,26000 | 7,47094 |

Norma: Frobenius del tensor completo, dividida por la norma del objetivo.
Estos masters no fueron generados exigiendo `A4 <= 1e-4`; sus desviaciones
también incluyen el muestreo finito. El barrido de tiempos existente llega a
Vf25/AR20 y no constituye validación de empaquetamiento Vf60/AR30.

### 2. Volumen y extremos de la fibra

El rasterizador y el conteo del CLI experimental usan cápsulas:

`Vcapsula = pi*d²*L/4 + pi*d³/6`.

Para AR30, el casquete añade 2,2222 % respecto al cilindro. Una geometría con
60 % nominal de cápsulas equivale a **58,70 % nominal de cilindros** con esos
mismos ejes. Cambiar sólo la etiqueta de volumen produciría una comparación
incorrecta. AR aquí es longitud del segmento central dividida por diámetro;
la longitud total de la cápsula incluye otro diámetro.

Para seguir el artículo propongo cilindros planos en conteo y raster final,
manteniendo el contacto conservador de cápsulas. Debe ser una opción explícita
y coherente en metadatos, exportación y validación.

### 3. Presupuesto insuficiente por defecto

[`Settings`](../../src/cufhom/pipeline/sam_staged_insertion.py) usa lote 4,
40 etapas e inicio de 12. Sin fallos permite 168 fibras; incluso una cota generosa
que conceda 48 iniciales sin descontar etapas de arranque sólo permite 204.

| AR / caja / diámetro | Fibras para 60 % de cilindros | Para 60 % de cápsulas |
|---|---:|---:|
| 30 / 60 / 1 | 5500 | 5381 |
| 30 / 64 / 1 | 6675 | 6530 |
| 33 / 70 / 1 | 7940 | 7783 |

Con AR30/caja60, 204 cápsulas sólo representan 2,27 %. Con lote 4 e inicio 12,
alcanzar 5381 requiere al menos 1344 etapas exitosas. El CLI debería comprobar
esa compatibilidad antes de ejecutar. La API admite `initial`; el CLI aún no
expone reanudación desde checkpoint.

En producción, `run_and_compact` limita `max_iter_per_batch` a 160–240 aunque
se aumente `sam_collective_fire_max_iter`. En la ruta A4 el CLI concede 3000
iteraciones; el solver tiene un paso máximo de 0,3. Son límites de implementación,
no límites físicos de Vf.

### 4. Escala de rotación histórica

En [`SAMLiteGenerator.rotational_scale`](../../src/cufhom/pipeline/sam_generator.py)
la expresión no coincide con la división por `AR + 2/3` que usa
[`rotational_inertia`](../../src/cufhom/pipeline/sam_paper_reference.py).
A AR30/d1 da 72171,42 frente a 78,45, un factor 920,01.

Esto **no implica una ralentización total de 920 veces**: FIRE usa
`rotation_gain >= 2*rot_scale/L²`, que cancela esa escala cuando domina esa rama.
Otras rutas emplean su inverso directamente. Una corrección requiere comparar
trayectorias, pasos y convergencia de cada ruta; cambiar únicamente la fórmula
no es una solución validada.

## Mediciones nuevas

### Contactos y vecinos

[Runner](profile_geometry.py), [resultados secuenciales](profile_serial.json).
AR30, caja60/d1, centros y orientaciones aleatorios con solapes; tres muestras
después de un calentamiento. Una hebra Numba. No son geometrías empaquetadas
ni tiempos de generación completa.

| Fibras | Objetivo exhaustivo (s) | Objetivo con lista simple reutilizada (s) | Construir simple (s) | Construir multinivel (s) |
|---|---:|---:|---:|---:|
| 512 | 0,00868 | 0,00116 | 0,00949 | 0,09519 |
| 2048 | 0,13708 | 0,01457 | 0,15637 | 1,52445 |
| 5500 | 0,95638 | 0,09929 | 1,09721 | 10,88485 |

Todas las energías, gradientes y máximos coinciden bit a bit con el cálculo
exhaustivo. A 5500 fibras, la lista final conserva 184063 entradas incluyendo
autocontactos, frente a más de 15 millones de pares potenciales.
El beneficio del caché depende de cuántas evaluaciones se hagan entre reconstrucciones.

El constructor `build_aabb_pairs` recorre todos los pares dos veces; su filtro
reduce trabajo posterior pero conserva coste cuadrático. La implementación
multinivel actual encarece mucho una reconstrucción completa. En el
[benchmark de trayectorias](neighbor_trajectories.json), 512 fibras/AR33 tardaron
0,1100 s con lista simple y 0,6582 s con multinivel. No recomendaría activar
multinivel por defecto con esta evidencia.

Los kernels de contacto y de construcción de la ruta A4 usan `njit` serial,
sin `prange`. Cambiar `--threads` por sí solo no paraleliza esos bucles.

### Auditoría independiente

[`audit_periodic_fibers`](../../src/cufhom/pipeline/sam_periodic_audit.py) materializa
`triu_indices`, desplazamientos y segmentos de todos los pares. Con los estados
anteriores tarda 0,0811 s para 512 fibras y 1,2921 s para 2048.

Sólo los arrays principales persistentes suman aproximadamente 1,69 GiB para
5500 fibras y 3,52 GiB para 7940, sin incluir temporales. Es una estimación de
los arrays del código, no una medición del pico de memoria. La auditoría se
repite antes y después de relajar, al certificar cada candidato y al terminar.

### Pilotos completos de geometría continua

AR30, caja60, d1, isotropía A4, tolerancia de penetración/d = 1e-6 y A4 relativo
<= 1e-4. Los casos se ejecutaron sin rasterizar ni resolver FFT.

| Piloto | Fibras certificadas | Vf nominal cápsulas | Vf nominal cilindros | Resultado |
|---|---:|---:|---:|---|
| Inicio, lote64 | 448 | 4,9955 % | 4,8869 % | Conteo alcanzado |
| Continuación, lote224 | 1120 | 12,4888 % | 12,2173 % | Presupuesto temporal agotado |

El primero tardó 2,32 s dentro de la generación instrumentada con cProfile.
La continuación tardó 61,25 s incluyendo cierre/auditoría, con presupuesto
nominal de 60 s comprobado en eventos. Su último checkpoint tiene A4 relativo
2,89e-8 y penetración máxima/d 9,64e-7.

El primer intento de llegar a 1344 fibras, aproximadamente 15 % nominal de
cápsulas, agotó 5000 iteraciones / 7817 evaluaciones: penetración/d = 0,00311,
A4 relativo = 3,02e-5. Falló por contactos. Fue rechazado y se conservó el
checkpoint de 1120. El segundo intento se interrumpió por tiempo.
**Este piloto no determina el Vf máximo del método.**

`accepted=False` en los resúmenes globales indica que falta la certificación
voxelizada; los checkpoints continuos sí pasan la auditoría geométrica.
Se guardan [inicio](pilot_vf05_summary.json),
[continuación](pilot_continuation_summary.json), eventos y ambos NPZ.

## Optimizaciones que haría, por prioridad

1. **Definir el caso comparable:** cilindro físico, AR30, A2/A4 isotrópicos y
   separación explícita. Empezaría con caja64/d1, que deja margen respecto a 2L;
   mantendría AR33/caja70 como caso de comparación con el paper. Esta elección
   aumenta el conteo respecto a caja60 y debe incluirse en la comparación de coste.
2. **Lotes según Vf y presupuesto adaptativo:** ensayar adiciones de 1–2,5 puntos
   porcentuales al principio y reducirlas al crecer el coste de relajación.
   Recuperar tamaño de lote tras varios éxitos; ahora el lote sólo disminuye.
   Dar más iteraciones cuando el residual sigue bajando y reiniciar/reducir el
   lote cuando se estanca. Persistir presupuesto, semilla y estado RNG al reanudar.
3. **Acelerar la migración A4:** comparar descenso con paso espectral y salvaguarda,
   L-BFGS sobre traslaciones/direcciones, y FIRE con energía/gradiente A4 coherentes.
   Conservar métrica de inercia, norma unitaria y certificación final independiente.
   Medir evaluaciones hasta factibilidad y Vf certificado por segundo. La variante
   L-BFGS existente sólo controla A2; no es todavía el sustituto buscado.
4. **Vecinos y energía:** hacer ajustable el margen de Verlet y medir reconstrucciones
   en trayectorias reales. Evaluar una jerarquía espacial sobre segmentos auxiliares
   de cada eje recto para reducir pares candidatos, manteniendo cada fibra rígida.
   Comparar con AABB actual: para fibras largas/cajas pequeñas, una grilla simple
   puede producir demasiados candidatos. Paralelizar contactos con acumulación
   sin carreras y verificar fuerzas frente al cálculo exhaustivo.
5. **Auditoría por bloques:** recorrer pares en bloques de memoria acotada, conservando
   imágenes periódicas y autocontactos. Mantener un verificador independiente de
   las listas usadas por el optimizador. Eliminar evaluaciones diagnósticas repetidas
   sobre estados idénticos sólo cuando se conserve la garantía de certificación.
6. **Reducir coste secundario:** evaluar candidatos por lote sin copiar toda la tabla
   para cada candidato; acumular A4 con sus 15 componentes independientes y gradiente
   fusionado. Perfilar antes de priorizarlo sobre contactos y convergencia.
7. **Separar construcción continua y resolución voxelizada:** a AR30/caja64,
   256³ sólo da 4 vóxeles por diámetro; 384³ da 6 y 512³ da 8. Son puntos de ensayo
   para convergencia. Una separación de 0,001d no se resuelve con esas mallas, por
   lo que el éxito geométrico al 60 % no certifica propiedades mecánicas FFT.

En cada etapa exigiría que se mantengan dimensiones, número de fibras comprometidas,
A2/A4 y tolerancias declaradas. El informe debe registrar Vf objetivo, nominal
y voxelizado por separado, además de varias semillas y estadística local.

## Reproducción y comprobaciones

53 pruebas y 7 subpruebas pasaron: contactos, vecinos, orientación, referencia
A4, etapas y CLI. [Salida](tests.txt). Pytest requirió desactivar la autocarga de
plugins externos de ROS, que intentaban importar `yaml` fuera de las dependencias
del proyecto. No se alteró el entorno para resolverlo.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/test_sam_contact_energy.py tests/test_sam_neighbors.py \
  tests/test_sam_orientation.py tests/test_sam_paper_reference.py \
  tests/test_sam_staged.py tests/test_sam_a4_cli.py

.venv/bin/python benchmarks/sam_review_20261002/profile_geometry.py \
  --output /tmp/sam_profile_new.json

.venv/bin/python benchmarks/sam_review_20261002/pilot_continuation.py \
  --initial benchmarks/sam_review_20261002/pilot_vf05_last_geometrically_valid.npz \
  --output /tmp/sam_continuation_new --seconds 60
```

Usar rutas nuevas para conservar evidencia. El perfil de masters requiere el
manifest local indicado en el runner. `profile_serial.json` incluye hashes de
fuentes y geometrías. `profile.json` conserva una primera medición exploratoria
que coincidió parcialmente con otros ensayos; las tablas usan el pase secuencial.
Los tiempos por presupuesto de pared pueden producir distintos checkpoints en
otra máquina. Estos archivos documentan una revisión; las optimizaciones
propuestas todavía requieren implementación y comparación controlada.
