# Pruebas completas de las tres ideas HPC

## Resultado y alcance

G09 original 240³, RTX 5070 Ti, 40 incrementos uniformes 0→1 %, float32,
`rtol=1e-5`, material plástico demostrativo sin calibrar. Sin cambiar malla,
física ni tolerancias. **No se modificó producción ni sus defaults**: las
candidatas viven exclusivamente en benchmarks y parches temporales auditados.

La fusión de coordenadas adjuntas y `p.Ap` gana **2,49 % en curva completa**.
La referencia tangente isotrópica ensayada y el prototipo cuFFTDx no mejoran
a la fusión sola. No se concluye que toda referencia tangente o todo backend
cuFFTDx vaya a fallar: son resultados de las variantes concretas descritas.

## 1. Curvas completas: tres repeticiones alternadas

| Variante | Mediana, s | CG | Newton | Decisión |
| --- | ---: | ---: | ---: | --- |
| Producción actual | 57,801 | 4.839 | 100 | Referencia |
| Fusión `p.Ap` | **56,362** | 4.839 | 100 | Ganadora experimental |
| Referencia tangente Voigt de matriz | 58,727 | 4.785 | 101 | Menos CG, más tiempo |
| Fusión + referencia Voigt de matriz | 57,360 | 4.785 | 101 | Pierde frente a fusión sola |

Tiempos individuales: referencia 57,659 / 57,801 / 57,840 s; fusión
56,334 / 56,362 / 56,374 s. Ahorro mediano 1,439 s. Sin exportación de
campos ni serialización; incluye preparación, solución, auditorías y commits.
Una sola tarea de cómputo GPU durante las mediciones. Compilación CPU de los
prototipos cuFFTDx en paralelo, pero ninguna ejecución GPU concurrente.

Las tres curvas fusionadas guardadas coinciden con la referencia, incluidos
los seis esfuerzos medios. Todas las variantes conservan 40 incrementos y
cero rechazos. Máximo residuo con fusión 9,9012e-6; esfuerzo medio prescrito
2,8787e-7; Hill–Mandel 2,3397e-7; violación de fluencia 2,5146e-6;
disipación no negativa. Error máximo Voigt sobre escala de 15 MPa:
2,989e-7. No es una certificación de convergencia temporal o experimental.

[Reporte completo, metadatos y hashes](results/g09_lowlevel_curves_confirm_20260924/report.json).

En el piloto de una corrida, referencia 53,743 s; fusión 52,837 s;
Voigt matriz 54,675 s; Reuss matriz 66,344 s / 5.608 CG;
Voigt de todas las fases 57,904 s / 5.117 CG. Se descartaron estas dos últimas
del estudio repetido por empeorar tanto tiempo como CG.
[Piloto](results/g09_lowlevel_curves_pilot_20260923/report.json).
No mezclar los 53,743 s del piloto con los tiempos de la confirmación para
atribuir diferencias de sesión a un algoritmo.

### Referencias tangentes probadas

La tangente J2 compacta tiene cuatro autovalores desviadores `a` y uno `a+b`.
Su proyección isotrópica Voigt usa `mu = <5a+b>/10`, tomando la media sólo
en matriz; Reuss usa `mu = 5 / (2 <4/a+1/(a+b)>)`. Se conserva el módulo
volumétrico de matriz. La tercera variante incluye las contribuciones
invariantes volumétrica/desviadora de las fibras. Son aproximaciones **del
precondicionador**, nunca de la ley material. Se calculan antes de cada CG y
quedan congeladas durante éste. No se ensayó aquí una referencia anisótropa
media completa ni el precondicionador Green–Jacobi.

[DBFFT §2.5](https://arxiv.org/html/1905.12725v2) propone una referencia basada
en rigidez/tangente media; nuestras proyecciones isotrópicas son una adaptación
de coste reducido, no una reproducción exacta de su precondicionador anisótropo.
Los tiempos host registrados en `setup_wall_s` pueden incluir espera de
trabajo GPU anterior: no deben interpretarse como coste exclusivo de reducción.

### Seguridad de la fusión

Se reutiliza el bucle PCG original y su auditoría del residuo verdadero. La
salida prestada `Ap` se consume antes de la siguiente aplicación. Sólo se
reutiliza el escalar fusionado para el par exacto de vectores que lo produjo.
PTC con desplazamiento positivo usa explícitamente el camino original;
no se omite el término estabilizador `shift*p.p`. Todos los parches se
restauran al salir, también ante excepciones. La candidata no está importada
por ninguna API de producción.

## 2. cuFFTDx: compilado, verificado y medido, no sólo revisado

Prototipo híbrido: cuFFT compleja en XY, cuFFTDx de 240 puntos que fusiona
C2R en Z + tangente material de seis componentes + R2C en Z, seguido de cuFFT
en XY. Incluye todos los mapas, pasadas FFT y lanzamientos. Sin padding,
FP16, TF32 ni `fast_math`. Tablas reales G09; tangente congelada de un campo
de deformación sintético. **Estas mediciones son del operador, no de una
curva completa ni de un CG resuelto**.

| Almacenamiento y líneas por bloque | Operador actual, ms | Prototipo, ms |
| --- | ---: | ---: |
| Registros, 4 líneas | 8,082 | 21,192 |
| Registros, 8 líneas | 8,057 | 21,134 |
| Memoria compartida, 4 líneas | 8,095 | 21,840 |

Siete grupos alternados de cinco aplicaciones calientes por configuración.
Errores L2 del operador ≈3,233e-7; nueva entrada ≈3,232e-7; defecto de
simetría ≈1,603e-11 y producto energético positivo en las muestras. También
se comprobó 9×13×240 con bloques incompletos y respuesta inmutable.
No se llevó una candidata 2,6 veces más lenta a una curva de 40 pasos.

[Registros 4](results/g09_cufftdx_tangent_register4_20260924.json),
[registros 8](results/g09_cufftdx_tangent_register8_20260924.json),
[compartida 4](results/g09_cufftdx_tangent_shared4_20260924.json).

El perfil separado de la variante compartida localiza el problema: por
operador, ≈16,026 ms en FFT XY de acceso intercalado, 4,486 ms en el kernel
fusionado Z/tangente y 1,325 ms en mapas. Son intervalos de eventos CUDA, no
contadores activos de hardware. Reducir los intermedios de Z no compensa el
nuevo coste XY. Una futura variante necesita rediseñar ese acceso, con coste
de transposiciones/bloques incluido, antes de justificar otra curva completa.
[Perfil por etapas](results/g09_cufftdx_tangent_shared4_profile_20260924.json).

Los atributos del kernel muestran 255 registros y 264 bytes locales por
hilo en las variantes de registros; compartida baja a 175 registros y cero
bytes locales, pero requiere 30.720 bytes compartidos por bloque. No se
confunden esos atributos estáticos con ocupación o tráfico DRAM medidos.

SDK MathDx 26.06.1 CUDA13 descargado de
[NVIDIA](https://developer.nvidia.com/cufftdx-downloads) y extraído en
`/tmp/fft-cufftdx-trial.EmP6fA`, sin instalarlo en el sistema ni añadirlo al
repositorio. SHA256 del archivo descargado:
`59a9233db34b75568acbcc5284e6cefe6fad5577ee644f85044971d62eeea353`.
Compilador nvcc 13.0.88, SM120. Los ejemplos y
[traits oficiales](https://docs.nvidia.com/cuda/cufftdx/api/traits.html)
guiaron las longitudes/almacenamiento R2C/C2R y el uso de memoria compartida.

## 3. Validación y reproducción

**303 pruebas aprobadas**, 7,03 s; sólo aviso preexistente `numpy.matlib`.
Incluyen las 287 anteriores y 16 nuevas: carga/descarga, campos, estado
plástico, reinicio, fallback PTC, restauración tras excepciones y referencias
contrastadas con tangentes ensambladas en un caso pequeño de prueba.
Los kernels cuFFTDx se verificaron adicionalmente con los benchmarks citados;
no están incluidos en las 303 pruebas ni habilitados automáticamente.

```bash
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_nonlinear_lowlevel_curves.py \
  --output /tmp/g09-lowlevel-new --repeats 3 \
  --variants baseline fused_dot matrix_voigt fused_matrix_voigt

OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_nonlinear_lowlevel_curves.py

# Ajustar el include a la extracción local del SDK. Salidas temporales nuevas.
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 -arch=sm_120 --shared \
  -Xcompiler=-fPIC \
  -I/tmp/fft-cufftdx-trial.EmP6fA/nvidia-mathdx-26.06.1-cuda13/nvidia/mathdx/26.06/include \
  FFT/benchmarks/cufftdx_tangent_candidate.cu -o /tmp/fft-tangent-new.so

OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_cufftdx_tangent.py \
  --library /tmp/fft-tangent-new.so --profile-stages \
  --output /tmp/g09-cufftdx-new.json
```

Para las otras compilaciones: añadir `-DFFT_FPB=8`, o bien `-DFFT_SHARED`
con cuatro líneas. Valores predeterminados: 30 elementos por hilo y cuatro
líneas. No todas las combinaciones EPT/tamaño son compatibles con el SDK.

## 4. Consulta posterior: ¿los 40 pasos son estratégicos?

No: se mantiene `Δε=0,00025`, equivalente a 0,025 puntos porcentuales.
Los 40 incrementos físicos no son los 100 Newton ni los 4.839 CG.
Desglose de `baseline_2`, sin sumar preparación global:

| Tramo de deformación axial | Pasos | CG | Tiempo de intentos aceptados, s |
| --- | ---: | ---: | ---: |
| 0–0,05 % | 2 | 99 | 1,261 |
| 0,05–0,50 % | 18 | 2.558 | 29,360 |
| 0,50–1,00 % | 20 | 2.182 | 24,946 |

El primer punto con plasticidad local está en 0,05 %, tras el punto sin
plasticidad de 0,025 %. No hay una región elástica larga para ahorrar muchos
pasos. El tramo final es más suave, pero su pendiente incremental pasa de
≈2,831 GPa en 0,55 % a ≈2,362 GPa al llegar a 1 %; no es exactamente lineal.
Una curva macroscópica suave tampoco prueba que todos los puntos materiales
sigan trayectorias proporcionales ni que pueda saltarse su historia plástica.

Primer ensayo propuesto, **no ejecutado ni activado**: 32 pasos, conservando
los 24 primeros hasta 0,60 % y duplicando el incremento de 0,60 a 1,00 %
(ocho pasos en vez de dieciséis). Es una prueba sencilla de distribución,
sin control adaptativo costoso ni salto en la transición principal. Esos
umbrales son específicos de esta curva demostrativa, no de toda resina/G09.
Comparar en estados de carga coincidentes con 40 y 80 pasos, incluyendo
esfuerzos, estado plástico, equilibrio y tiempo total. Ahorrar ocho pasos no
garantiza ahorrar ocho veces el coste anterior: puede aumentar Newton/CG.

La comparación histórica 40/80 tenía ≈0,24 % de la escala de 15 MPa, por
encima del objetivo temporal de 0,1 %. No se debe usar la equivalencia entre
implementaciones de 40 pasos como prueba de convergencia de trayectoria.
No se reactivaron los controles adaptativos descartados ni se cambió el
espaciado actual en respuesta a esta consulta.
