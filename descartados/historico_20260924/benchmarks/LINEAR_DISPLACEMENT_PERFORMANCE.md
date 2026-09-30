# FFT lineal: campo indexado y PCG en coordenadas de desplazamiento

G09 original 240³, seis cargas, CuPy float32 y tolerancia de equilibrio 1e-5.
La opción integrada es `linear_formulation='displacement'`. La API usa ahora
`auto` por defecto: selecciona `displacement` cuando es compatible y `strain`
en los demás casos. Se puede forzar la ruta histórica mediante `strain`.

## Preparación constitutiva

Se transfieren offsets planos de los vóxeles de fibra en lugar de triples XYZ.
Se conserva la agrupación de orientaciones, el mapa de materiales y la tabla
de rigidez: ambos arrays GPU resultaron exactamente iguales. La preparación
con geometría compilada pasó de 7,27 a 4,80 ms (34 % menos). El coste total
cambió de 4,694 a 4,692 s; ese cambio global está dentro del ruido de medida.
Datos: [g09_cfield_flat_indices_20260922.json](results/g09_cfield_flat_indices_20260922.json).

Los 6,19 s citados anteriormente procedían de otro perfil histórico; no
representaban una medición aislada del arranque en frío ni se podía atribuir
su diferencia respecto a 4,83 s exclusivamente al campo constitutivo. Una
medición nueva, antes de integrar las coordenadas, dio 5,378 s desde la lectura
de geometría hasta completar el primer solve, después de importar Python:
0,014 s de lectura, 0,331 s de compilación y 5,033 s de resolución inicial.
Los tiempos de arranque dependen también de las cachés CUDA/disco.

## Formulación integrada

Se reutiliza la isometría Q del solver no lineal, fijando a cero todos los modos
medios de las fluctuaciones. El sistema de CG es Q* C Q, con referencia
homogénea de matriz. Las coordenadas tienen tres componentes complejas por
frecuencia RFFT; se conservan los modos admisibles, Mandel y los planos Nyquist.
La ley constitutiva y las seis cargas son las mismas. El factor global de
normalización del operador directo histórico se elimina coherentemente del RHS.

La prueba inicial dio 360 -> 288 iteraciones totales y 4,745 -> 3,740 s.
La comparación final integrada dio medianas de **4,759 s -> 3,755 s (21,1 %
menos tiempo)** en tres repeticiones por ruta, incluyendo auditoría del residuo
verdadero al finalizar cada carga. El rango de la opción rápida fue
3,734–3,758 s. Las repeticiones fueron alternadas y sin procesos GPU de prueba
concurrentes en el informe definitivo:
[repeticiones de la API integrada](results/g09_linear_displacement_release_20260922/report.json).
Los tiempos incluyen la escritura pequeña de metadatos; excluyen lectura de
geometría, compilación previa, calentamiento y exportación de campos.

Si la auditoría física falla, se corrige mediante el CG histórico manteniendo
el presupuesto total de iteraciones. Si vuelve a fallar, se lanza un error.
Las pruebas incluyen una falsa convergencia interna inyectada para comprobar
este mecanismo, matrices homogéneas y comparación con CPU64 en mallas pequeñas.

La diferencia relativa de C_eff observada entre rutas fue aproximadamente
1,31e-6 (0,000131 %). Supera el umbral comparativo conservador de 1e-6 y así se
registra: **no se certifica equivalencia de C_eff a 1e-6 para G09**. Ambas rutas
cumplen el residuo físico solicitado de 1e-5. La comprobación adicional con el
solver histórico a tolerancia 1e-6 fue inconclusa: se estancó en float32 y acabó
produciendo C_eff no finito. No se usa esa ejecución como referencia válida.
La selección rápida es automática en configuraciones compatibles; no se relajan
tolerancias físicas. Los resultados anteriores corresponden a selecciones
explícitas de cada ruta antes de cambiar el valor predeterminado.

## Papers y decisiones

Perfil posterior de la ruta predeterminada y prioridades basadas en CUDA:
[`LINEAR_NEXT_OPTIMIZATIONS.md`](LINEAR_NEXT_OPTIMIZATIONS.md).

- [Lucarini y Segurado, DBFFT (2019)](https://arxiv.org/abs/1905.12725):
  incógnitas de desplazamiento en Fourier y precondicionamiento de Krylov.
  Motiva la representación compacta; su aceleración publicada no se presupone
  transferible a G09.
- [Ladecký et al. (2023)](https://arxiv.org/abs/2203.02962):
  referencia homogénea y sistema bien escalado. Su discretización FE difiere
  de la espectral conservada aquí; adoptamos la idea de precondicionar.
- [Ladecký et al., J-FFT (2026)](https://arxiv.org/html/2508.02613v2):
  Green-Jacobi resulta competitivo con propiedades suaves. En su comparación
  de interfaces abruptas gana Green. Se pospone para la geometría binaria G09.

```bash
OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/profile_g09_fft.py \
  --formulation displacement --repeats 3 --output /tmp/g09-linear-fast.json

OPENBLAS_NUM_THREADS=1 python FFT/benchmarks/benchmark_linear_solver.py \
  --repeats 3 --output /tmp/g09-linear-validation
```
