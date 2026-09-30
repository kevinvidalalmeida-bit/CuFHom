# Separación verificada del proyecto principal

Este documento conserva el registro de la separación inicial. Actualización
del 30 de septiembre de 2026: el entorno GPU está operativo y existe el
repositorio privado CuFHom con el PR #1. La validación y los tiempos vigentes
están en [G09 lineal](../benchmarks/g09_linear_20260929/RESULTS.md).

## Qué se conservó

- Todos los módulos Python del núcleo `FFT/pipeline`, con namespace
  `cufhom.pipeline` para evitar conflictos con otros paquetes llamados pipeline.
- Núcleo CuFHom Core modificado, copiado sin cambiar sus algoritmos ni avisos.
- SAM por etapas, A2/A4, fibras flexibles, certificados de contactos,
  voxelización y diagnósticos geométricos.
- Solver FFT lineal, no lineal, kernels y rutas GPU integradas.
- Pruebas relevantes, ejemplos pequeños y benchmarks portados.
- Benchmarks/tests experimentales y reportes anteriores como archivo histórico.

`import_manifest.json` registra 169 copias con hashes del estado real de trabajo,
incluyendo archivos nuevos y modificaciones sin commit. Los hashes importados
describen la copia inicial; las adaptaciones posteriores de esta carpeta son
intencionales. Se verificó que los archivos originales mantienen sus hashes.
No se movió, borró ni editó el código del proyecto de origen.

## Qué se desacopló

- Se eliminó el cambio automático al entorno Python personal anterior.
- Las dependencias se instalan mediante `pyproject.toml` en `.venv` propio.
- Se eliminaron imports `geometry_runtime` y la lectura obligatoria de las
  configuraciones CNN; I/O y progreso usan `cufhom.runtime`.
- Los comandos de geometría y Sobol se instalan como `cufhom-generate` y `cufhom-sobol`.
- Los resultados Sobol se crean bajo el directorio de trabajo, no en el código
  instalado; el flujo Sobol mantiene su generador SAMLite histórico.
- El benchmark G09 requiere ahora rutas de datos explícitas. Los casos
  homogéneo/laminado no requieren archivos externos.
- Se quitó del test activo un caso de top-up AR16/Vf40 que dependía del driver
  CNN y sus configuraciones. No se ejecutó ni se presenta como prueba aprobada.
- Los experimentos que aún dependen de datos/rutas históricos se preservan en
  `descartados/historico_20260924/`, fuera de instalación y pruebas activas;
  no se afirman portados. El traslado desde `archive/` conserva sus hashes.

No se copiaron los ~1.4 GB de resultados FFT ni los runs/checkpoints CNN,
entornos anteriores, cachés Numba o carpetas de Git anidadas. La nueva copia
de código/documentación ronda 2.5 MB, sin contar entorno, cachés y resultados
de verificación. Los datos originales siguen donde estaban.

## Verificación de la separación inicial

Entorno nuevo CPU, Linux x86_64, Python 3.12.3. Versiones concretas en
`requirements-cpu.lock`. No utiliza system-site-packages ni el entorno anterior.

1. Instalación editable CPU/test y `pip check`: sin dependencias rotas.
2. `python -I -m pytest -q`: **154 passed, 101 skipped, 7 subtests passed**.
   Los 101 skips son pruebas que requieren CuPy. Hubo 25 warnings; no se
   ocultan como garantía de compatibilidad con versiones futuras.
3. Ejemplo `examples/smoke_cpu.py`: matriz homogénea 4³ igual a elasticidad
   isotrópica analítica y límite elástico del solver no lineal comprobados.
4. CLI no lineal homogéneo 8³: dos incrementos hasta 0.002, tensión axial
   0.006 GPa, residuo máximo ~8.18e-17, sin pasos rechazados.
5. CLI SAM A4, rechazo de A2/A4 incompatible, null rechazado, ruta A2 histórica,
   ruta flexible, imports aislados y `--help` de Sobol sin CUDA: incluidos en tests.
6. Wheel construido e instalado en un directorio temporal fuera del proyecto;
   importaciones, cierre ACG y ejemplo FFT CPU comprobados desde esa instalación.
7. Escaneo básico por patrones de claves privadas/tokens: sin coincidencias
   en el código/documentación seleccionados. No equivale a una auditoría exhaustiva.
8. Git local inicializado en rama `main`, sin commits ni remoto y sin publicación.

La primera ejecución sin `-I` encontró un plugin ROS heredado por PYTHONPATH.
Eso no es una dependencia del paquete. Las instrucciones usan modo aislado;
no se instaló ROS ni PyYAML para ocultar esa contaminación ambiental.

Evidencia local (ignorada por Git): `results/standalone-tests.xml`,
`results/nonlinear_standalone01/` y `dist/cufhom-0.1.0-py3-none-any.whl`.

## Pendientes registrados en la separación inicial

Los tres primeros puntos se resolvieron posteriormente; se conservan aquí
para interpretar el alcance de aquella verificación CPU.

- Crear/configurar entorno GPU y ejecutar las pruebas CuPy en él. Conservar el
  código GPU no significa que su instalación en este entorno esté validada.
- Elegir licencia propia, revisar archivo histórico y decidir qué publicar.
- Elegir nombre/visibilidad del remoto GitHub; primer commit y push posteriores.
- Cuando se quiera, adaptar el proyecto anterior para consumir este paquete.
  No mantener cambios divergentes en ambas copias sin una estrategia explícita.
- Repetir campañas densas o mecánicas grandes sólo por decisión del usuario.

La independencia de instalación y ejecución pequeña no certifica nuevos límites
de Vf/AR, precisión FFT en todo el dominio ni rendimiento GPU.
