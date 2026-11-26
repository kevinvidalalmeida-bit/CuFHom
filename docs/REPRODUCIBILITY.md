# Reproducción de los experimentos lineales

El código público incluye la biblioteca, sus pruebas, los ejemplos CPU y una
campaña reproducible en `benchmarks/paper/`. No necesita archivos editoriales.

## Preparación

```bash
python -m pip install -e '.[dev]'
python examples/smoke_cpu.py
python benchmarks/paper/run.py --dry-run
python benchmarks/paper/run.py --backend numpy --small --output results/smoke
```

`--small` verifica la instalación y las entradas en mallas pequeñas, sin CUDA.
La campaña completa necesita CuPy/CUDA y memoria suficiente. Cada ejecución exige un
directorio nuevo y conserva los fallos y auditorías. Los resultados nuevos no
sobrescriben mediciones anteriores ni sustituyen automáticamente las del artículo.

## Entradas y protocolos

- `benchmarks/paper/inputs/`: geometrías, materiales y una imagen pública
  comprimida con licencia CC BY 4.0 y procedencia. Las demás mallas se regeneran.
- `benchmarks/paper/protocols/`: mallas, tolerancias, precisiones y sesiones.
- `benchmarks/paper/run.py`: seis cargas Mandel por llamada; primera llamada y
  llamadas calientes separadas, siempre desde cero.
- [`benchmarks/paper/ablations/`](../benchmarks/paper/ablations/README.md): controles de implementación agrupados, ejecutados con `python -m benchmarks.paper.ablations`; resultados nuevos bajo `results/ablations/`.
- `benchmarks/paper/summarize.py`: CSV y diagnósticos de una campaña nueva.

La [guía de campaña](../benchmarks/paper/README.md) documenta cada cronómetro,
norma residual, modo de precisión y medida de memoria. Las dos formulaciones
usan normas distintas; una misma tolerancia no implica el mismo error físico.
Los tiempos dependen del equipo, el entorno y la carga del dispositivo.

La API de uso habitual permite seleccionar `precision="fp32"`, `"fp64"` o
`"mixed"` y detectar CPU/GPU automáticamente, como explica la
[guía de uso](USAGE.md). Los ejecutores de rendimiento usan dispositivos y
precisiones explícitos en sus protocolos: `run.py` exige CUDA por defecto y no
lo sustituye por CPU. Sus protocolos conservan sus tolerancias propias; no se
modifican al cambiar los valores predeterminados de la API de uso habitual.

## Referencias externas opcionales

Las revisiones usadas por los adaptadores se fijan en
[external_sources.json](../benchmarks/paper/protocols/external_sources.json).
Para obtener solo el código en checkouts locales ignorados:

```bash
git clone --filter=blob:none --sparse https://github.com/DataAnalyticsEngineering/UNOCG.git external/UNOCG
git -C external/UNOCG checkout 461f74f7a40f1c6c6d8ac053f6130428650a0133
git -C external/UNOCG sparse-checkout set unocg

git clone --filter=blob:none --sparse https://github.com/vondrejc/FFTHomPy.git external/FFTHomPy
git -C external/FFTHomPy checkout 2c23c80dd2cab46f1090103e613b4f886b3daac7
git -C external/FFTHomPy sparse-checkout set ffthompy
```

FANS aquí es `FansPreconditioner` de UNOCG, implementado en PyTorch. No necesita
pesos UNO ni datos de entrenamiento. Instala las dependencias de cada referencia
en su entorno de benchmark; PyTorch necesita CUDA para FANS y CUPTI para perfiles.
Los adaptadores reciben `--source` y conservan la revisión y huellas externas.
FFTHomPy restaura aliases antiguos en proceso, sin editar el código original.

## Distribución

El wheel contiene la biblioteca y sus licencias. El sdist añade las pruebas,
ejemplos y entradas compactas para reproducir experimentos. Los datos y programas
de terceros conservan sus licencias y atribuciones.

Este repositorio público contiene únicamente el software lineal y sus
benchmarks. Para exportar una instantánea del último commit:

```bash
python tools/export_public.py dist/cufhom-public.tar.gz
```

Antes de publicar cambios, comprueba todos los commits y etiquetas:

```bash
python tools/check_public_tree.py --all
```

La comprobación rechaza rutas reservadas incluso en commits anteriores.
Las reglas `export-ignore` afectan a los archivos descargables y no eliminan
contenido del historial Git. El material de investigación reservado se mantiene
en una copia local separada y no se envía a este repositorio.
