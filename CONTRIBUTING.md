# Contribuir a CuFHom

Describe el problema y el comportamiento esperado antes de proponer cambios
amplios en las formulaciones numéricas o en los formatos de datos.

## Entorno y comprobaciones

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m ruff check src core tests examples benchmarks tools
python -m ruff format --check src tests examples benchmarks tools
python -m pytest
python -m build
```

Usa nombres y formato coherentes con PEP 8, y docstrings PEP 257 en los módulos,
clases, funciones y métodos públicos del código propio. Documenta entradas,
salidas, unidades, convenciones tensoriales y fallos cuando no sean evidentes.
Ruff aplica las reglas configuradas en `pyproject.toml`; no sustituye una
revisión del contenido de las docstrings. El núcleo incorporado conserva su
formato y sus avisos originales.

Incluye pruebas de comportamiento con referencias analíticas o comprobaciones
independientes cuando corresponda. Evita depender de los datos de las campañas.
La [guía de pruebas](docs/TESTING.md) describe los marcadores CPU/GPU y la
habilitación explícita de CUDA.

## Resultados y procedencia

Las campañas reproducibles están en
[benchmarks/paper/](benchmarks/paper/README.md). Conserva las huellas de las
fuentes y entradas cuando cambies un solver. Cada ejecución debe escribir un
directorio nuevo para preservar resultados anteriores.
No ejecutes benchmarks comparativos junto con otras cargas de GPU; registra el
entorno, los residuos y la frontera exacta de cada cronómetro.

El código propio se distribuye bajo la [licencia MIT](LICENSE). Conserva las
licencias y atribuciones de terceros y documenta las condiciones de nuevos
datos o materiales externos. La reproducción y la distribución se describen en
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Versionado

Seguimos [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
`MAJOR.MINOR.PATCH`. `1.0.0` identifica la primera API pública estable;
los cambios incompatibles incrementan `MAJOR`, las funciones compatibles
incrementan `MINOR` y las correcciones compatibles incrementan `PATCH`.
Durante el desarrollo inicial se usa `0.y.z`. Una publicación conserva su
identidad: los resultados históricos mantienen la versión realmente ejecutada.
## Contenido público

Contribuye únicamente al software lineal y sus benchmarks. Mantén manuscritos,
figuras, registros de mediciones e investigación no lineal fuera de este
repositorio y de su historial. Las entradas redistribuibles y los protocolos
permiten generar resultados nuevos, que se guardan en directorios ignorados.
Ejecuta `python tools/check_public_tree.py --all` antes de enviar cambios.
Las exclusiones de archivos descargables no protegen el historial Git.
