# Archivo histórico, no comandos portados

Se conserva el trabajo experimental previo sin resultados binarios ni cachés.
Los scripts aquí pueden mencionar rutas `FFT/`, módulos `pipeline`, campañas
o datos externos del proyecto de origen. **No son comandos autocontenidos**.
No forman parte de la instalación ni de la suite `tests/`.

Las variantes que se necesiten deben portarse a `benchmarks/` con argumentos
explícitos y pruebas. El código de producción y las optimizaciones integradas
están en `src/`; no es necesario ejecutar este archivo para usar SAM/FFT.
