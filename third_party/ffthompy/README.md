# Núcleo FFTHomPy modificado

El código se instala desde `src/ffthompy`, conservando su aviso MIT y los avisos
individuales. Corresponde a la copia local modificada usada por el solver, no
a una descarga nueva ni a una versión upstream sin cambios.

Procedencia y hashes de cada archivo: `docs/import_manifest.json`.
Entre los cambios heredados se encuentran FFT/CuPy, representaciones de Cfield,
proyecciones, CG y kernels GPU. La separación de carpetas no los reemplazó.
No se debe instalar otro paquete que exporte `ffthompy` en el mismo entorno.

La licencia del código propio de SAM/FFT se decide por separado antes de publicar.
