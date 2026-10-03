# Publicación

El repositorio contiene el generador, FFT lineal/no lineal, pruebas y ejemplos.
Los resultados, entornos y las instalaciones locales de solvers de referencia
están excluidos mediante `.gitignore`.

```bash
python -m pytest -q
python examples/smoke_cpu.py
python -m pip wheel . --no-deps --wheel-dir dist
```

El núcleo incorporado conserva su aviso MIT. El autor debe elegir una licencia
para el código propio antes de distribuirlo como software de código abierto.
La limpieza del repositorio no asigna esa licencia ni publica el repositorio.
