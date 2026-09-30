# Ensayos de ensamblaje de Ceff

G09 240³, seis cargas, mismo residuo físico `1e-5` y 288 iteraciones CG.

- `reduction_kernel_trial.py`: la reducción genérica CuPy con productos
  float64 elevó el total **3,071 → 5,635 s** (cinco repeticiones). Descartada.
- `warp_trial.py`: reutilizar la reducción CUDA por warps del solver dio
  **3,070 → 3,040 s** (tres repeticiones). Aceptada e integrada como
  `postprocess_assembly='fused'`, con comprobación de compatibilidad y fallback.
  Aquí queda solo el prototipo temporal; no se importa desde producción.

Se evalúan todas las entradas de Ceff con sumas float64; no se sustituye el
ensamblaje energético por un promedio de tensiones ni se presupone simetría.
Los informes conservan los hashes del solver usado por los pilotos. El parche
local temporal está en cada script; esos hashes por sí solos no lo describen.
Los scripts se ejecutaban con `PYTHONPATH=benchmarks` y las rutas explícitas
`--geometry`, `--material`, `--output`; para el estado integrado usar los
benchmarks activos. Los pilotos llaman a `profile_g09_linear.run`, cuyo campo
`benchmark_sha256` identifica el runner compartido, no el script del piloto.
