# Ablaciones de implementación

```bash
python -m benchmarks.paper.ablations --pair hex_stencil --dry-run
CUFHOM_FUSED_COMPILE=sync python -m benchmarks.paper.ablations \
  --pair hex_stencil --precision fp64 --output results/ablations/stencil_fp64
```

Se alterna AB/BA entre tres sesiones pareadas; cada brazo usa su proceso,
un calentamiento y cinco llamadas calientes. El material y la referencia se
preparan fuera del reloj de las ablaciones. Se conservan auditorías, tensores,
conteos y contadores que indican que la variante realmente se ejecutó.
`summary.json` presenta los cocientes variante/control por sesión y su rango.

| Par | Control / variante |
| --- | --- |
| `hex_stencil` | Producción / gather general en todos los nodos. |
| `hex_krylov` | Producción / recorridos Krylov separados. |
| `hex_parseval` | Krylov separado / producto espacial adicional. |
| `hex_inverse` | Inversa almacenada / reconstruida; solo FP64. |
| `hex_fft`, `gani_fft` | Producción / ruta con FFT de biblioteca. |
| `gani_krylov` | Producción / actualizaciones vectoriales separadas. |
| `gani_krylov_energy` | Producción / Krylov y energía separados. |
| `gani_energy` | Vectores separados / vectores y energía separados. |
| `gani_slabs` | Producción / mismos kernels, planificación por etapas de slabs. |
| `gani_reference` | Referencia habitual / tensor 2μ₀I. Cambia el precondicionador. |
| `gani_loads` | Espectros de fase reutilizados / seis componentes de carga transformadas directamente. FP64, mixto o FP32. |
| `hex_evaluation`, `gani_evaluation` | Evaluación fresca df64 / IEEE FP64, manteniendo iteraciones FP32 y acumulación FP64; solo mixto. |

Los pares de evaluación rechazan otras precisiones para evitar comparar dos
rutas FP64 equivalentes por accidente. Cada auditoría conserva la aritmética
realmente utilizada. El par de cargas comprueba `phase_spectra_rhs` en las seis
cargas de ambos brazos; falla si el control no reutiliza los espectros o si la
variante sigue reutilizándolos. El resumen comprueba igualdad de entradas y, excepto en el par que estudia
la referencia, igualdad de referencia; también comprueba aceptación física y
matrices finitas antes de presentar razones;
conserva los seis conteos y residuos por llamada y las diferencias tensoriales.

```bash
CUFHOM_FUSED_COMPILE=sync python -m benchmarks.paper.ablations \
  --pair gani_loads --precision mixed --n 21 --sessions 1 \
  --output results/ablations/pilot_gani_loads
CUFHOM_FUSED_COMPILE=sync python -m benchmarks.paper.ablations \
  --pair hex_evaluation --precision mixed --n 24 --sessions 1 \
  --output results/ablations/pilot_hex_evaluation
```

Estos tamaños pequeños verifican los controles y su aceptación; no reproducen
los tiempos históricos. Omitir `--n` selecciona 189³/192³ para nuevas mediciones.

`--n 24` (HEX8) o `--n 21` (GaNi) solicita un piloto explícito. Las variantes
usan interfaces privadas y cambios locales al proceso, nunca modifican el
solver instalado. Deben fallar si esas interfaces cambian. Algunos controles
cambian varias operaciones, y pueden cambiar los conteos: no se atribuye el
cociente completo a una sola instrucción ni se multiplican factores. La ruta
GaNi por etapas con cuFFT y el control de planificación de slabs son distintos.
Los perfiles GPU de eventos complementan estas mediciones; sus tiempos no
sustituyen la duración completa de una llamada.

## Organización y salidas

Todo el código de estas ablaciones está en esta carpeta: `__init__.py` contiene
el ejecutor y la validación, `controls.py` las variantes locales al proceso,
y `__main__.py` la entrada del comando. No se modifica el solver de producción.

Ejecuta los comandos desde la raíz del repositorio. Si omites `--output`, se
crea una carpeta nueva bajo `results/ablations/` con el par, precisión y un
identificador único. Una ruta explícita también debe ser nueva. Las salidas
están ignoradas por Git y contienen JSON de tiempos, matrices 6×6, conteos y
auditorías; no se exportan campos volumétricos. No mezcles mediciones nuevas
con campañas históricas ni midas rendimiento con otros trabajos en la GPU.
El ejecutor usa compilación síncrona por defecto para completar el calentamiento
antes de las llamadas medidas; registra cualquier configuración explícita del entorno.

La tabla del artículo utiliza las mediciones históricas archivadas. Los nombres
`hex_gather`, `gani_slab_order`, `hex_ieee` y `gani_ieee` del archivo corresponden
a `hex_stencil`, `gani_slabs`, `hex_evaluation` y `gani_evaluation` en este ejecutor.
Los pares `gani_krylov` y `gani_reference` están disponibles para nuevas
extensiones; su presencia en el ejecutor no implica que tengan cifras en
la tabla histórica. El control de referencia cambia el precondicionador
y registra `reference_changed=true`, conservando el mismo problema físico.
