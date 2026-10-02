# Pilotos de la auditoría, 1 de octubre de 2026

Estos archivos son diagnósticos aislados. No forman parte del paquete instalado,
no cambian `src/` ni las opciones predeterminadas. Para ejecutarlos se necesita
añadir esta carpeta y `benchmarks/` a `PYTHONPATH` explícitamente.

La referencia de producción es `92c9ac5b8204288244e82536b254078d6d767e66`.
Todos los ensayos conservan campos/FFT FP32, reducciones FP64 y auditoría del
residual físico final. Los tiempos pertenecen a comparaciones alternadas de
la misma sesión; no deben compararse con campañas históricas sin su referencia.

| Ruta | Resultado | Decisión |
| --- | --- | --- |
| CG de una reducción | G08: 59,38 → 56,23 ms, 256 → 249 iteraciones; G09: 2,944 → 3,196 s, 276 en ambos | Descartado como default general; el vector/recurrencia extra perjudica mallas grandes |
| Inverso acústico local SPD | G08: 60,95 → 342,46 ms, 256 → 1027 iteraciones | Descartado; positividad no implica buena aproximación del inverso global |
| Mezcla acústica 25% | G08: 61,92 → 106,16 ms, 256 → 263 | Descartada |
| Suavizado en métrica física | G08: 60,05 → 67,80 ms, 256 → 249 | Descartado; el coste adicional supera las iteraciones ahorradas |
| Aceptación con norma de coordenadas y norma física | G09: 2,949 → 2,974 s; sin beneficio. Casos con corrección: ahorro aparente cercano a 10% a rtol público 1e-5 | No promover: requiere controlar el efecto de resolver menos por debajo de la tolerancia |
| Doble norma, residual físico ≤5e-6 | Vf=5%, AR=10, 256³: 4,412 → 4,771 s, 300 → 302 iteraciones; referencia residual ≤4,99e-6 | Descartada como aceleración: a precisión comparable es aproximadamente 8% más lenta |

`cufhom_dual_metric_strict.py` es exclusivamente el control de precisión del
último piloto. No altera la auditoría de la ecuación original. El candidato a
1e-5 permite un residual más cercano a 1e-5 que la referencia con corrección;
por eso el ahorro aparente no se acepta como mejora matemática demostrada.

Las validaciones incluyen sistemas SPD densos y coordenadas pares/impares,
identidad de norma física y comparación de los seis campos completos.
Los campos no se presentan como idénticos cuando los residuos finales difieren.
El piloto acústico se validó con presupuesto de 1500 iteraciones para poder
terminar; sus mediciones de rendimiento usan el presupuesto original de 1000.

## Evidencia

- [Informe completo](../../benchmarks/performance_audit_20261001/PERFORMANCE_AUDIT.md).
- [Comparaciones y dispersión](../../benchmarks/performance_audit_20261001/experiment_statistics.csv).
- [Errores de campos, tensor y residuos](../../benchmarks/performance_audit_20261001/field_validation.csv).
- [Informes medidos comprimidos](../../benchmarks/performance_audit_20261001/evidence.json.gz).
- [Hashes y procedencia](../../benchmarks/performance_audit_20261001/evidence_manifest.json).

El cambio de política cuFFT por componentes utiliza una capacidad ya existente
del solver. Se midió aparte y produjo campos/tensor idénticos en 128³ y 192³,
con ahorro de 22% y 24%. Tampoco se cambió el default durante la auditoría.
