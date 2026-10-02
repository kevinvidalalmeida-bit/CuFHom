# Alcance cerrado, 2 de octubre de 2026

- [x] Revisar el código de CG, mapas, material, proyección y posprocesado.
- [x] Capturar cuatro trazas warm de los predeterminados A/B anteriores.
- [x] Construir el espectro completo y resolver seis cargas de cinco casos
      CPU64 para cribar la compliance fina proyectada.
- [x] Derivar `H = I + Q_R* (C-C₀) Q_R`, implementar el split GPU y su
      variante de norma recurrente y medir seis casos con cinco repeticiones.
- [x] Comparar sus seis campos completos, tensor anisótropo SPD, contraste
      negativo y referencia CPU64 pequeña.
- [x] Medir la pérdida de precisión para fases blandas. Retirar ambos
      pilotos de la ruta general predeterminada y conservar su evidencia.
- [x] Implementar la reutilización de deformación/tensión de la última
      comprobación verdadera de CG mediante interfaces explícitas.
- [x] Medir seis casos con cinco pares, comparar todos los campos y añadir
      controles de campo inicial, falsa convergencia y operador distinto.
- [x] Verificar 345 pruebas y 7 subpruebas de producción y 13 del piloto.
- [x] Empaquetar fuentes, hashes y logs; verificar reproducción offline de
      los once artefactos byte a byte.

Las extensiones a escalares residentes en GPU, espectro de tensión retenido
y Ceff por tensiones medias tienen condiciones y prioridades documentadas.
No se presentan como cambios implementados ni como ganancias medidas.
