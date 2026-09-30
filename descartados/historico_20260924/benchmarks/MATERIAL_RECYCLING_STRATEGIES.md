# Reutilizar simulaciones de materiales: siguiente desarrollo

Alcance: homogenización **lineal**, geometría G09 fija, seis cargas Mandel,
misma discretización y tolerancia. No reutilizar estados plásticos entre
materiales como si fueran el mismo problema no lineal.

## Lo implementado

`initial_solution_fields` ya funciona en el solver de coordenadas rápidas.
Se proyecta el snapshot físico con el proyector compatible original y se
resuelve la corrección con **C del nuevo material**, no con la rigidez anterior.
La aceptación usa el residuo de las ecuaciones originales. La escala de
tolerancia es la del RHS completo, no la del pequeño residuo inicial.
Se descarta una inicialización peor que cero. El código no altera los campos
de entrada, el orden de candidatos, los criterios de enriquecimiento ni el ROM.

Esto habilita predictores mejores, pero copiar el anterior no garantiza
ahorro total. El benchmark `benchmark_material_sequence.py` incluye descarga,
copia, reordenamiento y recuperación de los campos, además de CG.

## Prioridad 1: predictor Galerkin con los operadores ya compilados

La campaña conserva snapshots compatibles en `basis.active_fields` y construye
`Kq`, `Bq`, `Dq`. `scripts/rom_reduced_operator.py::_rom_ceff` ya devuelve
amplitudes además del tensor efectivo. Actualmente esas amplitudes no se usan
para inicializar el siguiente FFT en `append_sobol_batch`.

Para el nuevo material mu, los siete coeficientes constitutivos theta permiten:

```text
K(mu) = sum_q theta_q(mu) Kq
B(mu) = sum_q theta_q(mu) Bq
K(mu) a = -B(mu)                  # r x r, seis RHS, float64
e_predictor = V a                 # campos compatibles, seis cargas
FFT: A(mu) delta = b(mu) - A(mu) e_predictor
e_final = e_predictor + delta
```

La combinación es óptima en energía dentro del espacio V si se resuelve el
problema reducido SPD exactamente. Si V contiene el snapshot anterior, no
tiene mayor error energético que ese snapshot. **No** implica que cualquier
reconstrucción resulte más rápida ni una cota universal de iteraciones CG.
La proyección reducida como ayuda a un solver iterativo está estudiada por
[Nguyen y Chen, 2018](https://arxiv.org/abs/1804.06363).

Integración propuesta, todavía no implementada:

- Antes de `ensure_snapshot`, evaluar el material usando **sólo el prefijo ya
  resuelto**; nunca usar el ROM congelado que incluye materiales futuros.
- Reconstruir por bloques usando las coordenadas físicas correctas de V.
  Aplicar cualquier transformación de base/QR y la permutación inversa de
  vóxeles. No pasar directamente un campo ordenado por fases al FFT.
- Reutilizar Kq/Bq ya disponibles; no volver a construirlos sólo para el
  predictor. Si falta un prefijo completo o el sistema reducido falla,
  conservar el arranque anterior/cero y registrar el motivo.
- Medir reconstrucción, tráfico y FFT corregido juntos. No sustituir la
  auditoría FOM por una precisión sólo del tensor efectivo ROM.

## Prioridad 2: CG aumentado con un subespacio reciclado pequeño

El predictor actúa una vez. Un método aumentado mantiene durante CG la
corrección respecto a un espacio U de simulaciones/direcciones previas.
Para A restringido al espacio compatible:

```text
Ec = U* A_new U
x_coarse = U solve(Ec, U* b_new)
P = I - U solve(Ec, U* A_new)
resolver P* A_new P y = P* b_new en el complemento admisible
x = x_coarse + P y
```

No formar P denso. Actualizar A_new U y Ec al cambiar el material: las
direcciones del CG anterior no siguen siendo conjugadas automáticamente.
El operador proyectado es semidefinido en el espacio completo; CG debe
trabajar en su complemento definido positivo. Reproyectar y auditar por
redondeo; no aplicar CG ordinario indiscriminadamente al sistema singular.
Esta familia y el control del tamaño del espacio reciclado se desarrollan en
[Carlberg, Forstall y Tuminaro, 2016, §2.2–4](https://arxiv.org/html/1512.05820v2).

Propuesta para nuestra implementación, no un resultado medido: empezar con
6–12 vectores auxiliares, limitados por memoria, probar representación de
desplazamientos y reutilizarla en las seis cargas. Comprimir **sólo la ayuda
del solver**, no la base científica completa ni la malla. Comparar predictor
solo frente a predictor+CG aumentado para saber si la corrección por iteración
compensa su coste.

## Otras opciones y precauciones

- Las matrices pequeñas afines permiten amortizar los proyectores reciclados;
  véase [Panagiotopoulos et al., 2023](https://doi.org/10.1016/j.cma.2022.115765).
  Su benchmark no es nuestro FFT de fibras TI; sus ganancias no se transfieren
  automáticamente.
- La deflación espectral pura puede ayudar si hay modos lentos bien separados.
  El diagnóstico local de Ritz de G09 no mostró unos pocos valores extremos
  claramente aislados: priorizar información de soluciones/residuos, sin
  interpretar ese diagnóstico como un espectro completo certificado.
- Un predictor tangente en parámetros necesita sensibilidades o un modelo
  secante. No pagar ahora 42 problemas adicionales por material sin demostrar
  amortización; Kq/Bq y los snapshots ya disponibles son el primer recurso.

Aceptación de cualquier candidato: mismo orden/prefijo de materiales, mismo
residuo físico, comparación de Ceff y campos, memoria máxima, CG y **tiempo
total por secuencia** con preparación y transporte incluidos. No extrapolar
una reducción de iteraciones a una aceleración de tiempo.
