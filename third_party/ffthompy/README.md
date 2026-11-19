# FFTHomPy attribution

`core/` incorporates and modifies FFTHomPy by Jaroslav Vondřejc. Its original
[MIT license](../../core/LICENSE) and source notices are retained in the
repository and installed package. The internal namespace is `cufhom_core`.
CuFHom adds GPU FFTs, constitutive fields, CG and specialized kernels; renaming
the namespace does not change the origin of the incorporated code.

Upstream: [vondrejc/FFTHomPy](https://github.com/vondrejc/FFTHomPy).
The independent CPU comparator uses revision
`2c23c80dd2cab46f1090103e613b4f886b3daac7`, recorded in
the corresponding upstream repository.
It is installed separately and is not substituted by `cufhom_core` in benchmark
comparisons. CuFHom's own code is distributed under the root MIT license.
