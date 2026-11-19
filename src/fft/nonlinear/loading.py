"""Validated piecewise-linear mixed loading in Mandel coordinates.

This module handles only six-component host controls, never voxel fields.
The scalar ``path`` API is intentionally left to the solver for compatibility.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MixedLoadingPath:
    """Fixed strain/stress mask with strictly increasing path parameter.

    ``strain`` is dimensionless Mandel strain; ``stress`` is Mandel stress in
    GPa. Only strain-controlled entries of strain and stress-controlled entries
    of stress are prescribed. The other entries are ignored and canonicalized
    to zero. If parameter is omitted, knots are equally spaced between 0 and 1.
    The parameter is a numerical path coordinate, not physical time.
    """

    strain_control: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    parameter: np.ndarray

    @classmethod
    def from_mapping(cls, loading):
        """Build an immutable mixed-loading path from validated host controls."""
        if not isinstance(loading, Mapping):
            raise ValueError("loading must be a mapping")
        required = {"strain_control", "strain", "stress"}
        if not required.issubset(loading):
            raise ValueError("loading requires strain_control, strain and stress")
        if set(loading) - required - {"parameter"}:
            raise ValueError(
                "Unknown loading fields: "
                + ", ".join(sorted(map(str, set(loading) - required - {"parameter"})))
            )
        mask = np.asarray(loading["strain_control"])
        # Do not silently turn numeric values, strings or NaNs into a mask.
        if mask.shape != (6,) or mask.dtype.kind != "b":
            raise ValueError("loading.strain_control must contain six booleans")
        try:
            strain = np.array(loading["strain"], dtype=np.float64, copy=True)
            stress = np.array(loading["stress"], dtype=np.float64, copy=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "loading strain/stress must be finite numeric (knots,6) arrays"
            ) from exc
        if (
            strain.ndim != 2
            or strain.shape[1:] != (6,)
            or len(strain) < 2
            or stress.shape != strain.shape
            or not np.isfinite(strain).all()
            or not np.isfinite(stress).all()
        ):
            raise ValueError(
                "loading strain/stress must be finite matching (knots,6) arrays with at least two knots"
            )
        try:
            parameter = np.array(
                loading.get("parameter", np.linspace(0.0, 1.0, len(strain))),
                dtype=np.float64,
                copy=True,
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "loading.parameter must be finite and strictly increasing"
            ) from exc
        if (
            parameter.shape != (len(strain),)
            or not np.isfinite(parameter).all()
            or not np.isfinite(np.diff(parameter)).all()
            or np.any(np.diff(parameter) <= 0)
        ):
            raise ValueError(
                "loading.parameter must match the knots and be finite and strictly increasing"
            )
        mask = mask.copy()
        strain[:, ~mask] = 0.0
        stress[:, mask] = 0.0
        for value in (mask, strain, stress, parameter):
            value.setflags(write=False)
        return cls(mask, strain, stress, parameter)

    def at(self, parameter):
        """Interpolate within one segment; return independent six-vectors.

        Exact knots are returned without interpolation roundoff. Extrapolation
        is prohibited, including for restart conditions.
        """
        parameter = float(parameter)
        if (
            not np.isfinite(parameter)
            or not self.parameter[0] <= parameter <= self.parameter[-1]
        ):
            raise ValueError("Loading parameter is outside the prescribed path")
        right = int(np.searchsorted(self.parameter, parameter, side="left"))
        if parameter == self.parameter[right]:
            return self.strain[right].copy(), self.stress[right].copy()
        left = right - 1
        fraction = (parameter - self.parameter[left]) / (
            self.parameter[right] - self.parameter[left]
        )
        return (
            (1.0 - fraction) * self.strain[left] + fraction * self.strain[right],
            (1.0 - fraction) * self.stress[left] + fraction * self.stress[right],
        )

    def descriptor(self):
        """Return a JSON-compatible description of the complete loading path."""
        return dict(
            strain_control=self.strain_control.tolist(),
            strain=self.strain.tolist(),
            stress=self.stress.tolist(),
            parameter=self.parameter.tolist(),
        )

    def fingerprint_configuration(self, integrator):
        """Invariant protocol, excluding future knots that may change on restart.

        The confirmed parameter and prescribed tensors are separately checked
        against the checkpoint. A continuation must start at those conditions.
        """
        return dict(
            kind="piecewise_linear_mixed",
            version=1,
            strain_control=self.strain_control.tolist(),
            integrator=integrator,
            state="confirmed_plastic_strain_Mandel_and_hardening",
            parameter="strictly_increasing_numerical_coordinate",
        )

    def validate_start(self):
        """Check restart controls against the path at the saved parameter."""
        if np.any(self.strain[0]) or np.any(self.stress[0]):
            raise ValueError(
                "General loading must begin with zero prescribed strain/stress; "
                "apply confinement in a segment from zero or restart a compatible checkpoint"
            )

    def validate_restart(self, metadata, history):
        """Check boundary conditions before trusting a confirmed GPU state."""
        if not isinstance(history, list) or not history:
            raise ValueError("Invalid checkpoint mixed loading history")
        last_parameter = None
        for confirmed in history:
            try:
                value = float(confirmed["target_parameter"])
                e = np.asarray(confirmed["target_strain_mandel"], dtype=float)
                s = np.asarray(confirmed["target_stress"], dtype=float)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Invalid checkpoint mixed loading history") from exc
            if (
                not np.isfinite(value)
                or e.shape != (6,)
                or s.shape != (6,)
                or not np.isfinite(e).all()
                or not np.isfinite(s).all()
                or np.any(e[~self.strain_control])
                or np.any(s[self.strain_control])
                or (last_parameter is not None and value <= last_parameter)
            ):
                raise ValueError("Invalid checkpoint mixed loading history")
            last_parameter = value
        try:
            saved = metadata["loading_state"]
            mask = np.asarray(saved["strain_control"])
            parameter = float(saved["parameter"])
            strain = np.asarray(saved["strain"], dtype=float)
            stress = np.asarray(saved["stress"], dtype=float)
            row = history[-1]
            row_parameter = float(row["target_parameter"])
            row_strain = np.asarray(row["target_strain_mandel"], dtype=float)
            row_stress = np.asarray(row["target_stress"], dtype=float)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("Invalid checkpoint mixed loading state") from exc
        if (
            mask.shape != (6,)
            or mask.dtype.kind != "b"
            or not np.array_equal(mask, self.strain_control)
            or strain.shape != (6,)
            or stress.shape != (6,)
            or not np.isfinite(strain).all()
            or not np.isfinite(stress).all()
            or not np.isfinite(parameter)
            or row_parameter != parameter
            or not np.array_equal(row_strain, strain)
            or not np.array_equal(row_stress, stress)
        ):
            raise ValueError("Checkpoint mixed loading state mismatch")
        # Retain the original parameter on continuation: rebasing it would
        # change predictor spacing and erase the meaning of history targets.
        tolerance = 32 * np.finfo(float).eps
        if not np.isclose(self.parameter[0], parameter, rtol=tolerance, atol=tolerance):
            raise ValueError("Loading must begin at the checkpoint target_parameter")
        if not np.allclose(
            self.strain[0], strain, rtol=tolerance, atol=tolerance
        ) or not np.allclose(self.stress[0], stress, rtol=tolerance, atol=tolerance):
            raise ValueError("Checkpoint prescribed strain/stress mismatch")
        return float(self.parameter[0])
