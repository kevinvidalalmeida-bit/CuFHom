"""CLI dispatch and rejection coverage without importing CuPy or solving."""

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def invoke(monkeypatch, tmp_path):
    calls = []

    def fake_solve(parameters):
        calls.append(parameters)
        parameters["on_step"](
            dict(
                strain=[0.0] * 6,
                stress=[0.0] * 6,
                newton_iterations=1,
                cg_iterations=3,
                seconds=0.0,
                residual=0.0,
                plastic_fraction=0.0,
            )
        )
        return dict(wall_seconds=0.0, history=[dict(residual=0.0)], rejected_steps=[])

    # Stub the entire package chain before runpy so no solver/material/GPU
    # imports occur. Only the pure NumPy curve conversions are loaded.
    names = (
        "cufhom",
        "cufhom.fft",
        "cufhom.fft.nonlinear",
        "cufhom.fft.nonlinear.curves",
    )
    modules = {name: ModuleType(name) for name in names}
    modules["cufhom"].fft = modules["cufhom.fft"]
    modules["cufhom.fft"].nonlinear = modules["cufhom.fft.nonlinear"]
    modules["cufhom.fft.nonlinear"].curves = modules["cufhom.fft.nonlinear.curves"]
    modules["cufhom.fft"].solve_nonlinear_homogenization = fake_solve
    modules["cufhom.fft.nonlinear.curves"].__dict__.update(
        runpy.run_path(str(ROOT / "src/fft/nonlinear/curves.py"))
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    main = runpy.run_path(str(ROOT / "src/cli/nonlinear.py"))["main"]

    def call(*args):
        monkeypatch.setattr(
            sys,
            "argv",
            ["cufhom-nonlinear", "--output", str(tmp_path / "output"), *args],
        )
        main()
        return calls

    return call, calls


@pytest.mark.parametrize(
    "args,model,integrator",
    [
        ((), "j2", "midpoint"),
        (("--model", "j2", "--integrator", "backward_euler"), "j2", "backward_euler"),
        (("--model", "drucker_prager"), "drucker_prager", "backward_euler"),
    ],
)
def test_model_dispatch_preserves_backend_default(invoke, args, model, integrator):
    call, _ = invoke
    parameters = call(*args)[0]
    assert parameters["model"] == model
    assert parameters["integrator"] == integrator
    assert parameters["backend"] == "scipy"
    if model == "drucker_prager":
        assert parameters["matrix"]["dilatancy"] is None
    else:
        assert "friction" not in parameters["matrix"]


@pytest.mark.parametrize(
    "args,message",
    [
        (("--friction", ".3"), "require --model drucker_prager"),
        (("--model", "j2", "--dilatancy", "0"), "require --model drucker_prager"),
        (
            ("--model", "drucker_prager", "--integrator", "midpoint"),
            "requires --integrator backward_euler",
        ),
        (("--model", "drucker_prager", "--friction", "nan"), "requires finite"),
        (("--model", "drucker_prager", "--friction", "3"), "requires finite"),
        (("--model", "drucker_prager", "--dilatancy", ".4"), "requires finite"),
        (("--case", "geometry"), "requires explicit --geometry and --material"),
        (("--material", "unused.json"), "only used with --case geometry"),
    ],
)
def test_invalid_model_options_never_reach_solver(invoke, capsys, args, message):
    call, calls = invoke
    with pytest.raises(SystemExit) as exc:
        call(*args)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not calls


def test_external_material_preserved_unless_explicitly_overridden(
    invoke, tmp_path, capsys
):
    geometry = tmp_path / "geometry.npz"
    np.savez(
        geometry, phase=np.zeros((2, 2, 2), dtype=np.uint8), ori=np.zeros((2, 2, 2, 3))
    )
    material = ROOT / "examples/material_drucker_prager.json"
    before = material.read_bytes()
    call, _ = invoke
    common = (
        "--case",
        "geometry",
        "--geometry",
        str(geometry),
        "--material",
        str(material),
        "--model",
        "drucker_prager",
    )
    parameters = call(*common)[0]
    assert parameters["matrix"] == json.loads(before)["matrix"]
    first_output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert first_output[0]["linear_solver"] == "bicgstab"
    assert first_output[0]["krylov"] == 3 and "cg" not in first_output[0]
    parameters = call(*common, "--dilatancy", ".3")[-1]
    assert parameters["matrix"]["dilatancy"] == 0.3
    assert parameters["matrix"]["friction"] == 0.3
    assert material.read_bytes() == before
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[0]["linear_solver"] == "cg"


def test_declared_material_model_mismatch_rejected_before_geometry_read(invoke, capsys):
    call, calls = invoke
    with pytest.raises(SystemExit) as exc:
        call(
            "--case",
            "geometry",
            "--model",
            "j2",
            "--geometry",
            "missing.npz",
            "--material",
            str(ROOT / "examples/material_drucker_prager.json"),
        )
    assert exc.value.code == 2 and not calls
    assert "disagrees with --model" in capsys.readouterr().err


def test_existing_j2_geometry_command_keeps_default(invoke, tmp_path):
    geometry = tmp_path / "geometry.npz"
    np.savez(geometry, phase=np.zeros((2, 2, 2), np.uint8), ori=np.zeros((2, 2, 2, 3)))
    material = tmp_path / "j2.json"
    material.write_text(
        json.dumps(
            dict(
                matrix=dict(E=3.0, nu=0.3, yield_stress=0.015, hardening=0.03), fiber={}
            )
        )
    )
    call, _ = invoke
    parameters = call(
        "--case", "geometry", "--geometry", str(geometry), "--material", str(material)
    )[0]
    assert parameters["model"] == "j2" and parameters["integrator"] == "midpoint"


def test_damage_and_general_loading_dispatch(invoke):
    call, _ = invoke
    p = call(
        "--model",
        "drucker_prager",
        "--backend",
        "cupy",
        "--damage",
        str(ROOT / "examples/material_gradient_damage.json"),
        "--loading",
        str(ROOT / "examples/loading_nonproportional.json"),
        "--loading-increment",
        ".2",
        "--lengths",
        "1",
        "2",
        "3",
    )[0]
    assert p["damage"]["model"] == "implicit_gradient"
    assert p["loading"]["strain_control"] == [True, False, False, False, False, True]
    assert p["initial_increment"] == 0.2 and "path" not in p
    assert p["lengths"] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "args,message",
    [
        (
            (
                "--model",
                "drucker_prager",
                "--damage",
                str(ROOT / "examples/material_gradient_damage.json"),
            ),
            "requires --model drucker_prager --backend cupy",
        ),
        pytest.param(
            (
                "--backend",
                "cupy",
                "--damage",
                str(ROOT / "examples/material_gradient_damage.json"),
            ),
            "requires --model drucker_prager --backend cupy",
            marks=pytest.mark.gpu,
        ),
        (
            (
                "--loading",
                str(ROOT / "examples/loading_nonproportional.json"),
                "--unload",
            ),
            "--loading requires one display component",
        ),
    ],
)
def test_invalid_damage_or_mixed_cli_contract(invoke, capsys, args, message):
    call, calls = invoke
    with pytest.raises(SystemExit):
        call(*args)
    assert not calls
    assert message in capsys.readouterr().err
