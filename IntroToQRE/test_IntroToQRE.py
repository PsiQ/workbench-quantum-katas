"""Interactive grading for the ``FiltersAndResourceEstimation`` notebook.

These exercises are graded from the notebook via the ``problem`` decorator (the
same pattern used elsewhere in this repo), not by pytest. Decorate your answer
with ``@problem``; it looks up the matching ``test_<name>`` function, runs it,
and prints ``Correct!`` or a hint.
"""
from __future__ import annotations

from typing import Callable

import psiqdk.workbench.opcodes as opcodes
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.qre import resource_estimator
from pytest import mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_IntroToQRE as ref
    ref_available = True
except ImportError:
    ref_available = False
    # Skip all tests in this file - pytest checks reference solutions and that won't work without these imports
    pytestmark = mark.skip("No importnb/reference file available")


def problem(arg):
    """Run ``test_<arg.__name__>`` against ``arg`` and print a verdict."""
    test_name = "test_" + arg.__name__.lower()
    test_func = globals().get(test_name)
    if test_func is None:
        print(f"Test {test_name} not found")
        return arg
    try:
        test_func(arg)
    except Exception as e:  # noqa: BLE001 - surface any failure as feedback
        print("Incorrect")
        print(str(e))
    else:
        print("Correct!")
    return arg


#####################################################################################################################################
# Exercise: count the resources manually without any decompositions

@mark.parametrize("fun", [ref.count_raw_gates] if ref_available else [])
def test_count_raw_gates(fun):
    actual = fun()

    if not isinstance(actual, dict):
        raise ValueError(f"expected a dict, got {type(actual).__name__}")

    if len(actual) != 2:
        raise ValueError(f"Incorrect number of items in the dictionary: expected 2, got {len(actual)}")

    actual_ts = actual.get("t_gates")
    if actual_ts != 1:
        raise ValueError(f"Incorrect number of T gates: expected 1, got {actual_ts}")

    actual_rotations = actual.get("rotations")
    if actual_rotations != 3 and actual_rotations != 4:
        raise ValueError(f"Incorrect number of rotations: expected 3 or 4, got {actual_rotations}")



#####################################################################################################################################
# Exercise: extract only the resources that matter

def _reference_circuit() -> QPU:
    """A small circuit distinct from the notebook's own, for grading."""
    qpu = QPU(num_qubits=8)
    reg = Qubits(6, "reg", qpu)
    reg[0].t()
    reg[1].rz(30, reg[2])
    reg[3].x(reg[4:6])
    reg[0:3].rz(45)
    return qpu


@mark.parametrize("fun", [ref.fetch_relevant_resources] if ref_available else [])
def test_fetch_relevant_resources(fun: Callable[[QPU], dict]) -> None:
    relevant_keys = ["t_gates", "toffs", "rotations", "qubit_highwater"]

    qpu = _reference_circuit()
    result = fun(qpu)

    if not isinstance(result, dict):
        raise ValueError(f"Expected a dictionary, got {type(result).__name__}")

    extra = set(result) - set(relevant_keys)
    if extra:
        raise ValueError(
            f"The dictionary should have exactly the keys {relevant_keys}; "
            f"your solution's return has unexpected key(s) {sorted(extra)}"
        )

    missing = set(relevant_keys) - set(result)
    if missing:
        raise ValueError(
            f"The dictionary should have exactly the keys {relevant_keys}; "
            f"your solution's return has missing key(s) {sorted(missing)}"
        )

    expected = resource_estimator(qpu).resources()
    for key in relevant_keys:
        if result[key] != expected[key]:
            raise ValueError(f"Incorrect value for {key!r}: expected {expected[key]}, got {result[key]}")


#####################################################################################################################################
# Exercise: apply rotation synthesis

def _t_gates_per_qubit(qpu: QPU) -> list[int]:
    """Count the T gates landing on each qubit in the (already compiled) circuit."""
    counts = [0, 0]
    for op in qpu.get_instructions():
        if getattr(op, "opcode", None) != opcodes.OP_qc_t:
            continue
        target = getattr(op, "target", 0)
        if target and (target & (target - 1)) == 0:  # single-qubit -> one-hot mask
            qubit = target.bit_length() - 1
            counts[qubit] += 1
    return counts


@mark.parametrize("fun", [ref.rs_synth_circuit] if ref_available else [])
def test_rs_synth_circuit(fun: Callable[[], QPU]) -> None:
    qpu = fun()

    if not isinstance(qpu, QPU):
        raise ValueError(f"Expected the function to return a QPU, got {type(qpu).__name__}")

    if qpu.get_filter_by_name(">>rs-synth-filter>>") is None:
        raise ValueError(
            "'>>rs-synth-filter>>' is not in the list of the QPU's filters"
        )

    # rs-synth turns every rotation into Clifford+T, so none should be left.
    if resource_estimator(qpu).resources().get("rotations") != 0:
        raise ValueError(
            'The circuit still has un-synthesized rotations. Did you add '
            '">>rs-synth-filter>>" to the QPU\'s pre_filters?'
        )

    t_per_qubit = _t_gates_per_qubit(qpu)
    if t_per_qubit[0] != 1:
        raise ValueError(
            "Qubit 0 should carry an Rz(45°) (compiled to exactly one T gate) but has "
            f"{t_per_qubit[0]} T gates; did you put the right rotation(s) there?"
        )
    if t_per_qubit[1] < 2:
        raise ValueError(
            "Qubit 1 should carry an off-grid Rz(22.5°) that synthesizes into several "
            f"T gates, but has {t_per_qubit[1]} T gates; did you put the right rotation(s) there?"
        )

    qpu.draw()


#####################################################################################################################################
# Exercise: count the resources off the filtered diagram

def _walkthrough_filtered_circuit() -> QPU:
    """The same circuit + filters as the notebook's exercise."""
    qpu = QPU(
        num_qubits=8,
        pre_filters=[
            ">>clean-ladder-filter>>",
            ">>single-control-filter>>",
        ],
    )
    reg = Qubits(6, "reg", qpu)
    reg[4].t()
    reg[5].s(reg[4])
    reg[0].rz(123, reg[1:3])
    reg[2].x(reg[3:6])
    reg[1].x(reg[3:6])
    reg[0:3].rz(45)
    return qpu


@mark.parametrize("fun", [ref.count_compiled_gates] if ref_available else [])
def test_count_compiled_gates(fun: Callable[[], dict]) -> None:
    result = fun()

    if not isinstance(result, dict):
        raise ValueError(f"Expected a dictionary, got {type(result).__name__}")

    expected = resource_estimator(_walkthrough_filtered_circuit()).resources()
    walkthrough_keys = ["t_gates", "toffs", "rotations", "gidney_lelbows"]

    for key in walkthrough_keys:
        if key not in result:
            raise ValueError(f"Return dictionary missing the {key!r} count")
        if result[key] != expected[key]:
            raise ValueError(f"Incorrect value for {key!r}: you counted {result[key]}, the circuit has {expected[key]}")
