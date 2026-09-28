"""Interactive grading for the ``IntroToWorkbench`` notebook.

These exercises are graded from the notebook via the ``problem`` decorator (the
same pattern used elsewhere in this repo), not by pytest. Decorate your answer
with ``@problem``; it looks up the matching ``test_<name>`` function, runs it,
and prints ``Correct!`` or a hint.
"""
from __future__ import annotations

from typing import Callable

import psiqdk.workbench.opcodes as opcodes
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.ops import opcode_to_opname
from psiqdk.workbench.ops._check_ops import is_physical_operation
from psiqdk.workbench.qre import resource_estimator
from pytest import mark

import numpy as np
import random as rd

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
        print(f"Test {test_name} not found.")
        return arg
    try:
        test_func(arg)
    except Exception as e:  # noqa: BLE001 - surface any failure as feedback
        print("Incorrect")
        print(str(e))
    else:
        print("Correct!")
    return arg


####################################################################################################

@mark.parametrize("fun", [ref.instantiate_qpu] if ref_available else [])
def test_instantiate_qpu(fun):
    for num_qubits in [1, 2, 3, 4, 5, 6]:
        actual = fun(num_qubits)

        if not isinstance(actual, QPU):
            print(f"Testing {num_qubits} qubits")
            raise ValueError(f"Expected a QPU, got {type(actual).__name__}.")

        if actual.num_qubits != num_qubits:
            print(f"Testing {num_qubits} qubits")
            raise ValueError(f"Expected {num_qubits} qubits, got {actual.num_qubits}.")


####################################################################################################

@mark.parametrize("fun", [ref.instantiate_qubits] if ref_available else [])
def test_instantiate_qubits(fun):
    for num_qubits in [1, 2, 3, 4, 5, 6]:
        qpu = QPU(num_qubits=num_qubits)
        actual = fun(qpu, num_qubits, f"reg{num_qubits}")

        if not isinstance(actual, Qubits):
            print(f"Testing Qubit Instantiation with {num_qubits} qubits")
            raise ValueError(f"Expected a Qubits, got {type(actual).__name__}.")

        if actual.num_qubits != num_qubits:
            print(f"Testing Qubit Instantiation with {num_qubits} qubits")
            raise ValueError(f"Expected {num_qubits} qubits, got {actual.num_qubits}.")
        
        if actual.name != f"reg{num_qubits}":
            print(f"Testing Qubit Instantiation with {num_qubits} qubits")
            raise ValueError(f"Expected reg{num_qubits} name, got {actual.name}.")


####################################################################################################

@mark.parametrize("fun", [ref.apply_gates] if ref_available else [])
def test_apply_gates(fun):
    angles = [i for i in range(10, 180, 10)]
    qubit_sizes = [1, 2, 3, 4, 5, 6]
    for qubit_size in qubit_sizes:
        for angle in angles:
            qpu = QPU(num_qubits=qubit_size)
            reg = Qubits(qubit_size, f"reg{qubit_size}", qpu=qpu)
            fun(reg, angle)

            found_x = False
            found_rx = False

            target_size = 0
            for i in range(qubit_size):
                target_size |= 1 << i

            for inst in qpu.get_instructions():
                if inst.opcode is opcodes.OP_qc_x:
                    if inst.target == target_size:
                        found_x = True
                    else:
                        print(f"Testing gate application with angle {angle} and qubit size {qubit_size}")
                        raise ValueError(f"Expected x gate on qubit mask {target_size}, got {inst.target}.")
                    continue

                if inst.opcode is opcodes.OP_qc_rx and found_x:
                    if inst.target == target_size:
                        found_rx = True
                    else:
                        raise ValueError(f"Expected rx gate on qubit mask {target_size}, got {inst.target}.")

                    if inst.theta != angle:
                        print(f"Testing gate application with angle {angle} and qubit size {qubit_size}")
                        raise ValueError(f"Expected rx gate with angle {angle}, got {inst.theta}.")
                    continue

                if is_physical_operation(inst.opcode):
                    print(f"Testing gate application with angle {angle} and qubit size {qubit_size}")
                    raise ValueError(f"Found an unexpected quantum operation {opcode_to_opname(inst.opcode)}.")

            if not found_x:
                raise ValueError("Expected x gate, but no x gate found.")
            if not found_rx:
                raise ValueError("Expected rx gate, but no rx gate found.")


####################################################################################################

@mark.parametrize("fun", [ref.apply_controlled_gates] if ref_available else [])
def test_apply_controlled_gates(fun):
    qpu = QPU(num_qubits=3)
    reg1 = Qubits(1, f"reg1", qpu=qpu)
    reg2 = Qubits(1, f"reg2", qpu=qpu)
    fun(reg1, reg2)

    found_list = [False, False, False]

    reg1_mask = 1
    reg2_mask = 2

    looking_for_cx = [(reg1_mask, reg2_mask, False), (reg1_mask, reg2_mask, True), (reg2_mask, reg1_mask, False)]
    looking_for_index = 0
    seen_cxs = 0
    for inst in qpu.get_instructions():
        if inst.opcode is not opcodes.OP_qc_x:
            if is_physical_operation(inst.opcode):
                raise ValueError(f"Found an unexpected quantum operation {opcode_to_opname(inst.opcode)}.")
            continue
        if looking_for_index >= len(looking_for_cx):
            raise ValueError("Found more cx gates than expected.")
        seen_cxs += 1
        condition, target, negated = looking_for_cx[looking_for_index]
        if inst.condition == condition and inst.target == target:
            if negated and inst.cond_xor == condition:
                found_list[looking_for_index] = True
                looking_for_index += 1
            elif not negated and inst.cond_xor == 0:
                found_list[looking_for_index] = True
                looking_for_index += 1
            else:
                if negated:
                    bad_negation = 'negated' if inst.cond_xor != 0 else 'non-negated'
                    raise ValueError(
                        f"Expected negated controlled x gate on qubit mask {condition} and {target}, "
                        f"but found {bad_negation} cx gate with condition {inst.cond_xor} and target {inst.target}."
                    )
                else:
                    bad_negation = 'negated' if inst.cond_xor != 0 else 'non-negated'
                    raise ValueError(
                        f"Expected controlled x gate on qubit mask {condition} and {target}, "
                        f"but found {bad_negation} cx gate with condition {inst.cond_xor} and target {inst.target}."
                    )

@mark.parametrize("fun", [ref.indexing_and_slicing] if ref_available else [])
def test_indexing_and_slicing(fun):
    for i in range(5, 10):
        qpu = QPU(num_qubits=i)
        reg = Qubits(i, f"reg", qpu=qpu)
        fun(reg)

        reg1_mask = 2
        reg2_mask = 0
        for j in range(1, i):
            if j % 2 == 0:
                reg2_mask |= 1 << j

        looking_for_x = [reg1_mask, reg2_mask]
        looking_for_index = 0
        for inst in qpu.get_instructions():
            if inst.opcode is not opcodes.OP_qc_x:
                if is_physical_operation(inst.opcode):
                    print(f"Testing indexing and slicing with {i} qubits")
                    raise ValueError(f"Found an unexpected quantum operation {opcode_to_opname(inst.opcode)}.")
                continue
            if looking_for_index >= len(looking_for_x):
                print(f"Testing indexing and slicing with {i} qubits")
                raise ValueError("Found more cx gates than expected.")
            if inst.target == looking_for_x[looking_for_index]:
                looking_for_index += 1
            else:
                print(f"Testing indexing and slicing with {i} qubits")
                raise ValueError(
                    f"Expected x gate on qubit mask {looking_for_x[looking_for_index]}, got {inst.target}."
                )
        if looking_for_index != len(looking_for_x):
            print(f"Testing indexing and slicing with {i} qubits")
            raise ValueError(f"Expected {len(looking_for_x)} x gates, got {looking_for_index}.")


####################################################################################################

@mark.parametrize("fun", [ref.concatenate_registers] if ref_available else [])
def test_concatenate_registers(fun):
    qpu = QPU(num_qubits=3)
    reg1 = Qubits(1, f"reg1", qpu=qpu)
    reg2 = Qubits(1, f"reg2", qpu=qpu)
    reg3 = Qubits(1, f"reg3", qpu=qpu)
    fun(reg1, reg2, reg3)

    check_1_mask = 5
    check_2_mask = 6
    check_3_mask = 7

    looking_for_ops = [(check_1_mask, opcodes.OP_qc_x), (check_2_mask, opcodes.OP_qc_y), (check_3_mask, opcodes.OP_qc_z)]
    looking_for_index = 0
    for inst in qpu.get_instructions():
        if looking_for_index >= len(looking_for_ops):
            raise ValueError("Found more operations than expected.")
        if not is_physical_operation(inst.opcode):
            continue
        if is_physical_operation(inst.opcode) and inst.opcode != looking_for_ops[looking_for_index][1]:
            raise ValueError(f"Found an unexpected quantum operation {opcode_to_opname(inst.opcode)}.")
        if inst.opcode is not looking_for_ops[looking_for_index][1]:
            raise ValueError(
                f"Expected {opcode_to_opname(looking_for_ops[looking_for_index][1])} gate on qubit mask "
                f"{looking_for_ops[looking_for_index][0]}, got {opcode_to_opname(inst.opcode)}."
            )
        if inst.target != looking_for_ops[looking_for_index][0]:
            raise ValueError(
                f"Expected {opcode_to_opname(looking_for_ops[looking_for_index][1])} gate on qubit mask "
                f"{looking_for_ops[looking_for_index][0]}, got {inst.target}."
            )
        looking_for_index += 1
    if looking_for_index != len(looking_for_ops):
        raise ValueError(f"Expected {len(looking_for_ops)} operations, got {looking_for_index}.")


####################################################################################################

@mark.parametrize("fun", [ref.examine_state_vector] if ref_available else [])
def test_examine_state_vector(fun):
    for i in [2, 3, 4]:
        qpu = QPU(num_qubits=i)
        reg = Qubits(i, f"reg", qpu=qpu)

        rd.seed(0xC0FFEE)

        arr = np.array([0]*2**i)
        val = rd.randint(0, 2**i - 1)
        arr[val] = 1

        expected_amps = [val]

        reg.push_state(arr)

        amps = fun(qpu)

        if len(amps) != len(expected_amps):
            raise ValueError(f"Expected {len(expected_amps)} amplitudes, got {len(amps)}.")

        if not np.allclose(amps, expected_amps):
            raise ValueError(f"Expected maximum amplitudes {arr}, got {amps}.")
        
        qpu = QPU(num_qubits=i)
        reg = Qubits(i, f"reg", qpu=qpu)

        num_equal = rd.randint(2, 2**i - 1)
        expected_amps = rd.sample(range(2**i), num_equal)
        amp = 1 / np.sqrt(num_equal)

        arr = np.array([0.0]*2**i)
        for index in expected_amps:
            arr[index] = amp
        
        reg.push_state(arr)

        amps = fun(qpu)

        if len(amps) != len(expected_amps):
            raise ValueError(f"Expected {len(expected_amps)} amplitudes, got {len(amps)}.")

        if not np.allclose(amps, expected_amps):
            raise ValueError(f"Expected maximum amplitudes {arr}, got {amps}.")


####################################################################################################

@mark.parametrize("fun", [ref.apply_and_use_measurements] if ref_available else [])
def test_apply_and_use_measurements(fun):
    qpu = QPU(num_qubits=3)
    count = [0]
    
    def fun_to_run() -> Qubits:
        count[0] += 1
        qpu.reset(num_qubits=3)
        reg = Qubits(3, f"reg", qpu=qpu)
        reg[0].had()
        reg[1].x(reg[0])
        reg[2].x(reg[0])
        return reg

    result_dict = fun(fun_to_run)

    if count[0] != 1000:
        raise ValueError(f"Expected 1000 runs of the function, got {count[0]}.")

    measure_found = False
    for i in qpu.get_instructions():
        if i.opcode is not opcodes.OP_qc_read:
            continue
        if i.target != 7:
            raise ValueError(f"Expected read on qubit mask 7, got {i.target}.")
        measure_found = True
    
    if not measure_found:
        raise ValueError('Expected read on qubit "reg", not found.')

    if 0 not in result_dict or 7 not in result_dict:
        raise ValueError(f"Expected 0 and 7 in result dictionary, got {list(result_dict.keys())}.")

    if result_dict[0] + result_dict[7] != 1000:
        raise ValueError(
            f"Expected 1000 measurements on in result dictionary, got {result_dict[0] + result_dict[7]}."
        )


####################################################################################################

@mark.parametrize("fun", [ref.GHZ] if ref_available else [])
def test_ghz(qbk_class):
    for i in range(3, 6):
        qpu = QPU(num_qubits=i)
        reg1 = Qubits(i, f"reg", qpu=qpu)
        qbk = qbk_class()
        qbk.compute(reg1)

        find = [opcodes.OP_qc_x] * (i - 1) + [opcodes.OP_qc_had]
        for inst in qpu.get_instructions():
            if not is_physical_operation(inst.opcode):
                continue
            if inst.opcode is not find[-1]:
                print("Testing GHZ with {i} qubits")
                raise ValueError(f"Found an unexpected quantum operation {opcode_to_opname(inst.opcode)}.")
            
            find = find[:-1]
        
        amps = qpu.pull_state()
        expected_amps = np.array([0.0]*2**i)
        expected_amps[0] = 1.0 / np.sqrt(2)
        expected_amps[2**i - 1] = 1.0 / np.sqrt(2)
        if not np.allclose(amps, expected_amps):
            print(f"Testing GHZ with {i} qubits")
            raise ValueError(f"Expected amplitudes {expected_amps}, got {amps}.")

        if len(find) != 0:
            print(f"Testing GHZ with {i} qubits")
            raise ValueError(f"Expected {i} operations, got {i - len(find)}.")


####################################################################################################

@mark.parametrize("fun", [ref.apply_qubrick] if ref_available else [])
def test_apply_qubrick(fun):
    for i in range(2, 6):
        qpu = fun(i + 1)

        find = [opcodes.OP_qc_x] * i + [opcodes.OP_qc_had]
        qubrick_run = False
        for inst in qpu.get_instructions():
            if not is_physical_operation(inst.opcode):
                if inst.opcode is opcodes.OP_qc_qbk_compute_start:
                    qubrick_run = True
                continue
            if inst.opcode is not find[-1]:
                print("Testing GHZ Qubrick with {i} qubits")
                raise ValueError(f"Found an unexpected quantum operation {opcode_to_opname(inst.opcode)}.")
            
            find = find[:-1]

        if not qubrick_run:
            print("Testing GHZ Qubrick with {i} qubits")
            raise ValueError("Expected a qubrick to be used, not found.")

        if len(find) != 0:
            print("Testing GHZ Qubrick with {i} qubits")
            raise ValueError(f"Expected {i + 1} operations, got {i + 1 - len(find)}.")
        
        amps = qpu.pull_state()
        expected_amps = np.array([0.0]*2**(i + 1))
        expected_amps[0] = 1.0 / np.sqrt(2)
        expected_amps[2**(i + 1) - 1] = 1.0 / np.sqrt(2)
        if not np.allclose(amps, expected_amps):
            print("Testing GHZ Qubrick with {i} qubits")
            raise ValueError(f"Expected amplitudes {expected_amps}, got {amps}.")
