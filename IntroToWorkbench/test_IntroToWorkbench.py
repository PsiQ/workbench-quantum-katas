"""Interactive grading for the ``IntroToWorkbench`` notebook.

These exercises are graded from the notebook via the ``problem`` decorator (the
same pattern used elsewhere in this repo), not by pytest. Decorate your answer
with ``@problem``; it looks up the matching ``test_<name>`` function, runs it,
and prints ``Correct!`` or a hint.
"""
from __future__ import annotations

from re import T
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
            import Workbook_IntroToWorkbench as ref
    ref_available = True
except ImportError:
    ref_available = False
    # Skip all tests in this file - pytest checks reference solutions and that won't work without these imports
    pytestmark = mark.skip("No importnb/reference file available")

log_message = ""

def problem(arg):
    try:
        # Build test name
        test_name = "test_" + arg.__name__.lower()
        # Find test function; if none found, raise an exception
        test_func = globals()[test_name]
    except KeyError:
        print(f"Test {test_name} not found")
    else:
        try:
            test_func(arg)
        except Exception as e:
            print("Incorrect")
            if log_message != "":
                print(log_message)
            print(str(e))
        else:
            print("Correct!")
    
    return arg


####################################################################################################

@mark.parametrize("fun", [ref.instantiate_qpu] if ref_available else [])
def test_instantiate_qpu(fun):
    for num_qubits in range(1, 7):
        actual = fun(num_qubits)

        if not isinstance(actual, QPU):
            print(f"Testing {num_qubits} qubits")
            raise ValueError(f"Expected a QPU object, got {type(actual).__name__}.")

        if actual.num_qubits != num_qubits:
            print(f"Testing {num_qubits} qubits")
            raise ValueError(f"Expected {num_qubits} qubits, got {actual.num_qubits}.")


####################################################################################################

@mark.parametrize("fun", [ref.instantiate_qubits] if ref_available else [])
def test_instantiate_qubits(fun):
    for num_qubits in range(1, 7):
        qpu = QPU(num_qubits=num_qubits)
        actual = fun(qpu, num_qubits, f"reg{num_qubits}")

        if not isinstance(actual, Qubits):
            print(f"Testing qubit instantiation with {num_qubits} qubits")
            raise ValueError(f"Expected a Qubits object, got {type(actual).__name__}.")

        if actual.num_qubits != num_qubits:
            print(f"Testing qubit instantiation with {num_qubits} qubits")
            raise ValueError(f"Expected {num_qubits} qubits, got {actual.num_qubits}.")
        
        if actual.name != f"reg{num_qubits}":
            print(f"Testing qubit instantiation with {num_qubits} qubits")
            raise ValueError(f"Expected name 'reg{num_qubits}', got '{actual.name}'.")


####################################################################################################

@mark.parametrize("fun", [ref.apply_gates] if ref_available else [])
def test_apply_gates(fun):
    for qubit_size in range(1, 7):
        for angle in range(10, 180, 10):
            qpu = QPU(num_qubits=qubit_size)
            reg = Qubits(qubit_size, f"reg{qubit_size}", qpu=qpu)
            fun(reg, angle)

            target_mask = (1 << qubit_size) - 1
            found_insts = []
            expected_insts = [opcodes.OP_qc_x, opcodes.OP_qc_rx]

            for inst in qpu.get_instructions():
                if inst.opcode is opcodes.OP_qc_x or inst.opcode is opcodes.OP_qc_rx:
                    found_insts.append(inst)
                    continue
                
                if is_physical_operation(inst.opcode):
                    raise ValueError(f"Found an unexpected quantum operation {opcode_to_opname(inst.opcode)}.")


            if len(found_insts) != len(expected_insts):
                raise ValueError(f"Expected {len(expected_insts)} operations, got {len(found_insts)}.")

            for index, inst in enumerate(found_insts):
                if inst.opcode is not expected_insts[index]:
                    raise ValueError(f"Expected {opcode_to_opname(expected_insts[index])} operation, found {opcode_to_opname(inst.opcode)} operation.")

                if inst.target != target_mask:
                    target_str = str(bin(inst.target)[2:].zfill(qubit_size))
                    target_str_list = []
                    for i,s in enumerate(target_str[::-1]):
                        if s == '1':
                            target_str_list.append(str(i))
                    target_str = ", ".join(target_str_list)
                    raise ValueError(f"For {qubit_size} qubits, expected `x` gate applied to all qubits of the register, got a subset of qubits {target_str} instead.")

                if inst.opcode is opcodes.OP_qc_rx:
                    if inst.theta != angle:
                        raise ValueError(f"Expected `rx` gate with angle {angle}, got angle {inst.theta}.")


####################################################################################################

@mark.parametrize("fun", [ref.apply_controlled_gates] if ref_available else [])
def test_apply_controlled_gates(fun):
    qpu = QPU(num_qubits=3)
    reg1 = Qubits(1, f"reg1", qpu=qpu)
    reg2 = Qubits(1, f"reg2", qpu=qpu)
    fun(reg1, reg2)

    reg1_mask = 1
    reg2_mask = 2

    looking_for_cx = [(reg1_mask, reg2_mask, False), (reg1_mask, reg2_mask, True), (reg2_mask, reg1_mask, False)]
    found_insts = []
    for inst in qpu.get_instructions():
        if inst.opcode is not opcodes.OP_qc_x:
            if is_physical_operation(inst.opcode):
                raise ValueError(f"Found an unexpected quantum operation {opcode_to_opname(inst.opcode)}.")
            continue
        found_insts.append(inst)

    if len(found_insts) != len(looking_for_cx):
        raise ValueError(f"Expected {len(looking_for_cx)} cx gates, got {len(found_insts)}.")

    for index, inst in enumerate(found_insts):
        condition, target, negated = looking_for_cx[index]
        if inst.condition == condition and inst.target == target:
            if negated and inst.cond_xor == condition:
                continue
            elif not negated and inst.cond_xor == 0:
                continue

        negated_str = '==0' if negated else '==1'
        bad_negation = f'=={inst.cond_xor ^ 1}' if inst.cond_xor != 0 else '==1'
        condition_name = "reg1" if inst.condition & reg1_mask else "reg2"
        target_name = "reg1" if inst.target & reg1_mask else "reg2"
        expected_condition = "reg1" if condition & reg1_mask else "reg2"
        expected_target = "reg1" if target & reg1_mask else "reg2"
        raise ValueError(
            f"Expected controlled x gate on qubit condition {expected_condition}{negated_str} and target {expected_target}, "
            f"but found cx gate with condition {condition_name}{bad_negation} and target {target_name}."
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
        found_insts = []
        for inst in qpu.get_instructions():
            if inst.opcode is not opcodes.OP_qc_x:
                if is_physical_operation(inst.opcode):
                    print(f"Testing indexing and slicing with {i} qubits")
                    raise ValueError(f"Found an unexpected quantum operation {opcode_to_opname(inst.opcode)}.")
                continue
            found_insts.append(inst)
        
        if len(found_insts) != len(looking_for_x):
            print(f"Testing indexing and slicing with {i} qubits")
            raise ValueError(f"Expected {len(looking_for_x)} x gates, got {len(found_insts)}.")

        for index, inst in enumerate(found_insts):
            if inst.target != looking_for_x[index]:
                print(f"Testing indexing and slicing with {i} qubits")
                expected_str = str(bin(looking_for_x[index])[2:].zfill(i))
                expected_str_list = []
                for i,s in enumerate(expected_str[::-1]):
                    if s == '1':
                        expected_str_list.append(str(i))
                expected_str = ", ".join(expected_str_list)
                target_str = str(bin(inst.target)[2:].zfill(i))
                target_str_list = []
                for i,s in enumerate(target_str[::-1]):
                    if s == '1':
                        target_str_list.append(str(i))
                target_str = ", ".join(target_str_list)
                raise ValueError(f"Expected x gate on qubits {expected_str} of `reg`, got qubits {target_str} of `reg`.")


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
    found_insts = []
    for inst in qpu.get_instructions():
        if not is_physical_operation(inst.opcode):
            continue
    
        found_insts.append(inst)
    
    if len(found_insts) != len(looking_for_ops):
        raise ValueError(f"Expected {len(looking_for_ops)} operations, got {len(found_insts)}.")

    for index, inst in enumerate(found_insts):
        if inst.opcode != looking_for_ops[index][1]:
            raise ValueError(f"Found an unexpected quantum operation {opcode_to_opname(inst.opcode)} for operation {index}, expected {opcode_to_opname(looking_for_ops[index][1])}.")
        if inst.target != looking_for_ops[index][0]:
            reg_list = []
            if looking_for_ops[index][0] & 1:
                reg_list.append("reg1")
            if looking_for_ops[index][0] & 2:
                reg_list.append("reg2")
            if looking_for_ops[index][0] & 4:
                reg_list.append("reg3")
            expected_reg_str = ", ".join(reg_list)

            reg_list = []
            if inst.target & 1:
                reg_list.append("reg1")
            if inst.target & 2:
                reg_list.append("reg2")
            if inst.target & 4:
                reg_list.append("reg3")
            reg_str = ", ".join(reg_list)

            raise ValueError(
                f"Expected {opcode_to_opname(looking_for_ops[index][1])} gate on qubits {expected_reg_str}, found on {reg_str}."
            )


####################################################################################################

@mark.parametrize("fun", [ref.examine_state_vector] if ref_available else [])
def test_examine_state_vector(fun):
    for i in range(2, 5):
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
            raise ValueError(f"Expected maximum amplitudes {expected_amps}, got {amps}.")
        
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
            raise ValueError(f"Expected maximum amplitudes {expected_amps}, got {amps}.")


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
            target_str = str(bin(i.target)[2:].zfill(3))
            target_str_list = []
            for i,s in enumerate(target_str[::-1]):
                if s == '1':
                    target_str_list.append(str(i))
            target_str = ", ".join(target_str_list)
            raise ValueError(f"Expected read on qubit `reg`, got read on qubit subset {target_str}.")
        measure_found = True
    
    if not measure_found:
        raise ValueError('Expected read on qubit `reg`, not found.')

    if 0 not in result_dict or 7 not in result_dict:
        raise ValueError(f"Expected 0 and 7 in result dictionary, got {list(result_dict.keys())}.")

    if result_dict[0] + result_dict[7] != 1000:
        raise ValueError(
            f"Expected 1000 measurements on in result dictionary, got {result_dict[0] + result_dict[7]}."
        )


####################################################################################################

@mark.parametrize("qbk_class", [ref.GHZ] if ref_available else [])
def test_ghz(qbk_class):
    for i in range(3, 6):
        qpu = QPU(num_qubits=i)
        reg1 = Qubits(i, f"reg", qpu=qpu)
        qbk = qbk_class()
        qbk.compute(reg1)

        find = [opcodes.OP_qc_had] + [opcodes.OP_qc_x] * (i - 1)
        found_insts = []
        for inst in qpu.get_instructions():
            if not is_physical_operation(inst.opcode):
                continue
            found_insts.append(inst)

        if len(find) != len(found_insts):
            print(f"Testing GHZ with {i} qubits")
            raise ValueError(f"Expected {len(find)} operations, got {len(found_insts)}.")

        for index, inst in enumerate(found_insts):
            if inst.opcode is not find[index]:
                print(f"Testing GHZ with {i} qubits")
                raise ValueError(f"Expected {opcode_to_opname(find[index])} operation, found {opcode_to_opname(inst.opcode)} operation.")
        
        amps = qpu.pull_state()
        expected_amps = np.array([0.0]*2**i)
        expected_amps[0] = 1.0 / np.sqrt(2)
        expected_amps[2**i - 1] = 1.0 / np.sqrt(2)
        if not np.allclose(amps, expected_amps):
            print(f"Testing GHZ with {i} qubits")
            raise ValueError(f"Expected amplitudes {expected_amps}, got {amps}.")


####################################################################################################

@mark.parametrize("fun", [ref.apply_qubrick] if ref_available else [])
def test_apply_qubrick(fun):
    for i in range(3, 6):
        qpu = fun(i)

        find = [opcodes.OP_qc_had] + [opcodes.OP_qc_x] * (i - 1)
        qubrick_run = False
        found_insts = []
        for inst in qpu.get_instructions():
            if not is_physical_operation(inst.opcode):
                if inst.opcode is opcodes.OP_qc_qbk_compute_start:
                    qubrick_run = True
                continue
            found_insts.append(inst)

        if not qubrick_run:
            print(f"Testing GHZ Qubrick with {i} qubits")
            raise ValueError("Expected a qubrick to be used, not found.")

        if len(find) != len(found_insts):
            print(f"Testing GHZ Qubrick with {i} qubits")
            raise ValueError(f"Expected {len(find)} operations, got {len(found_insts)}.")
        
        for index, inst in enumerate(found_insts):
            if inst.opcode is not find[index]:
                print(f"Testing GHZ with {i} qubits")
                raise ValueError(f"Expected {opcode_to_opname(find[index])} operation, found {opcode_to_opname(inst.opcode)} operation.")
        
        amps = qpu.pull_state()
        expected_amps = np.array([0.0]*2**i)
        expected_amps[0] = 1.0 / np.sqrt(2)
        expected_amps[2**i - 1] = 1.0 / np.sqrt(2)
        if not np.allclose(amps, expected_amps):
            print("Testing GHZ Qubrick with {i} qubits")
            raise ValueError(f"Expected amplitudes {expected_amps}, got {amps}.")
