from itertools import product
from math import sqrt, cos, sin
from random import seed, uniform
from typing import Callable, Sequence
from psiqdk.workbench import QPU, Qubits
from pytest import approx, mark
from warnings import catch_warnings
from functools import partial

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_ArbitraryStatePreparation as ref
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


def check_state_vector(
    fun: Callable,                                # Callable that is being tested
    n_qubits: int,                                # Number of qubits in the main register
    expected_vector: Sequence[complex | float],   # State vector it should prepare
    *,
    n_aux: int=0,                                 # Number of auxiliary qubit(s) the solution is allowed to use
    initial_vector: Sequence[float] | None = None # The initial state (before the solution is called)
) -> None:
    # Construct the qpu and register
    total_num_qubits = n_qubits + n_aux   # Will be unnecessary once auto-resize is in
    qpu = QPU(num_qubits=total_num_qubits)
    qpu.enable_qubit_allocation_debugging()
    reg = Qubits(n_qubits, 'reg', qpu)

    if initial_vector is not None:
        reg.push_state(initial_vector)

    fun(reg)

    # Extract the state vector
    actual_vector = qpu.pull_state()

    # Check that there are no leftover auxiliary qubits, only reg
    num_qubits_after = qpu._get_qubit_heap().allocated_mask.bit_count()
    if num_qubits_after > n_qubits:
        raise ValueError("Any auxiliary qubits allocated should be returned to the |0⟩ state and released")

    if any([abs(amp) > 1e-9 for amp in actual_vector[2 ** n_qubits:]]):
        raise ValueError("Any auxiliary qubits allocated should be returned to the |0⟩ state and released")
    
    # Keep only the amplitudes of the main qubits
    actual_vector = actual_vector[:2 ** n_qubits]

    if actual_vector != approx(expected_vector):
        raise ValueError("State vectors should be equal\n"
                         "    Expected state vector:\n"
                         f"    {[round(val.real, 6) for val in expected_vector]}\n"
                         "    Actual state vector:\n"
                         f"    {[round(float(val.real), 6) for val in actual_vector]}"
                         )


#####################################################################################################################################

one_qubit_states = [
        [1., 0.],
        [0., 1.],
        [-1., 0.],
        [0., -1.],
        [1/sqrt(2), 1/sqrt(2)],
        [1/sqrt(2), -1/sqrt(2)],
        [-1/sqrt(2), 1/sqrt(2)],
        [-1/sqrt(2), -1/sqrt(2)],
        [0.6, 0.8],
        [0.6, -0.8],
        [-0.6, 0.8],
        [-0.6, -0.8]
    ]

@mark.parametrize("fun", [ref.prepare_one_qubit_state] if ref_available else [])
def test_prepare_one_qubit_state(fun):
    for alpha, beta in one_qubit_states:
        global log_message
        log_message = f"Testing {alpha=}, {beta=}"
        fun_alpha_beta = partial(fun, alpha=alpha, beta=beta)
        check_state_vector(fun_alpha_beta, 1, [alpha, beta])


#####################################################################################################################################

@mark.parametrize("fun", [ref.prepare_conditional_state] if ref_available else [])
def test_prepare_conditional_state(fun):
    for second_qubit_state in one_qubit_states:
        for alpha, beta in one_qubit_states:
            for c in [0, 1]:
                global log_message
                log_message = f"Testing |ψ⟩ = {second_qubit_state[0]}⋅|0⟩ + {second_qubit_state[1]}⋅|1⟩, {alpha=}, {beta=}, {c=}"
                fun_alpha_beta = partial(fun, alpha=alpha, beta=beta, c=c)
                initial_vector = [second_qubit_state[0], 0, second_qubit_state[1], 0]
                expected_vector = [second_qubit_state[0] * alpha, second_qubit_state[0] * beta, second_qubit_state[1], 0] if c == 0 else \
                                  [second_qubit_state[0], 0, second_qubit_state[1] * alpha, second_qubit_state[1] * beta]
                check_state_vector(fun_alpha_beta, 2, expected_vector, initial_vector=initial_vector)


#####################################################################################################################################

@mark.parametrize("fun", [ref.prepare_three_basis_states_two_qubits] if ref_available else [])
def test_prepare_three_basis_states_two_qubits(fun):
    for a in [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [1/sqrt(2), 1/sqrt(4), 1/sqrt(4)],
        [1/sqrt(3), -1/sqrt(3), 1/sqrt(3)],
        [1/sqrt(2), 1/sqrt(2), 0],
        [0, -1/sqrt(2), 1/sqrt(2)],
        [-1/sqrt(6), -1/sqrt(2), -1/sqrt(3)],
    ]:
        global log_message
        log_message = f"Testing {a=}"
        fun_a = partial(fun, a=a)
        expected_vector = a + [0]
        check_state_vector(fun_a, 2, expected_vector)


#####################################################################################################################################

two_qubit_states = [ 
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [-1., 0., 0., 0.],
        [0., -1., 0., 0.],
        [0., 0., -1., 0.],
        [0., 0., 0., -1.],        
        [0.5, 0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5, -0.5],
        [0.5, -0.5, 0.5, 0.5],
        [0.5, 0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [1. / sqrt(2.), 0., 0., 1. / sqrt(2.)],
        [1. / sqrt(2.), 0., 0., -1. / sqrt(2.)],
        [0., 1. / sqrt(2.), 1. / sqrt(2.), 0.],
        [0., 1. / sqrt(2.), -1. / sqrt(2.), 0.],
        [0.36, 0.48, 0.64, -0.48],
        [1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.), 0.]
    ]

@mark.parametrize("fun", [ref.prepare_two_qubit_state] if ref_available else [])
def test_prepare_two_qubit_state(fun):
    for a in two_qubit_states:
        global log_message
        log_message = f"Testing {a=}"
        fun_a = partial(fun, a=a)
        check_state_vector(fun_a, 2, a)


#####################################################################################################################################

@mark.parametrize("qbk_class", [ref.NaiveStatePrep] if ref_available else [])
def test_naivestateprep(qbk_class):
    # Reuse one- and two-qubit test cases
    tests = list(product([1], one_qubit_states)) + list(product([2], two_qubit_states))

    # Generate random multi-qubit test case
    seed(42)
    for n in range(3, 5):
        for _ in range(10):
            amps = [uniform(-1.0, 1.0) for _ in range(2 ** n)]
            norm = sqrt(sum(aj ** 2 for aj in amps))
            a = [aj / norm for aj in amps]
            tests.append((n, a))
    
    for n, a in tests:
        global log_message
        log_message = f"Testing {a=}"
        qbk = qbk_class()
        fun_a = partial(qbk.compute, a=a)
        check_state_vector(fun_a, n, a)

