from math import sqrt, sin, cos
from cmath import exp, pi
from functools import partial
from psiqdk.workbench import QPU, Qubits
from pytest import approx, mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_MultiQubitGates as ref
    ref_available = True
except ImportError:
    ref_available = False
    # Skip all tests in this file - pytest checks reference solutions and that won't work without these imports
    pytestmark = mark.skip("No importnb/reference file available")


def problem(fun):
    test_name = "test_" + fun.__name__
    try:
        test_func = globals()[test_name]
    except KeyError:
        print(f"Test {test_name} not found")
    else:
        try:
            test_func(fun)
        except Exception as e:
            print("Incorrect")
            print(e)
        else:
            print("Correct!")
    
    return fun

#########################################################################################################

def check_state_vector(
    fun: callable,        # Callable that is being tested
    n_qubits: int,        # Number of qubits in the register
    expected_vector: list[complex], # State vector it should prepare
    initial_state_prep: callable = None  # Another function that prepares the initial state (before fun is called)
) -> None:
    # Construct the qpu and register
    qpu = QPU(num_qubits=n_qubits)
    reg = Qubits(num_qubits=n_qubits, name='reg', qpu=qpu)

    if initial_state_prep is not None:
        initial_state_prep(reg)

    fun(reg)
    
    # Store the state vector of the current state
    actual_vector = reg.pull_state()

    if actual_vector != approx(expected_vector):
        print("Expected state vector:")
        print(expected_vector)
        print("Actual state vector:")
        print(actual_vector)
        raise ValueError("State vectors should be equal")


def format_matrix(a) -> str:
    return '\n'.join(f"{elem}" for elem in a)


def check_unitary_matrix(
    n_qubits: int,
    quantum_op: callable,
    expected_matrix: list[list[complex]]
) -> None:
    qpu = QPU(num_qubits=n_qubits, filters=['>>buffer>>', '>>unitary>>'])
    x = Qubits(n_qubits, "x", qpu)
    # Explicitly apply an identity gate to each qubit
    # to make Workbench return a matrix of the right size even when the solution is empty.
    x.identity()

    # Apply the unitary
    quantum_op(x)

    # Get the actual matrix of the unitary
    ufilter = qpu.get_filter_by_name('>>unitary>>')
    actual_matrix = ufilter.get()

    for actual_row, expected_row in zip(actual_matrix, expected_matrix):
        assert actual_row == approx(expected_row), f"""Unitary implemented by your solution:
{format_matrix(actual_matrix)}
Expected matrix:
{format_matrix(expected_matrix)}
"""
        
#############################################################################################################################

def test_apply_tensor_product(fun=ref.apply_tensor_product if ref_available else None):
    expected_matrix = [[0, -1j, 0, 0, 0, 0, 0, 0],
                       [1j,  0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, -1j, 0, 0, 0, 0],
                       [0, 0, 1j,  0, 0, 0, 0, 0],
                       [0, 0, 0, 0,  0, 1, 0, 0],
                       [0, 0, 0, 0, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0,  0, 1],
                       [0, 0, 0, 0, 0, 0, -1, 0],
                       ]
    check_unitary_matrix(3, fun, expected_matrix)


isq2 = 1 / sqrt(2)

def test_prepare_bell_state(fun=ref.prepare_bell_state if ref_available else None):
    expected_vector = [isq2, 0, 0, isq2]
    check_state_vector(fun, 2, expected_vector)


def test_entangle_two_qubits(fun=ref.entangle_two_qubits if ref_available else None):
    def initial_state_prep(reg: Qubits) -> None:
        reg.had()

    expected_vector = [0.5, 0.5, 0.5, -0.5]
    check_state_vector(fun, 2, expected_vector, initial_state_prep)


def test_swap_amplitudes(fun=ref.swap_amplitudes if ref_available else None):
    expected_matrix = [[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       ]
    check_unitary_matrix(2, fun, expected_matrix)


def test_fredkin_gate(fun=ref.fredkin_gate if ref_available else None):
    # Swap basis states 3 and 5
    expected_matrix = [[1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1],
                       ]
    check_unitary_matrix(3, fun, expected_matrix)


def test_controlled_rotation(fun=ref.controlled_rotation if ref_available else None):
    for i in range(20):
        theta = pi * i / 10
        expected_matrix = [[1, 0, 0, 0],
                           [0, cos(theta/2), 0, -sin(theta/2)],
                           [0, 0, 1, 0],
                           [0, sin(theta/2), 0, cos(theta/2)],
                           ]
        fun_theta = partial(fun, theta=theta)
        check_unitary_matrix(2, fun_theta, expected_matrix)


def test_controlled_phase(fun=ref.controlled_phase if ref_available else None):
    for i in range(20):
        theta = pi * i / 10
        expected_matrix = [[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, exp(1j * theta)],
                           ]
        fun_theta = partial(fun, theta=theta)
        check_unitary_matrix(3, fun_theta, expected_matrix)


def test_anti_controlled_gate(fun=ref.anti_controlled_gate if ref_available else None):
    expected_matrix = [[0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1],
                       ]
    check_unitary_matrix(2, fun, expected_matrix)
