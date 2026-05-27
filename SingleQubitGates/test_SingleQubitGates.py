from cmath import exp, pi
from functools import partial
from math import sqrt, cos, sin
from psiqdk.workbench import QPU, Qubits
from pytest import approx, mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_SingleQubitGates as ref
    ref_available = True
except ImportError as e:
    ref_available = False
    # Skip all tests in this file at once instead of one by one
    message = "importnb not installed" if "importnb" in str(e) else "workbook file not available"
    pytestmark = mark.skip(message)

# Run pytest in the folder to run all the tests on reference solutions 
# from the respective file instead of solutions in the Jupyter Notebook.

# "problem" decorator: specifies that executing this cell tests this function using a test with a fixed name
def problem(fun):
    try:
        # Build test name
        test_name = "test_" + fun.__name__
        # Find test function; if none found, raise an exception
        test_func = globals()[test_name]
    except KeyError:
        print(f"Test {test_name} not found")
    else:
        # Run the test on this function
        try:
            test_func(fun)
        except Exception as e:
            print("Incorrect")
            print(str(e))
        else:
            print("Correct!")

    return fun

####################################################################################################
def format_matrix(a) -> str:
    return '\n'.join(f"{elem}" for elem in a)


def check_single_qubit_unitary_matrix(
    quantum_op: callable,
    expected_matrix: list[list[complex]]
) -> None:
    qpu = QPU(num_qubits=1, filters=['>>buffer>>', '>>unitary>>'])
    x = Qubits(1, "x", qpu)
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


def single_qubit_stateprep_test(solution, expected_state):
    qpu = QPU(num_qubits=1)
    reg = Qubits(1, "reg", qpu)
    # Prepare the state asked for by the task
    qpu.label("Solution")
    solution(reg)

    prepared_state = qpu.pull_state()

    assert prepared_state == approx(expected_state), f"""Prepared state:
{prepared_state}
Expected state:
{expected_state}
"""


def test_state_flip(fun=ref.state_flip if ref_available else None):
    expected_matrix = [[0, 1], [1, 0]]
    check_single_qubit_unitary_matrix(fun, expected_matrix)


def test_sign_flip(fun=ref.sign_flip if ref_available else None):
    expected_matrix = [[1, 0], [0, -1]]
    check_single_qubit_unitary_matrix(fun, expected_matrix)


def test_apply_y(fun=ref.apply_y if ref_available else None):
    expected_matrix = [[0, -1j], [1j, 0]]
    check_single_qubit_unitary_matrix(fun, expected_matrix)


def test_sign_flip_on_zero(fun=ref.sign_flip_on_zero if ref_available else None):
    expected_matrix = [[-1, 0], [0, 1]]
    check_single_qubit_unitary_matrix(fun, expected_matrix)


def test_global_phase_minus_one(fun=ref.global_phase_minus_one if ref_available else None):
    expected_matrix = [[-1, 0], [0, -1]]
    check_single_qubit_unitary_matrix(fun, expected_matrix)


def test_global_phase_i(fun=ref.global_phase_i if ref_available else None):
    expected_matrix = [[1j, 0], [0, 1j]]
    check_single_qubit_unitary_matrix(fun, expected_matrix)


def test_basis_change(fun=ref.basis_change if ref_available else None):
    sqrt12 = 1 / sqrt(2)
    expected_matrix = [[sqrt12, sqrt12], [sqrt12, -sqrt12]]
    check_single_qubit_unitary_matrix(fun, expected_matrix)


def test_prepare_plus(fun=ref.prepare_plus if ref_available else None):
    sqrt12 = 1 / sqrt(2)
    expected_vector = [sqrt12, sqrt12]
    single_qubit_stateprep_test(fun, expected_vector)


def test_prepare_minus(fun=ref.prepare_minus if ref_available else None):
    sqrt12 = 1 / sqrt(2)
    expected_vector = [sqrt12, -sqrt12]
    single_qubit_stateprep_test(fun, expected_vector)


def test_relative_phase_i(fun=ref.relative_phase_i if ref_available else None):
    expected_matrix = [[1, 0], [0, 1j]]
    check_single_qubit_unitary_matrix(fun, expected_matrix)


def test_relative_phase_three_quarters_pi(fun=ref.relative_phase_three_quarters_pi if ref_available else None):
    expected_matrix = [[1, 0], [0, exp(3 * pi / 4 * 1j)]]
    check_single_qubit_unitary_matrix(fun, expected_matrix)


def test_amplitude_change(fun=ref.amplitude_change if ref_available else None):
    for ind in range(37):
        gamma = 2 * pi * ind / 36
        expected_matrix = [[cos(gamma), -sin(gamma)], [sin(gamma), cos(gamma)]]
        fun_gamma = partial(fun, gamma=gamma)
        check_single_qubit_unitary_matrix(fun_gamma, expected_matrix)


def test_relative_phase_change(fun=ref.relative_phase_change if ref_available else None):
    for ind in range(37):
        gamma = 2 * pi * ind / 36
        expected_matrix = [[1, 0], [0, exp(1j * gamma)]]
        fun_gamma = partial(fun, gamma=gamma)
        check_single_qubit_unitary_matrix(fun_gamma, expected_matrix)


def test_prepare_rotated_state(fun=ref.prepare_rotated_state if ref_available else None):
    for ind in range(11):
        alpha = cos(ind)
        beta = sin(ind)
        expected_vector = [alpha, -1j * beta]
        fun_alpha_beta = partial(fun, alpha=alpha, beta=beta)
        single_qubit_stateprep_test(fun_alpha_beta, expected_vector)


def test_prepare_arbitrary_state(fun=ref.prepare_arbitrary_state if ref_available else None):
    for ind in range(11):
        alpha = cos(ind)
        beta = sin(ind)
        theta = sin(ind * 3)
        expected_vector = [alpha, exp(1j * theta) * beta]
        fun_alpha_beta = partial(fun, alpha=alpha, beta=beta, theta=theta)
        single_qubit_stateprep_test(fun_alpha_beta, expected_vector)
