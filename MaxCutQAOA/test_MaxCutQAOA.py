from functools import partial
from cmath import cos, sin, exp, pi
import numpy as np
from psiqdk.workbench import QPU, Qubits
from pytest import approx, mark
from warnings import catch_warnings, filterwarnings

try:
    from importnb import Notebook
    with catch_warnings():
        # ignore only for this module/import
        filterwarnings("ignore", category=SyntaxWarning)
        filterwarnings("ignore", category=DeprecationWarning, module=r".*Workbook_GraphColoringGrover.*")
        # (or target the exact message)
        # filterwarnings("ignore", message=r"invalid escape sequence \\k", category=DeprecationWarning)
    
        with Notebook():
            import Workbook_MaxCutQAOA as ref
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

def classical_cost_reference(x: list[int]) -> int:
    sum = 0
    for i in range(len(x) - 1):
        sum += x[i] * x[i + 1] + (1 - x[i]) * (1 - x[i + 1])
    return sum

def test_classical_cost(fun=ref.classical_cost if ref_available else None):
    for n in range(2, 6):
        for x_int in range(2 ** n):
            x = [(x_int >> ind) % 2 for ind in range(n)]
            actual = fun(x)
            expected = classical_cost_reference(x)
            assert actual == expected, f"For {x=}, got cost={actual}, expected cost={expected}"


####################################################################################################

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


def test_phase_separation_unitary(fun=ref.phase_separation_unitary if ref_available else None):
    for n in range(2, 6):
        for j in range(11):
            gamma = j * pi / 10
            expected_matrix = [[0] * 2 ** n for _ in range(2 ** n)]
            for i_int in range(2 ** n):
                i = [(i_int >> ind) % 2 for ind in range(n)]
                expected_matrix[i_int][i_int] = exp(-1j * gamma * classical_cost_reference(i))
            fun_gamma = partial(fun, gamma=gamma)
            check_unitary_matrix(n, fun_gamma, expected_matrix)


def test_mixer_unitary(fun=ref.mixer_unitary if ref_available else None):
    for n in range(2, 6):
        for j in range(11):
            beta = j * pi / 10
            m = [[cos(beta), -1j * sin(beta)], [-1j * sin(beta), cos(beta)]]
            expected_matrix = m
            for _ in range(n - 1):
                expected_matrix = np.kron(expected_matrix, m)
            fun_beta = partial(fun, beta=beta)
            check_unitary_matrix(n, fun_beta, expected_matrix)
