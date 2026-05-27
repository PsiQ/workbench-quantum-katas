from dataclasses import dataclass
from functools import partial
import numpy as np
from numpy.linalg import matrix_power
from psiqdk.workbench import QPU, Qubits
from pytest import mark, approx
from random import randint, seed, random
from scipy.linalg import expm
from warnings import catch_warnings


@dataclass(frozen=True)
class HamiltonianTerm:
    c: float
    x_mask: int
    z_mask: int

try:
    from importnb import Notebook
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_GroundStateEnergy as ref
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

####################################################################################################

@dataclass(frozen=True)
class Pauli:
    name: str
    matrix: np.array
    is_x: bool
    is_z: bool

pauli_i = Pauli("I", np.array([[1, 0], [0, 1]]), False, False)
pauli_x = Pauli("X", np.array([[0, 1], [1, 0]]), True, False)
pauli_y = Pauli("Y", np.array([[0, -1j], [1j, 0]]), True, True)
pauli_z = Pauli("Z", np.array([[1, 0], [0, -1]]), False, True)

####################################################################################################

@mark.parametrize("fun", [ref.single_qubit_ppr_1, ref.single_qubit_ppr_2] if ref_available else [])
def test_single_qubit_ppr(fun):
    for pauli in [pauli_x, pauli_y, pauli_z]:
        print(f"Testing Pauli {pauli.name}...")
        for theta in range(5):
            expected_matrix = expm(1j * theta * pauli.matrix)
            fun_args = partial(fun, theta=theta, is_x=pauli.is_x, is_z=pauli.is_z)
            check_unitary_matrix(1, fun_args, expected_matrix)

####################################################################################################

def test_two_qubit_ppr(fun=ref.two_qubit_ppr if ref_available else None):
    for pauli0 in [pauli_i, pauli_x, pauli_y, pauli_z]:
        for pauli1 in [pauli_i, pauli_x, pauli_y, pauli_z]:
            if pauli0.name == "I" and pauli1.name == "I":
                continue
            print(f"Testing P0 = {pauli0.name}, P1 = {pauli1.name}...")
            tensor = np.kron(pauli1.matrix, pauli0.matrix)
            for theta in range(5):
                expected_matrix = expm(1j * theta * tensor)
                x_mask = int(pauli0.is_x) + (int(pauli1.is_x) << 1)
                z_mask = int(pauli0.is_z) + (int(pauli1.is_z) << 1)
                fun_args = partial(fun, theta=theta, x_mask=x_mask, z_mask=z_mask)
                check_unitary_matrix(2, fun_args, expected_matrix)


####################################################################################################

def masks_as_pauli(x_mask: int, z_mask: int, pos: int) -> Pauli:
    is_x = (x_mask & (1 << pos)) > 0
    is_z = (z_mask & (1 << pos)) > 0
    for pauli in [pauli_i, pauli_x, pauli_y, pauli_z]:
        if pauli.is_x == is_x and pauli.is_z == is_z:
            return pauli


def terms_as_matrix(t: float, terms: list[HamiltonianTerm]) -> list[list[complex]]:
    '''Convert two-qubit Hamiltonian terms to an expected matrix.'''
    expected_matrix = np.eye(4)
    for term in terms:
        p0 = masks_as_pauli(term.x_mask, term.z_mask, 0)
        p1 = masks_as_pauli(term.x_mask, term.z_mask, 1)
        tensor = np.kron(p1.matrix, p0.matrix)
        matrix = expm(-1j * t * term.c * tensor)
        # Do matrix multiplication in reverse order (first term to be applied is last in the product)
        expected_matrix = matrix @ expected_matrix
    return expected_matrix


def test_trotter_step(fun=ref.trotter_step if ref_available else None):
    seed(42)
    for t in range(10):
        # Generate a random Hamiltonian
        m = randint(1, 4)
        terms = []
        for _ in range(m):
            term = HamiltonianTerm(random(), randint(1, 4), randint(1, 4))
            terms.append(term)

        expected_matrix = terms_as_matrix(t, terms)

        fun_args = partial(fun, t=t, terms=terms)
        check_unitary_matrix(2, fun_args, expected_matrix)


def test_trotter_approximation(fun=ref.trotter_approximation if ref_available else None):
    seed(42)
    for t in range(10):
        # Generate a random Hamiltonian
        m = randint(1, 4)
        terms = []
        for _ in range(m):
            term = HamiltonianTerm(random(), randint(1, 4), randint(1, 4))
            terms.append(term)
        n = randint(2, 10)

        expected_step_matrix = terms_as_matrix(t / n, terms)
        expected_matrix = matrix_power(expected_step_matrix, n)

        fun_args = partial(fun, t=t, terms=terms, n=n)
        check_unitary_matrix(2, fun_args, expected_matrix)
