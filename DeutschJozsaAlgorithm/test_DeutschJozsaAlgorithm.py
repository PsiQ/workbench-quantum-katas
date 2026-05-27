from pytest import approx, mark
from warnings import catch_warnings
from psiqdk.workbench import QPU, Qubits

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_DeutschJozsaAlgorithm as ref
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

#####################################################################################################################   
        
def test_oracle_msb_x(fun=ref.oracle_msb_x if ref_available else None):
    for n in range(1, 5):
        expected_matrix = [[0] * 2 ** n for _ in range(2 ** n)]
        for i in range(2 ** n):
            expected_matrix[i][i] = 1 if i & (2 ** (n - 1)) == 0 else -1
        check_unitary_matrix(n, fun, expected_matrix)


def test_oracle_parity(fun=ref.oracle_parity if ref_available else None):
    for n in range(1, 5):
        expected_matrix = [[0] * 2 ** n for _ in range(2 ** n)]
        for i in range(2 ** n):
            expected_matrix[i][i] = -1 if i.bit_count() % 2 == 1 else 1
        check_unitary_matrix(n, fun, expected_matrix)


def oracle_zero(reg: Qubits) -> None:
    ...

def oracle_one(reg: Qubits) -> None:
    reg[0].x()
    reg[0].z()
    reg[0].x()
    reg[0].z()

def oracle_x_mod_2(reg: Qubits) -> None:
    reg[0].z()

def oracle_middle_bit(reg: Qubits) -> None:
    reg[1].z()

def oracle_msb_x(reg: Qubits) -> None:
    reg[-1].z()

def oracle_parity(reg: Qubits) -> None:
    reg.z()


def test_is_function_constant(fun=ref.is_function_constant if ref_available else None):
    def function_type(type: bool) -> str:
        return "None" if type is None else ("constant" if type else "balanced")

    for (oracle, expected, name) in [
        (oracle_zero, True, 'f(x) = 0'),
        (oracle_one, True, 'f(x) = 1'),
        (oracle_x_mod_2, False, 'f(x) = x mod 2'),
        (oracle_msb_x, False, 'f(x) = MSB(x)'),
        (oracle_parity, False, 'f(x) = PARITY(x)')
    ]:
        for n in range(1, 5):
            actual = fun(n, oracle)
            assert actual == expected, f"{name} for {n=} identified as {function_type(actual)} but it is {function_type(expected)}"


def test_bernstein_vazirani_algorithm(fun=ref.bernstein_vazirani_algorithm if ref_available else None):
    for (n, oracle, expected, name) in [
        (2, oracle_zero, [0, 0], 'f(x) = 0'),
        (3, oracle_zero, [0, 0, 0], 'f(x) = 0'),
        (2, oracle_parity, [1, 1], 'f(x) = PARITY(x)'),
        (3, oracle_parity, [1, 1, 1], 'f(x) = PARITY(x)'),
        (2, oracle_x_mod_2, [1, 0], 'f(x) = x mod 2'),
        (3, oracle_x_mod_2, [1, 0, 0], 'f(x) = x mod 2'),
        (2, oracle_msb_x, [0, 1], 'f(x) = MSB(x)'),
        (3, oracle_msb_x, [0, 0, 1], 'f(x) = MSB(x)'),
        (3, oracle_middle_bit, [0, 1, 0], 'f(x) = middle bit of x')
    ]:
        actual = fun(n, oracle)
        assert actual == expected, f"The bit string for {name} for {n=} identified as {actual} but it is {expected}"
