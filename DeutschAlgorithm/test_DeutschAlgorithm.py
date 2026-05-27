from pytest import approx, mark
from warnings import catch_warnings
from psiqdk.workbench import QPU, Qubits

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_DeutschAlgorithm as ref
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

#############################################################################################

@mark.parametrize("fun", [ref.oracle_one_minus_x_1, ref.oracle_one_minus_x_2] if ref_available else [])
def test_oracle_one_minus_x(fun):
    expected_matrix = [[-1, 0], [0, 1]]
    check_unitary_matrix(fun, expected_matrix)


def test_is_function_constant(fun=ref.is_function_constant if ref_available else None):
    def oracle_zero(reg: Qubits) -> None:
        ...

    def oracle_one(reg: Qubits) -> None:
        reg.x()
        reg.z()
        reg.x()
        reg.z()

    def oracle_x(reg: Qubits) -> None:
        reg.z()

    def oracle_one_minus_x(reg: Qubits) -> None:
        reg.x()
        reg.z()
        reg.x()

    def function_type(type: bool) -> str:
        return "None" if type is None else ("constant" if type else "variable")

    for (oracle, expected, name) in [
        (oracle_zero, True, 'f(x) = 0'),
        (oracle_one, True, 'f(x) = 1'),
        (oracle_x, False, 'f(x) = x'),
        (oracle_one_minus_x, False, 'f(x) = 1 - x')
    ]:
        actual = fun(oracle)
        assert actual == expected, f"{name} identified as {function_type(actual)} but it is {function_type(expected)}"
