from IPython.display import HTML
from numpy import eye, sqrt
from random import seed, uniform, randint
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
            import Workbook_LCUBlockEncoding as ref
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
    expected_vector: Sequence[float]             # State vector it should prepare
) -> None:
    # Construct the QPU and register
    qpu = QPU(num_qubits=n_qubits)
    qpu.enable_qubit_allocation_debugging()
    reg = Qubits(n_qubits, 'reg', qpu)

    fun(reg)

    # Extract the state vector
    actual_vector = qpu.pull_state()

    if actual_vector != approx(expected_vector):
        raise ValueError("State vectors should be equal\n"
                         "    Expected state vector:\n"
                         f"    {[round(val.real, 6) for val in expected_vector]}\n"
                         "    Actual state vector:\n"
                         f"    {[round(float(val.real), 6) for val in actual_vector]}"
                         )


#####################################################################################################################################

@mark.parametrize("qbk_class", [ref.OneQubitPrepare] if ref_available else [])
def test_onequbitprepare(qbk_class):
    seed(42)

    for _ in range(10):
        alpha = [uniform(0, 2), uniform(0, 2)]
        global log_message
        log_message = f"Testing {alpha=}"
        qbk = qbk_class()
        fun_alpha = partial(qbk.compute, alpha=alpha)
        l1norm = sum(alpha)
        expected_vector = [sqrt(a / l1norm) for a in alpha]
        check_state_vector(fun_alpha, 1, expected_vector)


#####################################################################################################################################

def format_matrix(a) -> str:
    return '\n'.join(f"{elem}" for elem in a)


@mark.parametrize("qbk_class", [ref.OneQubitSelectIZ] if ref_available else [])
def test_onequbitselectiz(qbk_class):
    qpu = QPU(num_qubits=2, filters=['>>buffer>>', '>>unitary>>'])
    target = Qubits(1, "target", qpu)
    index = Qubits(1, "index", qpu)

    # Explicitly apply an identity gate to each qubit
    # to make Workbench return a matrix of the right size even when the solution is empty.
    (target | index).identity()

    qbk = qbk_class()
    qbk.compute(index, target)

    # Get the actual matrix of the unitary
    ufilter = qpu.get_filter_by_name('>>unitary>>')
    actual_matrix = ufilter.get()

    # Construct the expected matrix of the SELECT
    # The index bit is the most significant one, 
    # so I is the top-left quadrant and Z is the bottom-right
    expected_matrix = eye(2 ** 2)
    expected_matrix[3][3] = -1

    for a, e in zip(actual_matrix, expected_matrix):
        assert a == approx(e), f"""Unitary implemented by your solution:
{format_matrix(actual_matrix)}
Expected matrix:
{format_matrix(expected_matrix)}
"""


#####################################################################################################################################

def print_highlighted_matrix(matrix: list[list[float]], block_size: int, header: str = "") -> None:
    """Print the matrix with elements in the top left block highlighted."""
    colors = ["#1F60E0", "#A7B2CB"]
    html = ["<table>"]
    if header != "":
        html.append(f"<th colspan='{len(matrix[0])}' style='padding: 5px 10px; font-weight: bold; color: {colors[0]}; text-align: center;'>{header}</th>")
    for row in range(len(matrix)):
        html.append("<tr>")
        for col in range(len(matrix[row])):
            color = colors[0] if row < block_size and col < block_size else colors[1]
            html.append(f"<td style='padding: 5px 10px; font-weight: bold; color: {color};'>{round(matrix[row][col], 4)}</td>")
        html.append("</tr>")
    html.append("</table>")
    display(HTML("".join(html)))


#####################################################################################################################################


@mark.parametrize("fun", [ref.lcu_decomposition] if ref_available else [])
def test_lcu_decomposition(fun):
    seed(42)

    for _ in range(10):
        # Generate alpha array and get beta array from it
        alpha = [randint(1, 10) / 10 for _ in range(4)]
        beta = [sum(alpha)]
        beta.append(alpha[0] - alpha[1] + alpha[2] - alpha[3])
        beta.append(alpha[0] + alpha[1] - alpha[2] - alpha[3])
        beta.append(alpha[0] - alpha[1] - alpha[2] + alpha[3])
        global log_message
        log_message = f"Testing {beta=}"

        alpha_res = fun(beta)
        assert alpha_res == approx(alpha), f"""Alpha values returned by your solution:
{alpha_res}
Expected values:
{alpha}
"""


#####################################################################################################################################

@mark.parametrize("qbk_class", [ref.TwoQubitPrepare] if ref_available else [])
def test_twoqubitprepare(qbk_class):
    seed(42)

    for _ in range(10):
        alpha = [uniform(0, 2) for _ in range(4)]
        global log_message
        log_message = f"Testing {alpha=}"
        qbk = qbk_class()
        fun_alpha = partial(qbk.compute, alpha=alpha)
        l1norm = sum(alpha)
        expected_vector = [sqrt(a / l1norm) for a in alpha]
        check_state_vector(fun_alpha, 2, expected_vector)


#####################################################################################################################################

@mark.parametrize("qbk_class", [ref.TwoQubitSelectIZ] if ref_available else [])
def test_twoqubitselectiz(qbk_class):
    qpu = QPU(num_qubits=4, filters=['>>buffer>>', '>>unitary>>'])
    target = Qubits(2, "target", qpu)
    index = Qubits(2, "index", qpu)

    # Explicitly apply an identity gate to each qubit
    # to make Workbench return a matrix of the right size even when the solution is empty.
    (target | index).identity()

    qbk = qbk_class()
    qbk.compute(index, target)

    # Get the actual matrix of the unitary
    ufilter = qpu.get_filter_by_name('>>unitary>>')
    actual_matrix = ufilter.get()

    # Construct the expected matrix of the SELECT.
    # The index bits are the two most significant ones
    expected_matrix = eye(2 ** 4)
    for ind in [5, 7, 10, 11, 13, 14]:
        expected_matrix[ind][ind] = -1

    for a, e in zip(actual_matrix, expected_matrix):
        assert a == approx(e), f"""Unitary implemented by your solution:
{format_matrix(actual_matrix)}
Expected matrix:
{format_matrix(expected_matrix)}
"""
