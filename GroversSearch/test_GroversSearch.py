from functools import partial
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.filter_presets import BIT_DEFAULT
from pytest import approx, mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_GroversSearch as ref
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

def int_to_bool_array(n: int, num_bits: int) -> list[bool]:
    """
    Converts an integer to a list of Booleans representing its bits.
    The order is from the least significant bit to the most significant bit.
    """
    return [bool(n & (1 << ind)) for ind in range(num_bits)]


def run_test_reversible(qpu: QPU, n_inputs: int, n_qubits: int, quantum_op, f):
    for input_mask in range(2 ** n_inputs):
        qpu.reset(n_qubits)
        x = Qubits(n_inputs, "x", qpu)
        y = Qubits(1, "y", qpu)

        # Prepare quantum input and run the reversible computation
        x.write(input_mask)
        quantum_op(x, y)

        # Convert integer input to a Boolean array
        input_str = (f"{{:0>{n_inputs}b}}").format(input_mask)
        input_le = int_to_bool_array(input_mask, n_inputs)

        # Evaluate classical function on the classical input
        res_expected = int(f(input_le))

        # Compare the results of classical and quantum computations
        res_x = x.read()
        res_y = y.read()

        # Show bit string input in little-endian (LSB first) to match qubit state
        if res_x != input_mask:
            raise Exception(f"Error for x={input_mask} ({input_str[::-1]}): the state of the input qubits changed")
        if res_y != res_expected:
            raise Exception(f"Error for x={input_mask} ({input_str[::-1]}): expected {res_expected}, got {res_y}")


def format_matrix(a) -> str:
    return '\n'.join(f"{elem}" for elem in a)


####################################################################################################

def f_starts_with(args: list[bool], pattern: list[bool]) -> bool:
    return args[:len(pattern)] == pattern


def test_oracle_starts_with(fun=ref.oracle_starts_with if ref_available else None):
    qpu = QPU(filters=BIT_DEFAULT)
    for (n, pattern) in [
        (2, [True]),
        (2, [True, False]),
        (3, [False, True]),
        (4, [True, True, False]),
        (5, [False])
    ]:
        print(f"Testing {n=}, {pattern=}")
        f = partial(f_starts_with, pattern=pattern)
        quantum_op = partial(fun, pattern=pattern)
        run_test_reversible(qpu, n, n + 1, quantum_op, f)


####################################################################################################

def test_marking_oracle_as_phase_oracle(fun=ref.marking_oracle_as_phase_oracle if ref_available else None):
    def mark_pattern(x: Qubits, y: Qubits, pattern: int):
        y.x(cond=x == pattern)

    for n in range(1, 4):
        for pattern in range(2 ** n):
            qpu = QPU(num_qubits=n + 1, filters=['>>buffer>>', '>>unitary>>'])
            x = Qubits(n, "reg", qpu)
            # Explicitly apply an identity gate to each qubit
            # to make Workbench return a matrix of the right size even when the solution is empty.
            x.identity()
            marking_oracle = partial(mark_pattern, pattern=pattern)
            fun(marking_oracle, x)

            # Get the actual matrix of the phase oracle
            ufilter = qpu.get_filter_by_name('>>unitary>>')
            complete_matrix = ufilter.get()
            # Trim the matrix to the size acting on just the first n qubits
            actual_matrix = [complete_matrix[ind][:2 ** n] for ind in range(2 ** n)]
            # print("Unitary implemented by your solution:")
            # print(actual_matrix)

            # Construct the expected matrix of the phase oracle marking one pattern
            expected_matrix = [[0] * 2 ** n for _ in range(2 ** n)]
            for ind in range(2 ** n):
                expected_matrix[ind][ind] = -1 if ind == pattern else 1
            # print("Expected matrix:")
            # print(expected_matrix)

            for a, e in zip(actual_matrix, expected_matrix):
                assert a == approx(e), f"""Testing {n=}, marking oracle for {pattern=}:
Unitary implemented by your solution:
{format_matrix(actual_matrix)}
Expected matrix:
{format_matrix(expected_matrix)}
"""


####################################################################################################

def test_conditional_phase_flip(fun=ref.conditional_phase_flip if ref_available else None):
    for n in range(2, 5):
        qpu = QPU(num_qubits=n, filters=['>>buffer>>', '>>unitary>>'])
        x = Qubits(n, "reg", qpu)
        # Explicitly apply an identity gate to each qubit
        # to make Workbench return a matrix of the right size even when the solution is empty.
        x.identity()

        fun(x)
        # Get the actual matrix of the phase oracle
        ufilter = qpu.get_filter_by_name('>>unitary>>')
        complete_matrix = ufilter.get()
        # Trim the matrix to the size acting on just the first n qubits, just in case
        actual_matrix = [complete_matrix[ind][:2 ** n] for ind in range(2 ** n)]

        # Construct the expected matrix of the phase oracle marking one pattern
        expected_matrix = [[0] * 2 ** n for _ in range(2 ** n)]
        expected_matrix[0][0] = 1
        for ind in range(1, 2 ** n):
            expected_matrix[ind][ind] = -1

        for a, e in zip(actual_matrix, expected_matrix):
            assert a == approx(e), f"""Testing {n=}
Unitary implemented by your solution:
{format_matrix(actual_matrix)}
Expected matrix:
{format_matrix(expected_matrix)}
"""
