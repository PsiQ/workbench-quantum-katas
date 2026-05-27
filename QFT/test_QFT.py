from cmath import exp, pi
from functools import partial
from math import sqrt
from psiqdk.workbench import QPU, Qubits
from pytest import approx, mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_QFT as ref
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

def test_single_qubit_qft(fun=ref.single_qubit_qft if ref_available else None):
    expected_matrix = [[1/sqrt(2), 1/sqrt(2)],
                       [1/sqrt(2), -1/sqrt(2)]]
    check_unitary_matrix(1, fun, expected_matrix)


####################################################################################################

def test_rotation_gate(fun=ref.rotation_gate if ref_available else None):
    for k in range(5):
        print(f"Testing {k=}")
        expected_matrix = [[1, 0],
                           [0, exp(2j * pi / (2 ** k))]]
        fun_k = partial(fun, k=k)
        check_unitary_matrix(1, fun_k, expected_matrix)


####################################################################################################

def test_binary_fraction_exponent_classical(fun=ref.binary_fraction_exponent_classical if ref_available else None):
    for n in range(1, 5):
        for j_int in range(2 ** n):
            expected_matrix = [[1, 0],
                               [0, exp(2j * pi * j_int / (2 ** n))]]
            j_bits = [(j_int & (1 << ind)) > 0 for ind in range(n)]
            print(f"Testing {n=}, j={j_bits} (int {j_int})")
            fun_j = partial(fun, j=j_bits)
            check_unitary_matrix(1, fun_j, expected_matrix)


####################################################################################################

def test_binary_fraction_exponent_quantum(fun=ref.binary_fraction_exponent_quantum if ref_available else None):
    def fun_wrapper(reg: Qubits) -> None:
        '''Wrapper to split one Qubits argument into two and call the solution with it.'''
        x = reg[0]
        j = reg[1:]
        fun(x, j)

    for n in range(1, 5):
        print(f"Testing {n=}")
        expected_matrix = [[0] * 2 ** (n + 1) for _ in range(2 ** (n + 1))]
        for ind in range(2 ** (n + 1)):
            if ind % 2 == 0:
                # x is in basis state |0⟩, no relative phase
                expected_matrix[ind][ind] = 1
            else:
                j = ind // 2
                expected_matrix[ind][ind] = exp(2j * pi * j / (2 ** n))
        check_unitary_matrix(n + 1, fun_wrapper, expected_matrix)


####################################################################################################

def test_binary_fraction_exponent_inplace(fun=ref.binary_fraction_exponent_inplace if ref_available else None):
    for n in range(1, 5):
        print(f"Testing {n=}")
        expected_matrix = [[0] * 2 ** n for _ in range(2 ** n)]
        for ind in range(2 ** n):
            ind_lsb = ind % (2 ** (n - 1))
            expected_matrix[ind_lsb][ind] = 1 / sqrt(2)
            expected_matrix[ind_lsb + 2 ** (n - 1)][ind] = 1 / sqrt(2) * exp(2j * pi * ind / (2 ** n))
        check_unitary_matrix(n, fun, expected_matrix)


####################################################################################################

def test_quantum_fourier_transform(fun=ref.quantum_fourier_transform if ref_available else None):
    for n in range(1, 5):
        print(f"Testing {n=}")
        expected_matrix = [[0] * 2 ** n for _ in range(2 ** n)]
        # Matrix from direct definition of QFT, not the task description (equivalent)
        for ind in range(2 ** n):
            for k in range(2 ** n):
                expected_matrix[k][ind] = 1 / sqrt(2 ** n) * exp(2j * pi * ind * k / (2 ** n))
        check_unitary_matrix(n, fun, expected_matrix)


####################################################################################################

def run_stateprep_test(n_qubits, solution, expected_state):
    qpu = QPU(num_qubits=n_qubits)
    reg = Qubits(n_qubits, "reg", qpu)
    # Prepare the state asked for by the task
    qpu.label("Solution")
    solution(reg)

    actual_state = qpu.pull_state()

    assert actual_state == approx(expected_state), f"""Actual state:
{actual_state}
Expected state:
{expected_state}
"""


def test_prepare_equal_superposition(fun=ref.prepare_equal_superposition if ref_available else None):
    for n in range(1, 5):
        print(f"Testing {n=}")
        expected_state = [1 / sqrt(2 ** n) for _ in range(2 ** n)]
        run_stateprep_test(n, fun, expected_state)


####################################################################################################

def test_prepare_periodic_state(fun=ref.prepare_periodic_state if ref_available else None):
    for n in range(1, 5):
        for freq in range(2 ** n):
            print(f"Testing {n=}, {freq=}")
            expected_state = [exp(2j * pi * k * freq / 2 ** n) / sqrt(2 ** n) for k in range(2 ** n)]
            fun_freq = partial(fun, freq=freq)
            run_stateprep_test(n, fun_freq, expected_state)


####################################################################################################

def test_prepare_alternating_amplitudes_state(fun=ref.prepare_alternating_amplitudes_state if ref_available else None):
    for n in range(1, 5):
        print(f"Testing {n=}")
        expected_state = [(1 if k % 2 == 0 else -1) / sqrt(2 ** n) for k in range(2 ** n)]
        run_stateprep_test(n, fun, expected_state)


####################################################################################################

def test_prepare_equal_superposition_even_states(fun=ref.prepare_equal_superposition_even_states if ref_available else None):
    for n in range(1, 5):
        print(f"Testing {n=}")
        expected_state = [(1 if k % 2 == 0 else 0) / sqrt(2 ** (n - 1)) for k in range(2 ** n)]
        run_stateprep_test(n, fun, expected_state)


####################################################################################################

def test_prepare_square_wave(fun=ref.prepare_square_wave if ref_available else None):
    for n in range(2, 5):
        print(f"Testing {n=}")
        expected_state = [(1 if k % 4 < 2 else -1) / sqrt(2 ** n) for k in range(2 ** n)]
        run_stateprep_test(n, fun, expected_state)


####################################################################################################

def test_get_signal_frequency(fun=ref.get_signal_frequency if ref_available else None):
    qpu = QPU()
    for n in range(1, 5):
        for f in range(2 ** n):
            state = [exp(2j * pi * k * f / 2 ** n) / sqrt(2 ** n) for k in range(2 ** n)]
            qpu.reset(n)
            reg = Qubits(n, "reg", qpu)
            reg.push_state(state)
            actual_f = fun(reg)
            assert actual_f == f, f"Incorrect answer for {n=}: expected {f=}, got {actual_f}"
