from math import sqrt
from cmath import exp, pi
from psiqdk.workbench import QPU, Qubits
from pytest import approx, mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_MultiQubitSystems as ref
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

####################################################################################################

def check_state_vector(
    fun: callable,        # Callable that is being tested
    expected_vector: list[complex], # State vector it should prepare
    initial_state_prep: callable = None# A routine that prepares the initial state (before the solution is called)
) -> None:
    # Construct the qpu and register
    qpu = QPU(num_qubits = 2)
    reg = Qubits(num_qubits = 2, name = 'reg', qpu = qpu)

    if initial_state_prep is not None:
        initial_state_prep(reg)

    fun(reg)

    # Extract the saved state vector
    actual_vector = reg.pull_state()

    if actual_vector != approx(expected_vector):
        print("Expected state vector:")
        print(expected_vector)
        print("Actual state vector:")
        print(actual_vector)
        raise ValueError("State vectors should be equal")


def test_prepare_oneone(fun=ref.prepare_oneone if ref_available else None):
    expected_vector = [0, 0, 0, 1]
    check_state_vector(fun, expected_vector)


def test_prepare_superposition(fun=ref.prepare_superposition if ref_available else None):
    sqrt12 = 1 / sqrt(2)
    expected_vector = [sqrt12, 0, -sqrt12, 0]
    check_state_vector(fun, expected_vector)


def test_prepare_real_amplitudes(fun=ref.prepare_real_amplitudes if ref_available else None):
    expected_vector = [0.5, 0.5, -0.5, -0.5]
    check_state_vector(fun, expected_vector)


def test_prepare_complex_amplitudes(fun=ref.prepare_complex_amplitudes if ref_available else None):
    expected_vector = [0.5, 0.5 * exp(1j * pi / 2), 0.5 * exp(1j * pi / 4), 0.5 * exp(1j * 3 * pi / 4)]
    check_state_vector(fun, expected_vector)


def bell_state_prep(reg: Qubits) -> None:
    reg[0].had()
    reg[1].x(cond=reg[0])


def test_prepare_bell_state_1(fun=ref.prepare_bell_state_1 if ref_available else None):
    sqrt12 = 1 / sqrt(2)
    expected_vector = [sqrt12, 0, 0, -sqrt12]
    check_state_vector(fun, expected_vector, bell_state_prep)


def test_prepare_bell_state_2(fun=ref.prepare_bell_state_2 if ref_available else None):
    sqrt12 = 1 / sqrt(2)
    expected_vector = [0, sqrt12, sqrt12, 0]
    check_state_vector(fun, expected_vector, bell_state_prep)


def test_prepare_bell_state_3(fun=ref.prepare_bell_state_3 if ref_available else None):
    sqrt12 = 1 / sqrt(2)
    expected_vector = [0, -sqrt12, sqrt12, 0]
    check_state_vector(fun, expected_vector, bell_state_prep)
