from cmath import exp, pi
from math import sqrt
from psiqdk.workbench import QPU, Qubits
from pytest import approx, mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_Qubit as ref
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


def test_fetch_single_qubit_amplitudes(fun=ref.fetch_single_qubit_amplitudes if ref_available else None):
    amplitudes = [0, 0.2, 0.4, 0.6, 0.8, 1]
    phases = [-pi, -5*pi/6, -3*pi/4, -2*pi/3, -pi/2, -pi/3, -pi/4, -pi/6, 0, pi/6, pi/4, pi/3, pi/2, 2*pi/3, 3*pi/4, 5*pi/6, pi]
    for amp_0 in amplitudes:
        amp_1 = sqrt(1 - amp_0 ** 2)
        for phase in phases:
            expected_state = [amp_0, amp_1 * exp(1j * phase)]
            qpu = QPU(num_qubits=1)
            reg = Qubits(1, "reg", qpu)
            reg.push_state(expected_state)

            returned_state = fun(reg)

            assert returned_state == approx(expected_state), f"""Returned amplitudes: 
{returned_state}
Expected amplitudes:
{expected_state}
"""
