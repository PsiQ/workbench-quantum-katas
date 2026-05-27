from functools import partial
from math import gcd
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.filter_presets import BIT_DEFAULT
from pytest import mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_ShorsFactoring as ref
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

def test_find_period_classical(fun=ref.find_period_classical if ref_available else None):
    for n in range(15, 26):
        for a in range(2, n):
            if gcd(n, a) > 1:
                continue
            print(f"Testing {n=}, {a=}")
            p_sol = fun(n, a)
            for p in range(1, p_sol):
                a_x = (a ** p) % n
                if a_x == 1:
                    raise Exception(f"Error for {n=}, {a=}: returned value {p_sol}, expected {p}")
            a_x = (a ** p_sol) % n
            if a_x != 1:
                raise Exception(f"Error for {n=}, {a=}: returned value {p_sol} is not a period")


####################################################################################################

def int_to_bool_array(n: int, num_bits: int) -> list[bool]:
    """
    Converts an integer to a list of Booleans representing its bits.
    The order is from the least significant bit to the most significant bit.
    """
    return [bool(n & (1 << ind)) for ind in range(num_bits)]


def run_test_reversible_inplace(n_qubits: int, quantum_op, f):
    '''Test specifically for reversible in-place arithmetic'''
    qpu = QPU(filters=BIT_DEFAULT)
    qpu.enable_qubit_allocation_debugging()
    for input_mask in range(2 ** n_qubits):
        qpu.reset(n_qubits)
        x = Qubits(n_qubits, "x", qpu)

        # Prepare quantum input and run the reversible computation
        x.write(input_mask)
        quantum_op(x)

        # Convert integer input to a Boolean array
        input_str = (f"{{:0>{n_qubits}b}}").format(input_mask)

        # Evaluate classical function on the classical input
        res_expected = f(input_mask)

        # Compare the results of classical and quantum computations
        res_actual = x.read()

        # Show bit string input in little-endian (LSB first) to match qubit state
        if res_actual != res_expected:
            raise Exception(f"Error for x={input_mask} ({input_str[::-1]}): expected {res_expected}, got {res_actual}")


####################################################################################################

def f_multiply_by_2_mod_15(x: int) -> int:
    if x == 15:
        return x
    return (x * 2) % 15


def test_multiply_by_2_mod_15(fun=ref.multiply_by_2_mod_15 if ref_available else None):
    run_test_reversible_inplace(4, fun, f_multiply_by_2_mod_15)


####################################################################################################

def f_multiply_by_4_mod_15(x: int) -> int:
    if x == 15:
        return x
    return (x * 4) % 15


def test_multiply_by_4_mod_15(fun=ref.multiply_by_4_mod_15 if ref_available else None):
    run_test_reversible_inplace(4, fun, f_multiply_by_4_mod_15)


####################################################################################################

def f_multiply_by_2k_mod_15(x: int, k: int) -> int:
    if x == 15:
        return x
    return (x * 2 ** k) % 15


def test_multiply_by_2k_mod_15(fun=ref.multiply_by_2k_mod_15 if ref_available else None):
    for k in range(16):
        print(f"Testing {k=}")
        fun_k = partial(fun, k=k)
        f_k = partial(f_multiply_by_2k_mod_15, k=k)
        run_test_reversible_inplace(4, fun_k, f_k)
