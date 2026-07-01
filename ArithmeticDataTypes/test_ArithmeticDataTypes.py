from functools import partial
from typing import Callable
from psiqdk.workbench import QPU, Qubits, QUInt, QInt
from psiqdk.workbench.filter_presets import BIT_DEFAULT
from pytest import mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_ArithmeticDataTypes as ref
    ref_available = True
except ImportError as e:
    ref_available = False
    # Skip all tests in this file at once instead of one by one
    message = "importnb not installed" if "importnb" in str(e) else "workbook file not available"
    pytestmark = mark.skip(message)

# Run pytest in the folder to run all the tests on reference solutions 
# from the respective file instead of solutions in the Jupyter Notebook.

log_message = ""

# "problem" decorator: specifies that executing this cell tests this function using a test with a fixed name
def problem(arg):
    try:
        # Build test name
        test_name = "test_" + arg.__name__.lower()
        # Find test function; if none found, raise an exception
        test_func = globals()[test_name]
    except KeyError:
        print(f"Test {test_name} not found")
    else:
        # Run the test on this function
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

####################################################################################################

def get_instructions(n_bits_a: int, n_bits_b: int, n_qubits: int, qtype: type[Qubits], quantum_op: Callable):
    '''Runs solution on a QPU and fetches the list of gate instructions produced for later replay.'''
    qpu = QPU(num_qubits=n_qubits, filters=BIT_DEFAULT)
    a = qtype(n_bits_a, "a", qpu)
    b = qtype(n_bits_b, "b", qpu)
    if n_bits_b == 0:
        quantum_op(a)
    else:
        quantum_op(a, b)

    # Check that there are no leftover auxiliary qubits - only a and b
    # num_qubits = qpu._get_qubit_heap().allocated_mask.bit_count()
    # if num_qubits != n_inputs_a + n_inputs_b:
    #     raise Exception("Your solution should release all auxiliary qubits it allocates")

    instructions = qpu.get_instructions(format='cpp')
    # The first two or three instructions
    # are going to be reset and qubit allocations - skip them
    return instructions[2 + (1 if n_bits_b > 0 else 0):]


def run_test_reversible(n_bits_a: int, n_bits_b: int, qtype: type[Qubits], quantum_op: Callable, f: Callable):
    # For naive operations, we don't need to allocate any auxiliary qubits
    n_qubits = n_bits_a + n_bits_b
    instructions = get_instructions(n_bits_a, n_bits_b, n_qubits, qtype, quantum_op)

    qpu = QPU(filters=BIT_DEFAULT)
    bit_sim = qpu.get_filter_by_name('>>bit-sim>>')
    qpu.enable_qubit_allocation_debugging()
    for input in range(2 ** (n_bits_a + n_bits_b)):
        # Reset QPU and allocate the registers of the required type
        qpu.reset(n_qubits)
        a = qtype(n_bits_a, "a", qpu)
        b = qtype(n_bits_b, "b", qpu)

        # Prepare quantum input a (signed if necessary)
        input_a = input % (2 ** n_bits_a) + a.min_value()
        a.write(input_a)

        # If input b is quantum, prepare it (signed if necessary)
        if n_bits_b > 0:
            input_b = (input >> n_bits_a) + b.min_value()
            b.write(input_b)

        # Flush the whole QPU before adding instructions to a specific filter
        qpu.flush()

        # Run the reversible computation as instructions replay
        bit_sim._put_native(instructions)

        # Evaluate classical function on the classical input
        res_expected = f(input_a) if n_bits_b == 0 else f(input_a, input_b)

        # Compare the results of classical and quantum computations
        res_a = a.read()
        if n_bits_b > 0:
            res_b = b.read()

        # Show bit string input in little-endian (LSB first) to match qubit state
        prefix = f"Error for a={input_a}" + ("" if n_bits_b == 0 else f", b={input_b}") + ": "
        if res_a != res_expected:
            raise Exception(f"{prefix}expected result a={res_expected}, got {res_a}")
        if n_bits_b > 0 and res_b != input_b:
            raise Exception(f"{prefix}the state of the register b changed to {res_b}")


####################################################################################################

def f_increment(a: int, n: int) -> int:
    return (a + 1) % (2 ** n)


@mark.parametrize("quantum_op", [ref.increment_1, ref.increment] if ref_available else [])
def test_increment(quantum_op):
    for n in range(1, 5):
        global log_message
        log_message = f"Testing {n=}"
        f = partial(f_increment, n=n)
        run_test_reversible(n, 0, QUInt, quantum_op, f)


####################################################################################################

def f_increment_power(a: int, n: int, p: int) -> int:
    return (a + 2 ** p) % (2 ** n)


@mark.parametrize("quantum_op", [ref.increment_power] if ref_available else [])
def test_increment_power(quantum_op):
    for n in range(1, 5):
        for p in range(n):
            global log_message
            log_message = f"Testing {n=}, {p=}"
            f = partial(f_increment_power, n=n, p=p)
            q = partial(quantum_op, p=p)
            run_test_reversible(n, 0, QUInt, q, f)


####################################################################################################

def f_add(a: int, b: int, n: int) -> int:
    return (a + b) % (2 ** n)


@mark.parametrize("quantum_op", [ref.increment_constant] if ref_available else [])
def test_increment_constant(quantum_op):
    for n in range(1, 5):
        for b in range(2 ** n):
            global log_message
            log_message = f"Testing {n=}, {b=}"
            f = partial(f_add, n=n, b=b)
            q = partial(quantum_op, b=b)
            run_test_reversible(n, 0, QUInt, q, f)


####################################################################################################

@mark.parametrize("qbk_class", [ref.NaiveAdd] if ref_available else [])
def test_naiveadd(qbk_class):
    for n in range(1, 5):
        for m in range(1, n):
            global log_message
            log_message = f"Testing {n=}, {m=}"
            f = partial(f_add, n=n)
            qbk = qbk_class()
            run_test_reversible(n, m, QUInt, qbk.compute, f)


####################################################################################################

def f_subtract(a: int, b: int, n: int) -> int:
    return (2 ** n + a - b) % (2 ** n)


@mark.parametrize("qbk_class", [ref.NaiveSubtract] if ref_available else [])
def test_naivesubtract(qbk_class):
    for n in range(1, 5):
        for m in range(1, n + 1):
            global log_message
            log_message = f"Testing {n=}, {m=}"
            f = partial(f_subtract, n=n)
            qbk = qbk_class()
            run_test_reversible(n, m, QUInt, qbk.compute, f)


####################################################################################################

def f_negate(a: int, n: int) -> int:
    return a if a == -2 ** (n-1) else -a


@mark.parametrize("quantum_op", [ref.negate] if ref_available else [])
def test_negate(quantum_op):
    for n in range(2, 5):
        global log_message
        log_message = f"Testing {n=}"
        f = partial(f_negate, n=n)
        run_test_reversible(n, 0, QInt, quantum_op, f)


####################################################################################################

def f_add_signed(a: int, b: int, n: int) -> int:
    sum = a + b
    if sum < -2 ** (n - 1):
        sum += 2 ** n
    if sum >= 2 ** (n - 1):
        sum -= 2 ** n
    return sum


@mark.parametrize("qbk_class", [ref.NaiveAddSigned] if ref_available else [])
def test_naiveaddsigned(qbk_class):
    for n in range(2, 5):
        global log_message
        log_message = f"Testing {n=}"
        f = partial(f_add_signed, n=n)
        qbk = qbk_class()
        run_test_reversible(n, n, QInt, qbk.compute, f)

####################################################################################################


@mark.parametrize("qbk_class", [ref.NaiveAddSignedExtension] if ref_available else [])
def test_naiveaddsignedextension(qbk_class):
    for n in range(2, 5):
        for m in range(2, n + 1):
            global log_message
            log_message = f"Testing {n=}, {m=}"
            f = partial(f_add_signed, n=n)
            qbk = qbk_class()
            run_test_reversible(n, m, QInt, qbk.compute, f)

