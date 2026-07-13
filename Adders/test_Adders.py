from functools import partial
from itertools import product
from typing import Callable
from psiqdk.workbench import QPU, Qubits, QUInt, QInt
from psiqdk.workbench.filter_presets import BIT_DEFAULT
from pytest import mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_Adders as ref
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

def get_instructions(n_inputs: int, n_qubits: int, quantum_op: Callable):
    '''Runs solution on a QPU and fetches the list of gate instructions produced for later replay.'''
    qpu = QPU(num_qubits=n_qubits, filters=BIT_DEFAULT)
    a = QUInt(n_inputs, "a", qpu)
    b = QUInt(n_inputs, "b", qpu)
    quantum_op(a, b)

    # Check that there are no leftover auxiliary qubits - only a and b
    num_qubits = qpu._get_qubit_heap().allocated_mask.bit_count()
    if num_qubits != 2 * n_inputs:
        raise Exception("Your solution should release all auxiliary qubits it allocates")

    instructions = qpu.get_instructions(format='cpp')
    # The first three instructions
    # are going to be reset and qubit allocations - skip them
    return instructions[3:]


def run_test_reversible(n_inputs: int, n_aux: int, quantum_op: Callable, f: Callable):
    '''Runs test of a reversible computation, assuming two unsigned input registers of the same size
       and possibly auxiliary qubits allocated.'''
    n_qubits = n_inputs * 2 + n_aux
    instructions = get_instructions(n_inputs, n_qubits, quantum_op)

    qpu = QPU(filters=BIT_DEFAULT)
    bit_sim = qpu.get_filter_by_name('>>bit-sim>>')
    qpu.enable_qubit_allocation_debugging()
    for input in range(2 ** (2 * n_inputs)):
        # Reset QPU and allocate the registers
        qpu.reset(n_qubits)
        a = QUInt(n_inputs, "a", qpu)
        b = QUInt(n_inputs, "b", qpu)

        # Prepare quantum inputs a and b (always unsigned)
        input_a = input % (2 ** n_inputs)
        a.write(input_a)

        input_b = (input >> n_inputs)
        b.write(input_b)

        # Flush the whole QPU before adding instructions to a specific filter
        qpu.flush()

        # Run the reversible computation as instructions replay
        bit_sim._put_native(instructions)

        # Evaluate classical function on the classical input
        res_expected = f(input_a, input_b)

        # Compare the results of classical and quantum computations
        res_a = a.read()
        res_b = b.read()

        # Show bit string input in little-endian (LSB first) to match qubit state
        prefix = f"Error for a={input_a}, b={input_b}: "
        if res_a != res_expected:
            raise Exception(f"{prefix}expected result a={res_expected}, got {res_a}")
        if res_b != input_b:
            raise Exception(f"{prefix}the state of the register b changed to {res_b}")


####################################################################################################

def f_add(a: int, b: int, n: int) -> int:
    return (a + b) % (2 ** n)


@mark.parametrize("quantum_op", [ref.sum_two_bits] if ref_available else [])
def test_sum_two_bits(quantum_op):
    n = 1
    f = partial(f_add, n=n)
    run_test_reversible(n, 0, quantum_op, f)

        
####################################################################################################

@mark.parametrize("quantum_op", [ref.carry_two_bits] if ref_available else [])
def test_carry_two_bits(quantum_op):
    # Custom test for custom-shaped block that uses multiple registers
    qpu = QPU(filters=BIT_DEFAULT)
    for input_a, input_b in product([0, 1], repeat=2):
        # Reset QPU and allocate the registers
        qpu.reset(3)
        a = QUInt(1, "a", qpu)
        b = QUInt(1, "b", qpu)
        c = QUInt(1, "c", qpu)
        # Prepare quantum inputs a and b
        a.write(input_a)
        b.write(input_b)
        # Run the computation
        quantum_op(a, b, c)
        res_expected = (input_a + input_b) >> 1
        res_a = a.read()
        res_b = b.read()
        res_c = c.read()
        # Compare the results of classical and quantum computations
        prefix = f"Error for a={input_a}, b={input_b}: "
        if res_c != res_expected:
            raise Exception(f"{prefix}expected result c={res_expected}, got {res_c}")
        if res_a != input_a:
            raise Exception(f"{prefix}the state of the register a changed to {res_a}")
        if res_b != input_b:
            raise Exception(f"{prefix}the state of the register b changed to {res_b}")


####################################################################################################

@mark.parametrize("quantum_op", [ref.sum_three_bits] if ref_available else [])
def test_sum_three_bits(quantum_op):
    # Custom test for custom-shaped block that uses multiple registers
    qpu = QPU(filters=BIT_DEFAULT)
    for input_a, input_b, input_c in product([0, 1], repeat=3):
        # Reset QPU and allocate the registers
        qpu.reset(3)
        a = QUInt(1, "a", qpu)
        b = QUInt(1, "b", qpu)
        c = QUInt(1, "c", qpu)
        # Prepare quantum inputs a and b
        a.write(input_a)
        b.write(input_b)
        c.write(input_c)
        # Run the computation
        quantum_op(a, b, c)
        res_expected = (input_a + input_b + input_c) % 2
        res_a = a.read()
        res_b = b.read()
        res_c = c.read()
        # Compare the results of classical and quantum computations
        prefix = f"Error for a={input_a}, b={input_b}, c={input_c}: "
        if res_a != res_expected:
            raise Exception(f"{prefix}expected result a={res_expected}, got {res_a}")
        if res_b != input_b:
            raise Exception(f"{prefix}the state of the register b changed to {res_b}")
        if res_c != input_c:
            raise Exception(f"{prefix}the state of the register c changed to {res_c}")


####################################################################################################

@mark.parametrize("quantum_op", [ref.carry_three_bits] if ref_available else [])
def test_carry_three_bits(quantum_op):
    # Custom test for custom-shaped block that uses multiple registers
    qpu = QPU(filters=BIT_DEFAULT)
    for input_a, input_b, input_c in product([0, 1], repeat=3):
        # Reset QPU and allocate the registers
        qpu.reset(4)
        a = QUInt(1, "a", qpu)
        b = QUInt(1, "b", qpu)
        c = QUInt(1, "c", qpu)
        d = QUInt(1, "d", qpu)
        # Prepare quantum inputs a and b
        a.write(input_a)
        b.write(input_b)
        c.write(input_c)
        # Run the computation
        quantum_op(a, b, c, d)
        res_expected = (input_a + input_b + input_c) >> 1
        res_a = a.read()
        res_b = b.read()
        res_c = c.read()
        res_d = d.read()
        # Compare the results of classical and quantum computations
        prefix = f"Error for a={input_a}, b={input_b}, c={input_c}: "
        if res_d != res_expected:
            raise Exception(f"{prefix}expected result d={res_expected}, got {res_d}")
        if res_a != input_a:
            raise Exception(f"{prefix}the state of the register a changed to {res_a}")
        if res_b != input_b:
            raise Exception(f"{prefix}the state of the register b changed to {res_b}")
        if res_c != input_c:
            raise Exception(f"{prefix}the state of the register c changed to {res_c}")


####################################################################################################

@mark.parametrize("qbk_class", [ref.RippleCarryAdderTwoBit] if ref_available else [])
def test_ripplecarryaddertwobit(qbk_class):
    n = 2
    f = partial(f_add, n=n)
    qbk = qbk_class()
    run_test_reversible(n, 1, qbk.compute, f)


####################################################################################################

@mark.parametrize("qbk_class", [ref.RippleCarryAdder] if ref_available else [])
def test_ripplecarryadder(qbk_class):
    for n in range(2, 6):
        global log_message
        log_message = f"Testing {n=}"
        f = partial(f_add, n=n)
        qbk = qbk_class()
        run_test_reversible(n, n - 1, qbk.compute, f)


####################################################################################################

@mark.parametrize("quantum_op", [ref.maj] if ref_available else [])
def test_maj(quantum_op):
    # Custom test for custom-shaped block that uses multiple registers
    qpu = QPU(filters=BIT_DEFAULT)
    for input_a, input_b, input_c in product([0, 1], repeat=3):
        # Reset QPU and allocate the registers
        qpu.reset(3)
        a = QUInt(1, "a", qpu)
        b = QUInt(1, "b", qpu)
        c = QUInt(1, "c", qpu)
        # Prepare quantum inputs a, b, c
        a.write(input_a)
        b.write(input_b)
        c.write(input_c)
        # Run the computation
        quantum_op(a, b, c)
        res_expected_a = (input_a + input_b) % 2
        res_expected_b = (input_a + input_b + input_c) >> 1
        res_expected_c = (input_c + input_b) % 2
        res_a = a.read()
        res_b = b.read()
        res_c = c.read()
        # Compare the results of classical and quantum computations
        prefix = f"Error for a={input_a}, b={input_b}, c={input_c}: "
        if res_a != res_expected_a or res_b != res_expected_b or res_c != res_expected_c:
            raise Exception(f"{prefix}expected results a={res_expected_a}, b={res_expected_b}, c={res_expected_c},"
                            f" got a={res_a}, b={res_b}, c={res_c}")


####################################################################################################

@mark.parametrize("quantum_op", [ref.uma] if ref_available else [])
def test_uma(quantum_op):
    # Custom test for custom-shaped block that uses multiple registers
    qpu = QPU(filters=BIT_DEFAULT)
    for bit_a, bit_b, bit_c in product([0, 1], repeat=3):
        # Reset QPU and allocate the registers
        qpu.reset(3)
        a = QUInt(1, "a", qpu)
        b = QUInt(1, "b", qpu)
        c = QUInt(1, "c", qpu)
        # Prepare quantum inputs a, b, c
        input_a = (bit_a + bit_b) % 2
        input_b = (bit_a + bit_b + bit_c) >> 1
        input_c = (bit_b + bit_c) % 2
        a.write(input_a)
        b.write(input_b)
        c.write(input_c)
        # Run the computation
        quantum_op(a, b, c)
        res_expected_a = (bit_a + bit_b + bit_c) % 2
        res_expected_b = bit_b
        res_expected_c = bit_c
        res_a = a.read()
        res_b = b.read()
        res_c = c.read()
        # Compare the results of classical and quantum computations
        prefix = f"Error for a={input_a}, b={input_b}, c={input_c} (inputs {input_a}, {input_b}, {input_c}): "
        if res_a != res_expected_a or res_b != res_expected_b or res_c != res_expected_c:
            raise Exception(f"{prefix}expected results a={res_expected_a}, b={res_expected_b}, c={res_expected_c},"
                            f" got a={res_a}, b={res_b}, c={res_c}")


####################################################################################################

@mark.parametrize("qbk_class", [ref.CuccaroAdderOneBit] if ref_available else [])
def test_cuccaroadderonebit(qbk_class):
    n = 1
    f = partial(f_add, n=n)
    qbk = qbk_class()
    run_test_reversible(n, 1, qbk.compute, f)


####################################################################################################

@mark.parametrize("qbk_class", [ref.CuccaroAdderTwoBit] if ref_available else [])
def test_cuccaroaddertwobit(qbk_class):
    n = 2
    f = partial(f_add, n=n)
    qbk = qbk_class()
    run_test_reversible(n, 1, qbk.compute, f)


####################################################################################################

@mark.parametrize("qbk_class", [ref.CuccaroAdder] if ref_available else [])
def test_cuccaroadder(qbk_class):
    for n in range(2, 6):
        global log_message
        log_message = f"Testing {n=}"
        f = partial(f_add, n=n)
        qbk = qbk_class()
        run_test_reversible(n, 1, qbk.compute, f)
