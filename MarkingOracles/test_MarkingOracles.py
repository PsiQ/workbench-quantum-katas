from functools import partial
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.filter_presets import BIT_DEFAULT
from pytest import mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_MarkingOracles as ref
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


def get_instructions(n_inputs: int, n_qubits: int, quantum_op):
    '''Runs solution on a QPU and fetches the list of gate instructions produced for later replay.'''
    qpu = QPU(num_qubits=n_qubits, filters=BIT_DEFAULT)
    x = Qubits(n_inputs, "x", qpu)
    y = Qubits(1, "y", qpu)
    quantum_op(x, y)

    # Check that there are no leftover auxiliary qubits - only x and y
    num_qubits = qpu._get_qubit_heap().allocated_mask.bit_count()
    if num_qubits != n_inputs + 1:
        raise Exception("Your solution should release all auxiliary qubits it allocates")

    instructions = qpu.get_instructions(format='cpp')
    # The first three instructions
    # are going to be reset and qubit allocations - skip them
    return instructions[3:]


def run_test_reversible(n_inputs: int, n_qubits: int, quantum_op, f):
    instructions = get_instructions(n_inputs, n_qubits, quantum_op)

    qpu = QPU(filters=BIT_DEFAULT)
    bit_sim = qpu.get_filter_by_name('>>bit-sim>>')
    qpu.enable_qubit_allocation_debugging()
    for input_mask in range(2 ** n_inputs):
        qpu.reset(n_qubits)
        x = Qubits(n_inputs, "x", qpu)
        y = Qubits(1, "y", qpu)

        # Prepare quantum input
        x.write(input_mask)
        qpu.flush()

        # Run the reversible computation as instructions replay
        bit_sim._put_native(instructions)

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


####################################################################################################

def f_kth_bit(args: list[bool], k: int) -> bool:
    return args[k]


def test_oracle_kth_bit(fun=ref.oracle_kth_bit if ref_available else None):
    for n in range(1, 5):
        for k in range(n):
            print(f"Testing {n=}, {k=}")
            f = partial(f_kth_bit, k=k)
            quantum_op = partial(fun, k=k)
            run_test_reversible(n, n + 1, quantum_op, f)


####################################################################################################

def f_parity(args: list[bool]) -> bool:
    return sum(args) % 2 == 1


def test_oracle_parity(fun=ref.oracle_parity if ref_available else None):
    for n in range(1, 5):
        print(f"Testing {n=}")
        run_test_reversible(n, n + 1, fun, f_parity)


####################################################################################################

def f_product(args: list[bool], r: list[bool]) -> bool:
    return sum(arg if bit else 0 for (arg, bit) in zip(args, r)) % 2 == 1


def test_oracle_product(fun=ref.oracle_product if ref_available else None):
    for n in range(2, 5):
        for mask in range(2 ** n):
            r = int_to_bool_array(mask, n)
            print(f"Testing {n=}, {r=}")
            f = partial(f_product, r=r)
            quantum_op = partial(fun, r=r)
            run_test_reversible(n, n + 1, quantum_op, f)


####################################################################################################

def f_product_negation(args: list[bool], r: list[bool]) -> bool:
    return sum(arg if bit else not arg for (arg, bit) in zip(args, r)) % 2


def test_oracle_product_negation(fun=ref.oracle_product_negation if ref_available else None):
    for n in range(2, 5):
        for mask in range(2 ** n):
            r = int_to_bool_array(mask, n)
            print(f"Testing {n=}, {r=}")
            f = partial(f_product_negation, r=r)
            quantum_op = partial(fun, r=r)
            run_test_reversible(n, n + 1, quantum_op, f)


####################################################################################################

def f_palindrome(args: list[bool]) -> bool:
    return args == args[::-1]


def test_oracle_palindrome(fun=ref.oracle_palindrome if ref_available else None):
    for n in range(2, 7):
        print(f"Testing {n=}")
        run_test_reversible(n, n + 1, fun, f_palindrome)


####################################################################################################

def f_periodic_p(args: list[bool], p: int) -> bool:
    n = len(args)
    return args[:n - p] == args[p:]


def test_oracle_periodic_p(fun=ref.oracle_periodic_p if ref_available else None):
    for n in range(1, 5):
        for p in range(1, n):
            print(f"Testing {n=}, {p=}")
            f = partial(f_periodic_p, p=p)
            quantum_op = partial(fun, p=p)
            run_test_reversible(n, n + 1, quantum_op, f)


####################################################################################################

def f_periodic(args: list[bool]) -> bool:
    N = len(args)
    for P in range(1, N):
        if f_periodic_p(args, P):
            return True
    return False


def test_oracle_periodic(fun=ref.oracle_periodic if ref_available else None):
    for n in range(2, 6):
        print(f"Testing {n=}")
        run_test_reversible(n, 2 * n + 1, fun, f_periodic)


####################################################################################################

def f_contains_substring_at_p(args: list[bool], pattern: list[bool], p: int) -> bool:
    return args[p:p + len(pattern)] == pattern


def test_oracle_contains_substring_at_p(fun=ref.oracle_contains_substring_at_p if ref_available else None):
    for (n, p, pattern) in [
        (2, 1, [True]),
        (3, 0, [False, True]),
        (4, 1, [True, True, False]),
        (5, 3, [False])
    ]:
        print(f"Testing {n=}, {pattern=}, {p=}")
        f = partial(f_contains_substring_at_p, pattern=pattern, p=p)
        quantum_op = partial(fun, pattern=pattern, p=p)
        run_test_reversible(n, n + 1, quantum_op, f)


####################################################################################################

def f_pattern_matching(args: list[bool], indices: list[int], pattern: list[bool]) -> bool:
    return [args[ind] for ind in indices] == pattern


def test_oracle_pattern_matching(fun=ref.oracle_pattern_matching if ref_available else None):
    for (n, indices, pattern) in [
        (2, [1], [True]),
        (3, [0, 2], [False, True]),
        (4, [1, 3], [True, False]),
        (5, [0, 1, 4], [True, True, False])
    ]:
        print(f"Testing {n=}, {indices=}, {pattern=}")
        f = partial(f_pattern_matching, indices=indices, pattern=pattern)
        quantum_op = partial(fun, indices=indices, pattern=pattern)
        run_test_reversible(n, n + 1, quantum_op, f)


####################################################################################################

def f_contains_substring(args: list[bool], pattern: list[bool]) -> bool:
    for p in range(len(args) - len(pattern) + 1):
        if f_contains_substring_at_p(args, pattern, p):
            return True
    return False


def test_oracle_contains_substring(fun=ref.oracle_contains_substring if ref_available else None):
    for (n, pattern) in [
        (2, [True]),
        (3, [False, True]),
        (4, [True, True, False]),
        (5, [False])
    ]:
        print(f"Testing {n=}, {pattern=}")
        f = partial(f_contains_substring, pattern=pattern)
        quantum_op = partial(fun, pattern=pattern)
        run_test_reversible(n, 2 * n + 1, quantum_op, f)


####################################################################################################

def f_balanced(args: list[bool]) -> bool:
    return sum(args) == len(args) // 2


def test_oracle_balanced(fun=ref.oracle_balanced if ref_available else None):
    for n in range(2, 7, 2):
        print(f"Testing {n=}")
        run_test_reversible(n, 2 * n + 1, fun, f_balanced)


####################################################################################################

def f_majority(args: list[bool]) -> bool:
    return sum(args) > len(args) // 2


def test_oracle_majority(fun=ref.oracle_majority if ref_available else None):
    for n in [3, 5, 7]:
        print(f"Testing {n=}")
        run_test_reversible(n, 2 * n + 1, fun, f_majority)


####################################################################################################

def f_bit_sum_divisible_by_three(args: list[bool]) -> bool:
    return sum(args) % 3 == 0


def test_oracle_bit_sum_divisible_by_three(fun=ref.oracle_bit_sum_divisible_by_three if ref_available else None):
    for n in range(2, 6):
        print(f"Testing {n=}")
        run_test_reversible(n, 2 * n + 1, fun, f_bit_sum_divisible_by_three)


####################################################################################################

def f_number_divisible_by_three(args: list[bool]) -> bool:
    return sum([2 ** ind if args[ind] else 0 for ind in range(len(args))]) % 3 == 0


def test_oracle_number_divisible_by_three(fun=ref.oracle_number_divisible_by_three if ref_available else None):
    for n in range(2, 6):
        print(f"Testing {n=}")
        run_test_reversible(n, 2 * n + 1, fun, f_number_divisible_by_three)
