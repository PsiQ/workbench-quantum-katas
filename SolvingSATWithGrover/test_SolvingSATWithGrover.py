from functools import partial
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.filter_presets import BIT_DEFAULT
from pytest import mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_SolvingSATWithGrover as ref
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

def f_and(args: list[bool]) -> bool:
    return all(args)


def test_oracle_and(fun=ref.oracle_and if ref_available else None):
    for n in range(1, 5):
        print(f"Testing {n=}")
        run_test_reversible(n, n + 1, fun, f_and)


####################################################################################################

def f_or(args: list[bool]) -> bool:
    return any(args)


def test_oracle_or(fun=ref.oracle_or if ref_available else None):
    for n in range(1, 5):
        print(f"Testing {n=}")
        run_test_reversible(n, n + 1, fun, f_or)


####################################################################################################

def f_sat_clause(args: list[bool], literals: list[tuple[int, bool]]) -> bool:
    for (ind, pos) in literals:
        if pos and args[ind] or not pos and not args[ind]:
            return True
    return False

def test_oracle_sat_clause(fun=ref.oracle_sat_clause if ref_available else None):
    for num_inputs, literals in [
        (1, [(0, True)]),
        (1, [(0, False)]),
        (2, [(0, True), (1, True)]),
        (2, [(0, False), (1, True)]),
        (3, [(1, False), (2, False)]),
        (3, [(1, False), (2, False), (0, True)])
    ]:
        print(f"Testing {num_inputs=}, {literals=}")
        f = partial(f_sat_clause, literals=literals)
        quantum_op = partial(fun, literals=literals)
        run_test_reversible(num_inputs, num_inputs + 1, quantum_op, f)


####################################################################################################

def f_sat_formula(args: list[bool], clauses: list[list[tuple[int, bool]]]) -> bool:
    for clause in clauses:
        if not f_sat_clause(args, clause):
            return False
    return True


def test_oracle_sat_formula(fun=ref.oracle_sat_formula if ref_available else None):
    for num_inputs, clauses in [
            (1, [[(0, True)], [(0, False)]]), # 0 solutions
            (1, [[(0, False)]]),              # 1 solution
            (2, [[(0, True)], [(1, True)]]),  # 1 solution
            (2, [[(0, False), (1, False)], [(0, True), (1, True)]]), # 2 solutions
            (2, [[(0, False), (1, False)]]),  # 3 solutions
            (3, [[(2, False), (1, True)], [(2, True), (1, False)]]), # 4 solutions
        ]:
        print(f"Testing {num_inputs=}, {clauses=}")
        f = partial(f_sat_formula, clauses=clauses)
        quantum_op = partial(fun, clauses=clauses)
        run_test_reversible(num_inputs, num_inputs + len(clauses) + 1, quantum_op, f)


####################################################################################################

def f_exactly1one(args: list[bool]) -> bool:
    return sum(args) == 1


def test_oracle_exactly1one(fun=ref.oracle_exactly1one if ref_available else None):
    run_test_reversible(3, 7, fun, f_exactly1one)


####################################################################################################

def f_exactly1one_sat_clause(args: list[bool], literals: list[tuple[int, bool]]) -> bool:
    s = 0
    for (ind, pos) in literals:
        if pos and args[ind] or not pos and not args[ind]:
            s += 1
    return s == 1


def test_oracle_exactly1one_sat_clause(fun=ref.oracle_exactly1one_sat_clause if ref_available else None):
    for num_inputs, literals in [
        (3, [(0, True), (1, True), (2, False)]),
        (3, [(1, False), (2, False), (0, True)]),
        (4, [(3, True), (1, False), (2, True)]),
        (4, [(0, False), (2, True), (3, False)])
    ]:
        print(f"Testing {num_inputs=}, {literals=}")
        f = partial(f_exactly1one_sat_clause, literals=literals)
        quantum_op = partial(fun, literals=literals)
        run_test_reversible(num_inputs, num_inputs + 4, quantum_op, f)


####################################################################################################

def f_exactly1one_sat_formula(args: list[bool], clauses: list[list[tuple[int, bool]]]) -> bool:
    for clause in clauses:
        if not f_exactly1one_sat_clause(args, clause):
            return False
    return True


def test_oracle_exactly1one_sat_formula(fun=ref.oracle_exactly1one_sat_formula if ref_available else None):
    for num_inputs, clauses in [
            (3, [[(0, True), (1, True), (2, True)],  [(0, False), (1, True), (2, True)]]),    # 0 solutions
            (3, [[(0, True), (1, True), (2, False)], [(0, True), (1, False), (2, True)], [(0, False), (1, True), (2, True)]]), # 1 solutions
            (3, [[(0, True), (1, True), (2, False)], [(0, False), (1, True), (2, True)]]),    # 2 solutions
            (3, [[(0, False), (1, False), (2, False)]])                                       # 3 solutions
        ]:
        print(f"Testing {num_inputs=}, {clauses=}")
        f = partial(f_exactly1one_sat_formula, clauses=clauses)
        quantum_op = partial(fun, clauses=clauses)
        run_test_reversible(num_inputs, num_inputs + len(clauses) + 4, quantum_op, f)
