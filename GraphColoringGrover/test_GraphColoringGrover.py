from functools import partial
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.filter_presets import BIT_DEFAULT
from pytest import mark
from warnings import catch_warnings, filterwarnings

try:
    from importnb import Notebook
    with catch_warnings():
        # ignore only for this module/import
        filterwarnings("ignore", category=SyntaxWarning)
        filterwarnings("ignore", category=DeprecationWarning, module=r".*Workbook_GraphColoringGrover.*")
        # (or target the exact message)
        # filterwarnings("ignore", message=r"invalid escape sequence \\k", category=DeprecationWarning)
    
        with Notebook():
            import Workbook_GraphColoringGrover as ref
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

def bits_to_int(bits: list[bool]):
    return sum([bits[i] * 2 ** i for i in range(len(bits))])

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


############## Problem 1. ######################################################################################

# Graphs with 6+ vertices can take several seconds to be processed;
# in the interest of keeping test runtime reasonable we're limiting most of the testing to graphs with 5 vertices or fewer.
test_graphs = [
        (3, []),
        (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
        (5, [(4, 0), (2, 1), (3, 1), (3, 2)]),
        (5, [(0, 1), (1, 2), (1, 3), (3, 2), (4, 2), (3, 4)]),
        (5, [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]),
        (6, [(0, 1), (0, 2), (0, 4), (0, 5), (1, 2), (1, 3), (1, 5), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)])
    ]

def test_is_vertex_coloring_valid(fun=ref.is_vertex_coloring_valid if ref_available else None):
    test_colorings = [
        # Every coloring would pass on a disconnected graph of 3 vertices
        [([0, 0, 0], True), ([2, 1, 3], True)],
        [([0, 2, 1, 3], True), ([3, 0, 1, 2], True), ([0, 2, 1, 0], False)],
        [([0, 1, 2, 3, 4], True), ([0, 2, 1, 0, 3], True), ([1, 0, 1, 2, 1], False), ([0, 0, 0, 0, 0], False)],
        [([0, 1, 0, 2, 1], True), ([0, 2, 0, 1, 3], True), ([0, 1, 0, 1, 2], False)],
        [([1, 2, 3, 1, 2], True), ([1, 2, 3, 4, 1], False)]
    ]
    
    for ((V, edges), colorings) in zip(test_graphs[:-1], test_colorings):
        for (coloring, expected) in colorings:
            if fun(V, edges, coloring) != expected:
                raise Exception(f"Coloring {coloring} evaluated incorrectly for graph V = {V}, edges = {edges}: expected {expected}")


############## Problem 2. ######################################################################################

def test_read_coloring(fun=ref.read_coloring if ref_available else None):
    qpu = QPU(filters=BIT_DEFAULT)
    for n_bits in range(1, 4):
        for V in range(1, 4):
            n_inputs = (V * n_bits)
            qpu.reset(n_inputs)
            qs = Qubits(n_inputs, "qs", qpu)
            
            for input_mask in range(2 ** n_inputs):
                # Prepare the register in the input state
                binary_state = int_to_bool_array(input_mask, n_inputs)
                
                # Apply the mask to the qubit register
                qs.write(input_mask)

                result_colors = fun(n_bits, qs)

                # Un-apply the mask to the qubit register
                if input_mask != 0:
                    qs.x(input_mask)
                
                # Get the expected coloring by splitting binary_state into parts and converting them into integers (little-endian)
                partitions = [binary_state[i * n_bits:(i + 1) * n_bits] for i in range(V)]
                expected_colors = [bits_to_int(color_i) for color_i in partitions]

                # Verify the return value
                if len(result_colors) != V:
                    raise Exception(f"Unexpected number of colors for {V=}, {n_bits=}, state {binary_state}: {len(result_colors)}")

                for (expected, actual) in zip(expected_colors, result_colors):
                    if actual != expected:
                        raise Exception(f"Unexpected colors for {V=}, {n_bits=}, state {binary_state}: expected {expected_colors}, got {result_colors}")
                
                if qs.read() != 0:
                    raise Exception(f"Error for {V=}, {n_bits=}, state {binary_state}: the state of the input qubits changed")
                


############## Problem 3. ######################################################################################

def f_color_equality(x: list[bool]) -> bool:
    n_bits = len(x) // 2
    return x[:n_bits] == x[n_bits:]

def test_oracle_color_equality(fun=ref.oracle_color_equality if ref_available else None):
    qpu = QPU(filters=BIT_DEFAULT)
    for n_bits in range(1, 4):
        def oracle_wrapper(x: Qubits, y: Qubits) -> None:
            '''Wrapper to split register x in two arguments and pass to the oracle'''
            fun(x[:n_bits], x[n_bits:], y)

        run_test_reversible(qpu, 2 * n_bits, 2 * n_bits + 1, oracle_wrapper, f_color_equality)


############## Problem 4. ######################################################################################

def bits_to_coloring(V: int, bits: list[bool]) -> list[int]:
    return [bits[2 * i] + bits[2 * i + 1] * 2 for i in range(V)]

def is_vertex_coloring_valid(V: int, edges: list[tuple[int, int]], colors: list[int]) -> bool:
    for (v0, v1) in edges:
        if colors[v0] == colors[v1]:
            return False
    return True


def get_oracle_instructions(n_inputs: int, n_qubits: int, oracle: callable):
    '''Runs solution on a QPU and fetches the list of gate instructions produced for later replay.'''
    qpu = QPU(num_qubits=n_qubits, filters=['>>buffer>>'])
    x = Qubits(n_inputs, "x", qpu)
    y = Qubits(1, "y", qpu)
    oracle(x=x, y=y)

    # Check that there are no leftover auxiliary qubits - only x and y
    num_qubits = qpu._get_qubit_heap().allocated_mask.bit_count()
    if num_qubits != n_inputs + 1:
        raise Exception("Your solution should release all auxiliary qubits it allocates")

    instructions = qpu.get_instructions(format='cpp')
    # The first three instructions
    # are going to be reset and qubit allocations - skip them
    return instructions[3:]


def check_oracle_recognizes_coloring(
    V: int,                       # Number of vertices
    edges: list[tuple[int, int]], # Edges
    oracle: callable,             # Oracle that takes V, edges and registers x and y
    classical_oracle: callable    # Classical function that takes V, edges, and coloring (see problem 1)
):
    n_inputs = 2 * V
    n_qubits = 2 * V + 1 + len(edges) + max(V, len(edges))

    # The solution is a sequence of reversible gates that doesn't depend on the inputs - can run it just once and replay later
    oracle_instructions = get_oracle_instructions(n_inputs, n_qubits, partial(oracle, V=V, edges=edges))

    qpu = QPU(filters=BIT_DEFAULT)
    bit_sim = qpu.get_filter_by_name('>>bit-sim>>')
    qpu.enable_qubit_allocation_debugging()
    # Try all possible colorings of 4 colors on V vertices and check if they are calculated correctly.
    # Hack: fix the color of the first vertex to 00, since all colorings are agnostic to the specific colors used.
    for partial_coloring_mask in range(2 ** (n_inputs - 2)):
        qpu.reset(n_qubits)
        x = Qubits(n_inputs, "x", qpu)
        y = Qubits(1, "y", qpu)

        complete_coloring_mask = partial_coloring_mask << 2

        # Prepare quantum input
        x.write(complete_coloring_mask)
        qpu.flush()

        # Apply the oracle
        bit_sim._put_native(oracle_instructions)

        # Evaluate classical function on the classical input
        coloring = bits_to_coloring(V, int_to_bool_array(complete_coloring_mask, n_inputs))
        res_expected = int(classical_oracle(V, edges, coloring))

        # Compare the results of classical and quantum computations
        res_x = x.read()
        res_y = y.read()

        if res_x != complete_coloring_mask:
            raise Exception(f"Error for x={complete_coloring_mask} ({coloring}): the state of the input qubits changed")
        if res_y != res_expected:
            raise Exception(f"Error for x={complete_coloring_mask} ({coloring}): expected {res_expected}, got {res_y}")


def test_oracle_vertex_coloring(fun=ref.oracle_vertex_coloring if ref_available else None):
    for V, edges in test_graphs[:-2]:
        check_oracle_recognizes_coloring(V, edges, fun, is_vertex_coloring_valid)

            
############## Problem 5. ######################################################################################

def test_is_weak_coloring_valid(fun=ref.is_weak_coloring_valid if ref_available else None):
    test_colorings = [
        # Every coloring would pass on a disconnected graph of 3 vertices
        [([0, 0, 0], True), ([2, 1, 3], True)],
        # Every coloring  would pass on a fully connected graph of 4 vertices,
        # except for the last coloring in which all vertices are of the same color.
        [([0, 2, 1, 3], True), ([3, 0, 1, 0], True), ([0, 0, 0, 0], False)],
        # The colorings for 5-vertex graphs:
        # - the first one is invalid for all graphs except disconnected
        # - the second one is valid for all types of graphs regardless of their structure
        # - two colorings that is valid or invalid depending on the graph        
        [([0, 0, 0, 0, 0], False), ([0, 1, 2, 3, 4], True), ([0, 1, 1, 2, 0], False), ([0, 0, 1, 1, 1], True)],
        [([0, 0, 0, 0, 0], False), ([0, 1, 2, 3, 4], True), ([0, 1, 1, 2, 0], True), ([0, 0, 1, 1, 1], False)],
        [([0, 0, 0, 0, 0], False), ([0, 1, 2, 3, 4], True), ([0, 1, 1, 2, 0], True), ([0, 0, 1, 1, 1], True)]
    ]
    for ((V, edges), colorings) in zip(test_graphs[:-1], test_colorings):
        for (coloring, expected) in colorings:
            if fun(V, edges, coloring) != expected:
                raise Exception(f"Coloring {coloring} evaluated incorrectly for graph V = {V}, edges = {edges}: expected {expected}")


############## Problem 6. ######################################################################################

def is_weak_coloring_valid_one_vertex(V: int, edges: list[tuple[int, int]], colors: list[int], vertex: int) -> bool:
    neighbor_count = 0
    has_different_neighbor = False

    for start, end in edges:
        if start == vertex or end == vertex:
            neighbor_count += 1
            if colors[start] != colors[end]:
                has_different_neighbor = True
    
    return neighbor_count == 0 or has_different_neighbor


def test_oracle_weak_coloring_one_vertex(fun=ref.oracle_weak_coloring_one_vertex if ref_available else None):
    for V, edges in test_graphs[:-1]:
        for vertex in range(V):
            oracle_vertex = partial(fun, vertex=vertex)
            fun_vertex = partial(is_weak_coloring_valid_one_vertex, vertex=vertex)
            check_oracle_recognizes_coloring(V, edges, oracle_vertex, fun_vertex)


############## Problem 7. ######################################################################################

def is_weak_coloring_valid(V: int, edges: list[(int,int)], colors: list[int]) -> bool:
    for v in range(V):
        if not is_weak_coloring_valid_one_vertex(V, edges, colors, v):
            return False
    return True

def test_oracle_weak_coloring(fun=ref.oracle_weak_coloring if ref_available else None):
    for V, edges in test_graphs[:-2]:
        check_oracle_recognizes_coloring(V, edges, fun, is_weak_coloring_valid)
