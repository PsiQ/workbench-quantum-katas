from itertools import product
from math import sqrt, cos, sin
from random import randint, seed, uniform
from typing import Callable, Sequence
from psiqdk.workbench import QPU, Qubits
from psiqdk.workbench.filter_presets import BIT_DEFAULT
from pytest import approx, mark
from warnings import catch_warnings
from functools import partial

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_ArbitraryStatePreparation as ref
    ref_available = True
except ImportError:
    ref_available = False
    # Skip all tests in this file - pytest checks reference solutions and that won't work without these imports
    pytestmark = mark.skip("No importnb/reference file available")


log_message = ""

def problem(arg):
    try:
        # Build test name
        test_name = "test_" + arg.__name__.lower()
        # Find test function; if none found, raise an exception
        test_func = globals()[test_name]
    except KeyError:
        print(f"Test {test_name} not found")
    else:
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


def check_state_vector(
    fun: Callable,                                # Callable that is being tested
    n_qubits: int,                                # Number of qubits in the main register
    expected_vector: Sequence[complex | float],   # State vector it should prepare
    *,
    n_aux: int=0,                                 # Number of auxiliary qubit(s) the solution is allowed to use
    initial_vector: Sequence[float] | None = None # The initial state (before the solution is called)
) -> None:
    # Construct the qpu and register
    total_num_qubits = n_qubits + n_aux   # Will be unnecessary once auto-resize is in
    qpu = QPU(num_qubits=total_num_qubits)
    qpu.enable_qubit_allocation_debugging()
    reg = Qubits(n_qubits, 'reg', qpu)

    if initial_vector is not None:
        reg.push_state(initial_vector)

    fun(reg)

    # Extract the state vector
    actual_vector = qpu.pull_state()

    # Check that there are no leftover auxiliary qubits, only reg
    num_qubits_after = qpu._get_qubit_heap().allocated_mask.bit_count()
    if num_qubits_after > n_qubits:
        raise ValueError("Any auxiliary qubits allocated should be returned to the |0⟩ state and released")

    if any([abs(amp) > 1e-9 for amp in actual_vector[2 ** n_qubits:]]):
        raise ValueError("Any auxiliary qubits allocated should be returned to the |0⟩ state and released")
    
    # Keep only the amplitudes of the main qubits
    actual_vector = actual_vector[:2 ** n_qubits]

    if actual_vector != approx(expected_vector):
        raise ValueError("State vectors should be equal\n"
                         "    Expected state vector:\n"
                         f"    {[round(val.real, 6) for val in expected_vector]}\n"
                         "    Actual state vector:\n"
                         f"    {[round(float(val.real), 6) for val in actual_vector]}"
                         )


#####################################################################################################################################

one_qubit_states = [
        [1., 0.],
        [0., 1.],
        [-1., 0.],
        [0., -1.],
        [1/sqrt(2), 1/sqrt(2)],
        [1/sqrt(2), -1/sqrt(2)],
        [-1/sqrt(2), 1/sqrt(2)],
        [-1/sqrt(2), -1/sqrt(2)],
        [0.6, 0.8],
        [0.6, -0.8],
        [-0.6, 0.8],
        [-0.6, -0.8]
    ]

@mark.parametrize("fun", [ref.prepare_one_qubit_state] if ref_available else [])
def test_prepare_one_qubit_state(fun):
    global log_message
    for alpha, beta in one_qubit_states:
        log_message = f"Testing {alpha=}, {beta=}"
        fun_alpha_beta = partial(fun, alpha=alpha, beta=beta)
        check_state_vector(fun_alpha_beta, 1, [alpha, beta])


#####################################################################################################################################

@mark.parametrize("fun", [ref.prepare_conditional_state] if ref_available else [])
def test_prepare_conditional_state(fun):
    global log_message
    for second_qubit_state in one_qubit_states:
        for alpha, beta in one_qubit_states:
            for c in [0, 1]:
                log_message = f"Testing |ψ⟩ = {second_qubit_state[0]}⋅|0⟩ + {second_qubit_state[1]}⋅|1⟩, {alpha=}, {beta=}, {c=}"
                fun_alpha_beta = partial(fun, alpha=alpha, beta=beta, c=c)
                initial_vector = [second_qubit_state[0], 0, second_qubit_state[1], 0]
                expected_vector = [second_qubit_state[0] * alpha, second_qubit_state[0] * beta, second_qubit_state[1], 0] if c == 0 else \
                                  [second_qubit_state[0], 0, second_qubit_state[1] * alpha, second_qubit_state[1] * beta]
                check_state_vector(fun_alpha_beta, 2, expected_vector, initial_vector=initial_vector)


#####################################################################################################################################

@mark.parametrize("fun", [ref.prepare_three_basis_states_two_qubits] if ref_available else [])
def test_prepare_three_basis_states_two_qubits(fun):
    global log_message
    for a in [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [1/sqrt(2), 1/sqrt(4), 1/sqrt(4)],
        [1/sqrt(3), -1/sqrt(3), 1/sqrt(3)],
        [1/sqrt(2), 1/sqrt(2), 0],
        [0, -1/sqrt(2), 1/sqrt(2)],
        [-1/sqrt(6), -1/sqrt(2), -1/sqrt(3)],
    ]:
        log_message = f"Testing {a=}"
        fun_a = partial(fun, a=a)
        expected_vector = a + [0]
        check_state_vector(fun_a, 2, expected_vector)


#####################################################################################################################################

two_qubit_states = [ 
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [-1., 0., 0., 0.],
        [0., -1., 0., 0.],
        [0., 0., -1., 0.],
        [0., 0., 0., -1.],        
        [0.5, 0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5, -0.5],
        [0.5, -0.5, 0.5, 0.5],
        [0.5, 0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [1. / sqrt(2.), 0., 0., 1. / sqrt(2.)],
        [1. / sqrt(2.), 0., 0., -1. / sqrt(2.)],
        [0., 1. / sqrt(2.), 1. / sqrt(2.), 0.],
        [0., 1. / sqrt(2.), -1. / sqrt(2.), 0.],
        [0.36, 0.48, 0.64, -0.48],
        [1. / sqrt(3.), -1. / sqrt(3.), 1. / sqrt(3.), 0.]
    ]

@mark.parametrize("fun", [ref.prepare_two_qubit_state] if ref_available else [])
def test_prepare_two_qubit_state(fun):
    global log_message
    for a in two_qubit_states:
        log_message = f"Testing {a=}"
        fun_a = partial(fun, a=a)
        check_state_vector(fun_a, 2, a)


#####################################################################################################################################

@mark.parametrize("qbk_class", [ref.NaiveStatePrep] if ref_available else [])
def test_naivestateprep(qbk_class):
    global log_message
    # Reuse one- and two-qubit test cases
    tests = list(product([1], one_qubit_states)) + list(product([2], two_qubit_states))

    # Generate random multi-qubit test case
    seed(42)
    for n in range(3, 5):
        for _ in range(10):
            amps = [uniform(-1.0, 1.0) for _ in range(2 ** n)]
            norm = sqrt(sum(aj ** 2 for aj in amps))
            a = [aj / norm for aj in amps]
            tests.append((n, a))
    
    for n, a in tests:
        log_message = f"Testing {a=}"
        qbk = qbk_class()
        fun_a = partial(qbk.compute, a=a)
        check_state_vector(fun_a, n, a)


#####################################################################################################################################

@mark.parametrize("fun", [ref.discretize_probabilities] if ref_available else [])
def test_discretize_probabilities(fun):
    global log_message

    tests = [
        # Already-discretized inputs
        ([0.5, 0.5], 1),
        ([0.75, 0.25], 2),
        ([0.5, 0.25, 0.25], 2),
        ([0.125, 0.25, 0.375, 0.25], 3),
        ([0.5, 0.25, 0.25], 3),
        # Non-discretized inputs that round nicely (results of rounding add up to 1)
        ([0.4, 0.6], 1),
        ([0.45, 0.2, 0.35], 2),
        # Non-discretized inputs that do not round nicely (results of rounding add up to something other than 1)
        ([0.3, 0.3, 0.3, 0.1], 2),      # Rounds down all
        ([0.3, 0.3, 0.3, 0.1], 3),      # Rounds down more than up
        ([0.5, 0.1, 0.1, 0.1, 0.2], 2), # Rounds down more than up
        ([0.2, 0.2, 0.2, 0.2, 0.2], 2), # Rounds up all
        ([0.2, 0.2, 0.2, 0.2, 0.2], 3), # Rounds up all
        ([0.3, 0.2, 0.2, 0.2, 0.1], 2), # Rounds up more than down
    ]

    # Add random tests
    seed(42)
    for _ in range(20):
        b = randint(1, 4)
        n = randint(2, 8)
        raw_nums = [uniform(0, 1) for _ in range(n)]
        s = sum(raw_nums)
        probs = [r / s for r in raw_nums]
        tests.append((probs, b))

    # Let's be a little paranoid about the generated tests
    for probs, _ in tests:
        if abs(sum(probs) - 1) > 1e-6:
            print(probs)
            raise RuntimeError("Tests are incorrect, sorry!")

    # Actual testing
    for probs, b in tests:
        log_message = f"Testing {probs=}, {b=}"
        N = len(probs)
        p2b = 1 << b
        discrete_probs = fun(probs, b)

        if len(discrete_probs) != N:
            raise ValueError(f"Returned list should have the same length as `probs`: expected {N}, got {len(discrete_probs)}")

        sum_dp = 0
        for p, dp in zip(probs, discrete_probs):
            if not isinstance(dp, int):
                raise ValueError(f"Each element of the returned list should be an integer, got {dp}")
            if dp < 0 or dp > p2b:
                raise ValueError(f"Each element of the returned list should be between 0 and {p2b}, got {dp}")
            if abs(dp / p2b - p) > 1 / p2b:
                raise ValueError(f"Each element of the returned list, divided by {p2b} should be within {1 / p2b} of its input counterpart. "
                                 f"For {p}, got {dp / p2b}")
            sum_dp += dp

        if sum_dp != p2b:
            raise ValueError(f"The sum of all elements of the returned list should be {p2b}, got {sum_dp}")



#####################################################################################################################################

@mark.parametrize("fun", [ref.repack] if ref_available else [])
def test_repack(fun):
    global log_message

    tests = [
        (1, 0, [0, 2]),
        (1, 0, [1, 1]),
        (1, 1, [4, 0]),
        (1, 1, [1, 3]),
        (1, 1, [2, 2]),
        (2, 0, [1, 1, 1, 1]),
        (2, 0, [0, 2, 0, 2]),
        (2, 0, [0, 3, 1, 0]),
        (2, 2, [0, 16, 0, 0]),
        (2, 2, [8, 4, 0, 4]),
        (2, 2, [7, 3, 1, 5]),
        (2, 2, [4, 4, 4, 4]),
    ]

    # Add random tests
    seed(42)
    for _ in range(42):
        n = randint(2, 5)
        b = randint(2, 5)
        # Values must be 1 << n numbers that add up to 1 << (n + b)
        points = [randint(0, 1 << (n + b)) for _ in range((1 << n) - 1)]
        points = [0] + sorted(points) + [1 << (n + b)]
        values = [points[i + 1] - points[i] for i in range(1 << n)]
        tests.append((n, b, values))

    # Let's be a little paranoid about the generated tests
    for n, b, values in tests:
        if len(values) != 1 << n or sum(values) != 1 << (n + b):
            print(values)
            raise RuntimeError("Tests are incorrect, sorry!")

    for n, b, values in tests:
        log_message = f"Testing {n=}, {b=}, {values=}"
        # Do a deep copy of the input list - the solution can modify it at will
        repacked_values = fun(n, b, values.copy())
        log_message += f"\nRepacked values {repacked_values}"

        if len(repacked_values) != len(values):
            raise ValueError(f"The length of repacked values should be the same as the length of original values: "
                             f"original values had length {len(values)}, repacked values has length {len(repacked_values)}"
                             )

        avg_column_height = 1 << b
        reconstructed_original_values = [0] * len(values)
        for ind, (alias, threshold) in enumerate(repacked_values):
            if alias < 0 or alias >= len(values):
                raise ValueError(f"Each alias should be between 0 and {len(values) - 1}, inclusive. "
                                 f"For column {ind}, got {alias}")

            if threshold < 0 or threshold > avg_column_height:
                raise ValueError(f"Each threshold should be between 0 and {avg_column_height}, inclusive. "
                                 f"For column {ind}, got {threshold}")

            reconstructed_original_values[ind] += threshold
            reconstructed_original_values[alias] += avg_column_height - threshold

        if reconstructed_original_values != values:
            raise ValueError(f"The total number of boxes of each color in repacked list should match the original value. "
                             f"In repacked list, got box numbers {reconstructed_original_values}, expected {values}")


#####################################################################################################################################

@mark.parametrize("qbk_class", [ref.AliasStateSwap] if ref_available else [])
def test_aliasstateswap(qbk_class):
    global log_message

    for n, b in product(range(1, 3), range(1, 4)):
        qpu = QPU(filters=BIT_DEFAULT)
        swapper = qbk_class()
        for input_state_reg, input_alias, input_threshold, input_coin_toss in product(
            range(2 ** n), range(2 ** n), range(2 ** b), range(2 ** b)):

            log_message = f"Testing {n=}, {b=}"
            log_message += f"\nError for state_reg={input_state_reg}, alias={input_alias}, threshold={input_threshold}, coin_toss={input_coin_toss}: "

            # Reset QPU and allocate the registers of the required type
            qpu.reset(3 * (n + b))
            state_reg = Qubits(n, "state_reg", qpu)
            alias = Qubits(n, "alias", qpu)
            threshold = Qubits(b, "threshold", qpu)
            coin_toss = Qubits(b, "coin_toss", qpu)

            # Prepare quantum inputs
            state_reg.write(input_state_reg)
            alias.write(input_alias)
            threshold.write(input_threshold)
            coin_toss.write(input_coin_toss)

            # Run the swap
            swapper.compute(state_reg, alias, threshold, coin_toss)

            # Verify the results
            res_state_reg = state_reg.read()
            res_alias = alias.read()
            res_threshold = threshold.read()
            res_coin_toss = coin_toss.read()

            if res_threshold != input_threshold or res_coin_toss != input_coin_toss:
                raise ValueError(f"The state of the register {'threshold' if res_threshold != input_threshold else 'coin_toss'} should not change")
            if input_coin_toss >= input_threshold:
                if res_state_reg != input_alias or res_alias != input_state_reg:
                    raise ValueError(f"Registers `state_reg` and `alias` should be swapped")
            else:
                if res_state_reg != input_state_reg or res_alias != input_alias:
                    raise ValueError(f"The state of the register {'state_reg' if res_state_reg != input_state_reg else 'alias'} should not change")
        

