from itertools import product
from math import sqrt
from random import randint, seed, uniform
from scipy.stats import chisquare
from typing import Callable, Sequence
from psiqdk.workbench import QPU, Qubits, Qubrick
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

def generate_classical_tests():
    '''Generate a list of tests in format (n, b, values) that serve as input to repacking.'''
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

    return tests


@mark.parametrize("fun", [ref.repack] if ref_available else [])
def test_repack(fun):
    global log_message

    tests = generate_classical_tests()

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

def repack_reference(n: int, b: int, values: list[int]) -> list[tuple[int, int]]:
    '''The reference solution to problem 2 from the workbook.
    
    We cannot import it from the workbook directly in case the learner doesn't have importnb installed.'''
    avg_column_height = 1 << b
    repacked_values = [(ind, 0) for ind in range(len(values))]

    for _ in range(len(values)):
        ind_below_avg = next((ind for ind in range(len(values)) if values[ind] < avg_column_height), None)
        ind_above_avg = next((ind for ind in range(len(values)) if values[ind] > avg_column_height), None)
        if ind_below_avg is None or ind_above_avg is None:
            break

        repacked_values[ind_below_avg] = (ind_above_avg, values[ind_below_avg])
        fixup = avg_column_height - values[ind_below_avg]
        values[ind_below_avg] += fixup
        values[ind_above_avg] -= fixup
        
    return repacked_values


@mark.parametrize("fun", [ref.sample] if ref_available else [])
def test_sample(fun):
    global log_message

    # Generate the same tests we used in problem 2
    tests = generate_classical_tests()

    n_samples = 1024
    alpha = 0.05

    for (n, b, values) in tests:
        # Convert the test case to repacked values that are input for this problem
        repacked_values = repack_reference(n, b, values.copy())

        log_message = f"Testing {n=}, {b=}, {values=}, {repacked_values=}"
        
        # Run each test 3 times, reject only if fails all 3
        passed = False
        for run_ind in range(3):
            expected_counts = [n_samples * val / (1 << (n + b)) for val in values]

            actual_counts = [0] * (1 << n)
            for _ in range(n_samples):
                sample = fun(b, repacked_values)
                # Check for kinds of errors that re-sampling is not going to fix
                if not isinstance(sample, int) or sample < 0 or sample >= (1 << n):
                    raise ValueError(f"Each returned value should be an integer between 0 and {(1 << n) - 1}, inclusive. Got {sample}")
                if expected_counts[sample] == 0:
                    raise ValueError(f"Value {sample} has expected probability of 0 and should not be present in samples.")
                actual_counts[sample] += 1

            # Filter out entries where expected_counts = 0 (chisquare doesn't support that)
            actual_counts_filtered = [actual_counts[ind] for ind in range(1 << n) if expected_counts[ind] > 0]
            expected_counts_filtered = [expected_counts[ind] for ind in range(1 << n) if expected_counts[ind] > 0]
            if len(expected_counts_filtered) == 1:
                # There is only one value left, and we know that all individual returns were valid - nothing to check
                passed = True
                break

            log_message += f"\nRun {run_ind}:\n" + \
                f"{actual_counts=}, {expected_counts=}\n"

            _, p_value = chisquare(f_obs=actual_counts_filtered, f_exp=expected_counts_filtered)
            if p_value >= alpha:
                passed = True
                break
            else:
                log_message += f"The sampling algorithm does not follow the required distribution; p_value={float(p_value)}"
        if not passed:
            raise ValueError("The sampling algorithm failed three test runs")


#####################################################################################################################################

def check_data_loader(
    qbk_class: type[Qubrick],          # Data loader qubrick
    data: list[int]                    # Data that is being loaded
) -> None:
    n_bits_index = len(data).bit_length()
    n_bits_target = max(data).bit_length()
    qpu = QPU(filters=BIT_DEFAULT)
    data_loader = qbk_class()
    # Run the test on every input basis state
    for j in range(2 ** n_bits_index):
        # Reset QPU and allocate the registers
        qpu.reset(n_bits_index + n_bits_target + 1)
        index = Qubits(n_bits_index, "index", qpu)
        target = Qubits(n_bits_target, "target", qpu)

        # Prepare quantum input for index (target is 0)
        index.write(j)

        # Load the data
        data_loader.compute(index, target, data)

        # Check the results
        res_index = index.read()
        res_target = target.read()

        prefix = f"Error for j={j}: "
        if res_index != j:
            raise Exception(f"{prefix}data loading shouldn't change the state of the index register")
        if j < len(data) and res_target != data[j]:
            raise Exception(f"{prefix}the state of the target register is {res_target}, expected {data[j]}")
        if j >= len(data) and res_target != 0:
            raise Exception(f"{prefix}the state of the target register is {res_target}, expected 0 for indices outside of data")


@mark.parametrize("qbk_class", [ref.DataLoader] if ref_available else [])
def test_dataloader(qbk_class):
    global log_message
    for data in [
        [1, 2, 3],
        [4, 0, 2, 1],
        [0, 1, 2, 3],
        [3],
        [1, 0]
    ]:
        log_message = f"Testing loading {data=}"
        check_data_loader(qbk_class, data)


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
        

