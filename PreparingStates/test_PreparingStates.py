from math import sqrt, cos, sin, log2
from cmath import exp, pi
from psiqdk.workbench import QPU, Qubits
from pytest import approx, mark
from warnings import catch_warnings
from functools import partial

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_PreparingStates as ref
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


def check_state_vector(
    fun: callable,                  # Callable that is being tested
    n_qubits: int,                  # Number of qubits in the main register
    expected_vector: list[complex], # State vector it should prepare
    n_aux: int=0                    # Number of auxiliary qubit(s) the solution is allowed to use
) -> None:
    # Construct the qpu and register
    total_num_qubits = n_qubits + n_aux   # Will be unnecessary once auto-resize is in
    qpu = QPU(num_qubits=total_num_qubits)
    qpu.enable_qubit_allocation_debugging()
    reg = Qubits(n_qubits, 'reg', qpu)
    fun(reg)

    # Extract the saved state vector
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
        print("Expected state vector:")
        print(expected_vector)
        print("Actual state vector:")
        print(actual_vector)
        raise ValueError("State vectors should be equal")


#####################################################################################################################################

def test_prepare_all_two_qubits(fun=ref.prepare_all_two_qubits if ref_available else None):
    expected_vector = [0.5] * 4
    check_state_vector(fun, 2, expected_vector)

    
@mark.parametrize("fun", [ref.prepare_all_two_qubits_phase_flip_1, ref.prepare_all_two_qubits_phase_flip_2] if ref_available else [])
def test_prepare_all_two_qubits_phase_flip(fun):
    expected_vector = [0.5] * 3 + [-0.5]
    check_state_vector(fun, 2, expected_vector)


def test_prepare_all_two_qubits_complex_phases(fun=ref.prepare_all_two_qubits_complex_phases if ref_available else None):
    expected_vector = [0.5, -0.5, 0.5j, -0.5j]
    check_state_vector(fun, 2, expected_vector)


isq2 = 1 / sqrt(2)

def test_prepare_bell_state(fun=ref.prepare_bell_state if ref_available else None):
    expected_vector = [isq2, 0, 0, isq2]
    check_state_vector(fun, 2, expected_vector)


@mark.parametrize("fun", [ref.prepare_any_bell_state_1, ref.prepare_any_bell_state_2] if ref_available else [])
def test_prepare_any_bell_state(fun):
    amps = [[isq2, 0, 0, isq2],
            [isq2, 0, 0, -isq2],
            [0, isq2, isq2, 0],
            [0, -isq2, isq2, 0]
           ]

    for index in range(4):
        expected_vector = amps[index]
        fun_index = partial(fun, index=index)
        check_state_vector(fun_index, 2, expected_vector)


@mark.parametrize("fun", [ref.prepare_ghz_state_1, ref.prepare_ghz_state_2] if ref_available else [])
def test_prepare_ghz_state(fun):
    for n in range(1, 6):
        expected_vector = [isq2] + [0] * (2 ** n - 2) + [isq2]
        check_state_vector(fun, n, expected_vector)


def test_prepare_all_n_qubits(fun=ref.prepare_all_n_qubits if ref_available else None):
    for n in range(1, 6):
        expected_vector = [isq2 ** n] * (2 ** n)
        check_state_vector(fun, n, expected_vector)


def test_prepare_even_odd_numbers(fun=ref.prepare_even_odd_numbers if ref_available else None):
    for n in range(1, 6):
        for even in [True, False]:
            print(f"Testing {n=}, {even=}")
            expected_vector = ([isq2 ** (n - 1), 0] if even else [0, isq2 ** (n - 1),]) * (2 ** (n - 1))
            fun_even = partial(fun, even=even)
            check_state_vector(fun_even, n, expected_vector)


def bitstring_as_int(bits):
    return sum([2 ** i if bits[i] else 0 for i in range(len(bits))])


def test_prepare_zero_and_bitstring(fun=ref.prepare_zero_and_bitstring if ref_available else None):
    tests = [
        [True],
        [True, False],
        [True, True],
        [True, False, False],
        [True, False, True],
        [True, True, False],
        [True, True, True]
    ]
    for bits in tests:
        n = len(bits)
        print(f"Testing {n=}, {bits=}")
        expected_vector = [0] * (2 ** n)
        ind = bitstring_as_int(bits)
        expected_vector[0] = expected_vector[ind] = isq2
        fun_bits = partial(fun, bits=bits)
        check_state_vector(fun_bits, n, expected_vector)


def test_prepare_two_bitstrings(fun=ref.prepare_two_bitstrings if ref_available else None):
    tests1 = [
        [False],
        [True],
        [False, False],
        [True, False],
        [False, False],
        [False, False, False],
        [False, False, False]
    ]
    tests2 = [
        [True],
        [False],
        [False, True],
        [False, True],
        [True, True],
        [False, False, True],
        [True, False, True]
    ]

    for bits1, bits2 in zip(tests1, tests2):
        n = len(bits1)
        print(f"Testing {n=}, {bits1=}, {bits2=}")
        expected_vector = [0] * (2 ** n)
        ind1 = bitstring_as_int(bits1)
        ind2 = bitstring_as_int(bits2)
        expected_vector[ind1] = expected_vector[ind2] = isq2
        fun_bits = partial(fun, bits1=bits1, bits2=bits2)
        check_state_vector(fun_bits, n, expected_vector, 1)


def test_prepare_four_bitstrings(fun=ref.prepare_four_bitstrings if ref_available else None):
    tests = [
        [[False, False], [False, True], [True, False], [True, True]],
        [[False, True, False], [True, False, False], [False, False, True], [True, True, False]],
        [[True, False, False], [False, False, True], [False, True, False], [True, True, True]],
        [[False, False, False], [False, True, False], [True, True, False], [True, False, True]]
    ]

    for bits in tests:
        n = len(bits[0])
        print(f"Testing {n=}, {bits=}")
        expected_vector = [0] * (2 ** n)
        for bits_ind in range(4):
            ind = bitstring_as_int(bits[bits_ind])
            expected_vector[ind] = 0.5
        fun_bits = partial(fun, bits=bits)
        check_state_vector(fun_bits, n, expected_vector, 2)


@mark.parametrize("fun", [ref.prepare_given_parity_1, ref.prepare_given_parity_2, ref.prepare_given_parity_3] if ref_available else [])
def test_prepare_given_parity(fun):
    for n in range(2, 7):
        for parity in [0, 1]:
            print(f'Testing {n=}, {parity=}')
            amp = 1 / sqrt(2) ** (n - 1)
            expected_vector = [amp if i.bit_count() % 2 == parity else 0 for i in range(2 ** n)]
            fun_parity = partial(fun, parity=parity)
            check_state_vector(fun_parity, n, expected_vector, 1)


def test_prepare_uneven_single_qubit(fun=ref.prepare_uneven_single_qubit if ref_available else None):
    for alpha_ind in range(10):
        alpha = pi*alpha_ind/10
        print(f'{alpha=:.4f}')
        expected_vector = [cos(alpha), sin(alpha)]
        fun_with_alpha = partial(fun, alpha=alpha)
        check_state_vector(fun_with_alpha, 1, expected_vector)


def test_prepare_uneven_two_qubit(fun=ref.prepare_uneven_two_qubit if ref_available else None):
    expected_vector = [isq2, 0.5, 0, 0.5]
    check_state_vector(fun, 2, expected_vector)


def test_prepare_even_two_qubit(fun=ref.prepare_even_two_qubit if ref_available else None):
    expected_vector = [1 / sqrt(3)] * 3 + [0]
    check_state_vector(fun, 2, expected_vector)


def test_prepare_even_two_qubit_phases(fun=ref.prepare_even_two_qubit_phases if ref_available else None):
    omega = exp(2j * pi / 3)
    expected_vector = [1 / sqrt(3), omega ** 2 / sqrt(3), omega / sqrt(3), 0]
    check_state_vector(fun, 2, expected_vector)


def test_prepare_hardy_state(fun=ref.prepare_hardy_state if ref_available else None):
    expected_vector = [3 / sqrt(12)] + [1 / sqrt(12)] * 3
    check_state_vector(fun, 2, expected_vector)


@mark.parametrize("fun", [ref.prepare_wstate_power_of_two_1, ref.prepare_wstate_power_of_two_2] if ref_available else [])
def test_prepare_wstate_power_of_two(fun):
    for k in range(4):
        n = 2 ** k
        print(f'Testing {n=}')
        expected_vector = [0] * 2 ** n
        for ind in range(n):
            expected_vector[2 ** ind] = 1 / sqrt(n)
        check_state_vector(fun, n, expected_vector, k)


@mark.parametrize("fun", [ref.prepare_wstate_1, ref.prepare_wstate_2, ref.prepare_wstate_3] if ref_available else [])
def test_prepare_wstate(fun):
    for n in range(1, 10):
        print(f'Testing {n=}')

        expected_vector = [0] * 2 ** n
        for ind in range(n):
            expected_vector[2 ** ind] = 1 / sqrt(n)

        p = 1
        while p < n:
            p *= 2
        check_state_vector(fun, n, expected_vector, int(log2(p)) + int(p - n))
