from psiqdk.workbench import QPU, Qubits, units
from psiqdk.workbench.utils.numpy_utils import fidelity
from pytest import mark
from warnings import catch_warnings
from math import sin, cos, acos, pi, sqrt
from cmath import exp, phase
from functools import partial

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_MultiQubitMeasurements as ref
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


def check_distinguish_states(
    n_qubits,    # Number of qubits in the register
    n_states,    # Number of different states
    state_names, # Readable names of the states
    state_amps,  # Amplitudes of each state
    fun          # Callable that is being tested
):
    n_shots = 100
    for expected_result in range(n_states):
        try:
            counts = {}
            for _ in range(n_shots):
                qpu = QPU(num_qubits=n_qubits)
                reg = Qubits(num_qubits=n_qubits, name='reg', qpu=qpu)

                reg.push_state(state_amps[expected_result])
                actual_result = fun(reg)

                if actual_result in counts:
                    counts[actual_result] += 1
                else:
                    counts[actual_result] = 1
            if len(counts) > 1:
                raise ValueError(f"Non-deterministic measurement outcome: {counts}, should only get {expected_result}.")
            elif len(counts) == 1 and expected_result not in counts:
                raise ValueError(f"Unexpected measurement outcome: expected {expected_result}, got {list(counts.keys())[0]}")

        except Exception as e:
            raise Exception(f"Testing on state {state_names[expected_result]}: {e}")


def check_partial_state_vector(
        fun: callable,                                    # Callable that is being tested
        num_qubits: int,                                  # Number of qubits
        expected_vector: list[complex],                   # State vector it should prepare
        initial_state_prep: callable = None,              # A routine that prepares the initial state (before the solution is called)
        slice_to_extract: slice = slice(None, None, None) # Slice that specify the qubit indices to extract statevector from
        ) -> None:
    # Construct the qpu and register
    qpu = QPU(num_qubits = num_qubits)
    reg = Qubits(num_qubits = 2, name = 'reg', qpu = qpu)

    if initial_state_prep is not None:
        initial_state_prep(reg)

    fun(reg)

    # Convert the warning into an error if the qubit is entangled - the task demands that it's not entangled
    with catch_warnings(action="error", category=UserWarning):
        # Extract the saved state vector on the second qubit
        actual_vector = reg[slice_to_extract].pull_state()
    fid = fidelity(actual_vector, expected_vector)

    assert 1 - fid < 1e-5, f'Expected state vector: {expected_vector}\nActual state vector: {actual_vector}.'

########################################################################################################################

def test_measure_basis_state(solution=ref.measure_basis_state if ref_available else None):
    amps = [[1, 0, 0, 0], 
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
    check_distinguish_states(2, 4, ["|00⟩", "|10⟩", "|01⟩", "|11⟩"], amps, solution)


def test_measure_plusminus_state(solution=ref.measure_plusminus_state if ref_available else None):
    amps = [[1, 1, -1, -1], 
            [1, -1, -1, 1]]
    check_distinguish_states(2, 2, ["|+-⟩", "|--⟩"], amps, solution)


def prepare_alpha_beta_state(reg: Qubits, alpha: complex, beta: complex) -> None:
    '''Generate input state in problem 3'''
    theta = 2 * acos(alpha)
    phi = phase(beta)
    reg[0].had()
    reg[1].ry(theta * units.rad)
    reg[1].phase(phi * units.rad)
    reg[1].x(cond = reg[0])


def test_state_selection_partial_meas(solution = ref.state_selection_partial_meas if ref_available else None):
    for angle_ind in range(5):
        for phase_ind in range(5):
            theta = pi * angle_ind / 5
            phi = pi * phase_ind / 5
            alpha, beta = cos(theta / 2), exp(1j * phi) * sin(theta / 2)
            prepare_current_state = partial(prepare_alpha_beta_state, alpha=alpha, beta=beta)
            print(f'Testing {alpha=:.3f}, {beta=:.3f}')
            for input_ind in range(2):
                solution_input_ind = partial(solution, ind=input_ind)
                expected_vector = [beta, alpha] if input_ind else [alpha, beta]
                # Repeat the solution multiple times, since it's probabilistic
                for _ in range(10):
                    check_partial_state_vector(solution_input_ind, 2, expected_vector, prepare_current_state, slice(1, 2))
        

def test_state_preparation_partial_meas(solution = ref.state_preparation_partial_meas if ref_available else None):
    one_sqrt3 = 1/sqrt(3)
    expected_vector = [one_sqrt3, one_sqrt3, one_sqrt3, 0]
    # Repeat the solution multiple times, since it's probabilistic
    for _ in range(20):
        check_partial_state_vector(solution, 3, expected_vector, slice_to_extract=slice(0, 2))
