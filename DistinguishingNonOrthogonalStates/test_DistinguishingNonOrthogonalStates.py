from cmath import exp, pi
from psiqdk.workbench import QPU, Qubits
from pytest import mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_DistinguishingNonOrthogonalStates as ref
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


def check_non_orthogonal_states(
    n_qubits,    # Number of qubits in the register
    n_states,    # Number of different states
    state_names, # Readable names of the states
    state_amps,  # Amplitudes of each state
    fun,         # Callable that is being tested
):
    n_shots_per_state = 1000
    total_counts = [[0, 0], [0, 0]]
    for expected_result in range(n_states):
        for _ in range(n_shots_per_state):
            qpu = QPU(num_qubits=n_qubits)
            reg = Qubits(n_qubits, 'reg', qpu)

            reg.push_state(state_amps[expected_result])
            actual_result = fun(reg)

            if actual_result not in range(n_states):
                raise ValueError(f"Your solution returned {actual_result}, but only 0 and 1 return values are allowed")

            total_counts[expected_result][actual_result] += 1
        
    total_correct = total_counts[0][0] + total_counts[1][1]

    if total_correct < n_shots_per_state * n_states * 0.8:
        for i in range(n_states):
            print(f"{100 * total_counts[i][i] / n_shots_per_state}% of shots correct for state {state_names[i]}")
        raise ValueError(f"Only {total_correct} out of {n_shots_per_state * n_states} total shots "
                            f"({100 * total_correct / (n_shots_per_state * n_states)}%) are correct")


def test_measure_zero_or_plus(solution=ref.measure_zero_or_plus if ref_available else None):
    amps = [[1, 0],
           [1, 1]]
    check_non_orthogonal_states(1, 2, ["|0⟩", "|+⟩"], amps, solution)


####################################################################################################################################


def check_non_orthogonal_states_inconclusive(
    n_qubits,    # Number of qubits in the register
    n_states,    # Number of different states
    state_names, # Readable names of the states
    state_amps,  # Amplitudes of each state
    fun,         # Callable that is being tested
):
    n_shots_per_state = 1000
    total_counts = {}
    for expected_result in range(n_states):
        counts = {-1: 0, 0: 0, 1: 0}
        for _ in range(n_shots_per_state):
            qpu = QPU(num_qubits=n_qubits)
            reg = Qubits(num_qubits=n_qubits, name='reg', qpu=qpu)

            reg.push_state(state_amps[expected_result])
            actual_result = fun(reg)

            if actual_result not in range(-1, n_states):
                raise ValueError(f"Your solution returned {actual_result}, but only -1, 0 and 1 return values are allowed")                

            counts[actual_result] += 1
        
        total_counts[expected_result] = counts
        
    inconclusive_shots = sum([total_counts[i][-1] for i in range(n_states)])
    incorrect_shots = sum([total_counts[expected_result][1 - expected_result] for expected_result in range(n_states)])
    total_shots = n_shots_per_state * n_states
    
    if incorrect_shots > 0 or \
        total_counts[0][0] < 0.1 * total_shots or \
        total_counts[1][1] < 0.1 * total_shots or \
        inconclusive_shots > 0.8 * total_shots :
        for i in range(n_states):
            print(f"{100 * total_counts[i][i] / n_shots_per_state}% of shots correct, "
                  f"{100 * total_counts[i][-1] / n_shots_per_state}% inconclusive for state {state_names[i]}")
        if incorrect_shots > 0:
            raise ValueError("Your solution should never be incorrect")
        if inconclusive_shots > 0.8 * total_shots:
            raise ValueError(f"{100 * inconclusive_shots / total_shots}% of total shots are inconclusive, which is higher than the 80% threshold")
        raise ValueError("One of the input states is identified correctly less than 10% of the time")


def test_measure_zero_or_plus_or_inconclusive(solution=ref.measure_zero_or_plus_or_inconclusive if ref_available else None):
    amps = [[1, 0],
           [1, 1]]
    check_non_orthogonal_states_inconclusive(1, 2, ["|0⟩", "|+⟩"], amps, solution)


####################################################################################################################################


def check_peres_wooters_game(
    state_names, # Readable names of the states
    state_amps,  # Amplitudes of each state
    fun          # Callable that is being tested
):
    possible_outcomes = [{1, 2}, {0, 2}, {0, 1}]
    n_shots_per_state = 1000
    for i in range(3):
        for _ in range(n_shots_per_state):
            qpu = QPU(num_qubits=2)
            reg = Qubits(1, 'reg', qpu)

            reg.push_state(state_amps[i])
            actual_result = fun(reg)

            if actual_result not in possible_outcomes[i]:
                raise ValueError(f"Unexpected measurement outcome for {state_names[i]}: expected {possible_outcomes[i]}, got {actual_result}")
    

def test_peres_wooters_game(solution=ref.peres_wooters_game if ref_available else None):
    w = exp(1j * 2 * pi / 3)
    amps = [[1, 1],
           [1, w],
           [1, w**2]]
    check_peres_wooters_game(["|A⟩", "|B⟩", "|C⟩"], amps, solution)