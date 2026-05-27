from functools import partial
from psiqdk.workbench import QPU, Qubits
from pytest import mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_SingleQubitMeasurements as ref
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

###############################################################################################

def check_distinguish_states(
    state_names, # Readable names of the states
    state_amps,  # Amplitudes of each state
    fun          # Callable that is being tested
):
    n_shots = 100
    for expected_result in range(2):
        try:
            counts = {}
            for _ in range(n_shots):
                qpu = QPU(num_qubits=1)
                reg = Qubits(num_qubits=1, name='reg', qpu=qpu)

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


###############################################################################################

def test_zero_or_one(fun=ref.zero_or_one if ref_available else None):
    amps = [[1, 0], 
            [0, 1]]
    check_distinguish_states(["|0⟩", "|1⟩"], amps, fun)


def test_plus_or_minus(fun=ref.plus_or_minus if ref_available else None):
    amps = [[1, 1], 
            [1, -1]]
    check_distinguish_states(["|+⟩", "|-⟩"], amps, fun)


def test_psi_plus_or_psi_minus(fun=ref.psi_plus_or_psi_minus if ref_available else None):
    amps = [[0.6, 0.8], 
            [-0.8, 0.6]]
    check_distinguish_states(["|Ψ+⟩", "|Ψ-⟩"], amps, fun)


def test_a_or_b(fun=ref.a_or_b if ref_available else None):
    from math import cos, sin, pi
    for i in range(11):
        alpha = i * pi / 10
        amps = [[cos(alpha), -1j * sin(alpha)], 
                [-1j * sin(alpha), cos(alpha)]]
        names = [f"|A⟩ = cos({i}π/10)|0⟩ - i sin({i}π/10)|1⟩", 
                 f"|B⟩ = -i sin({i}π/10)|0⟩ + cos({i}π/10)|1⟩"]
        sol = partial(fun, alpha=alpha)
        check_distinguish_states(names, amps, sol)
