from warnings import catch_warnings, filterwarnings
from pytest import mark

try:
    from importnb import Notebook
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_QPE as ref
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


############## Gates for testing ######################################################################################

def x_gate(q, cond=None):
    q.x(cond=cond)

def y_gate(q, cond=None):
    q.y(cond=cond)

def z_gate(q, cond=None):
    q.z(cond=cond)

def s_gate(q, cond=None):
    q.s(cond=cond)

def t_gate(q, cond=None):
    q.t(cond=cond)

def i_gate(q):
    q.identity()

def h_gate(q):
    q.had()


############## Problem 1. ######################################################################################

def test_eigenvalues_s(fun=ref.eigenvalues_s if ref_available else None):
    actual = fun()
    expected = [complex(1.0, 0.0), complex(0.0, 1.0)]
    
    if len(actual) != 2:
        raise Exception(f"The array of eigenvalues should have exactly two elements.")
    
    if actual[0] != expected[0] or actual[1] != expected[1]:
        raise Exception(f"Incorrect value for one of the eigenvalues.")


############## Problem 2. ######################################################################################

def test_eigenvectors_x(fun=ref.eigenvectors_x if ref_available else None):
    actual = fun()
    
    if len(actual) != 2:
        raise Exception(f"The array of eigenvectors should have exactly two elements.")
    
    for i in range(2):
        if len(actual[i]) != 2:
            raise Exception(f"Each eigenvector should have exactly two elements.")
        if abs(actual[i][0]) + abs(actual[i][1]) < 1e-9:
            raise Exception(f"Each eigenvector should be non-zero.")
        
    # One eigenvector has to have equal components, the other one - opposite ones
    equal  = abs(actual[0][0] - actual[0][1]) < 1e-9 and abs(actual[1][0] + actual[1][1]) < 1e-9
    oppose = abs(actual[0][0] + actual[0][1]) < 1e-9 and abs(actual[1][0] - actual[1][1]) < 1e-9
    if not equal and not oppose:
        raise Exception(f"Incorrect value for one of the eigenvectors.")


############## Problem 3. ######################################################################################

def test_is_eigenvector(fun=ref.is_eigenvector if ref_available else None):
    eigenvectors = [
        (z_gate, i_gate, "Z, |0⟩"),
        (z_gate, x_gate, "Z, |1⟩"),
        (s_gate, i_gate, "S, |0⟩"),
        (s_gate, x_gate, "S, |1⟩"),
        (x_gate, h_gate, "X, |+⟩"),
    ]
    for (U, P, msg) in eigenvectors:
        if not fun(U, P):
            raise Exception(f"Incorrect for (U, |ψ⟩) = ({msg}): expected true")

    not_eigenvectors = [
        (z_gate, h_gate, "Z, |+⟩"),
        (x_gate, x_gate, "X, |1⟩"),
        (x_gate, z_gate, "X, |0⟩"),
        (y_gate, h_gate, "Y, |+⟩"),
        (y_gate, x_gate, "Y, |1⟩")
    ]
    for (U, P, msg) in not_eigenvectors:
        if fun(U, P):
            raise Exception(f"Incorrect for (U, |ψ⟩) = ({msg}): expected false")


############## Problem 4. ######################################################################################

def test_one_bit_phase_estimation(fun=ref.one_bit_phase_estimation if ref_available else None):
    eigenvectors = [
        (z_gate, i_gate,  1, "Z, |0⟩"),
        (z_gate, x_gate, -1, "Z, |1⟩"),
        (s_gate, i_gate,  1, "S, |0⟩"),
        (x_gate, h_gate,  1, "X, |+⟩"),
    ]
    for (U, P, expected, msg) in eigenvectors:
        actual = fun(U, P)
        if actual != expected:
            raise Exception(f"Incorrect eigenvalue for (U, |ψ⟩) = ({msg}): expected {expected}, got {actual}")


############## Problem 5. ######################################################################################

def test_phase_estimation(fun=ref.phase_estimation if ref_available else None):
    tests = [
        (z_gate, i_gate, 1, 0, "Z, |0⟩"),
        (z_gate, x_gate, 1, 1, "Z, |1⟩"),
        (x_gate, h_gate, 1, 0, "X, |+⟩"),
        (s_gate, i_gate, 2, 0, "S, |0⟩"),
        (s_gate, x_gate, 2, 1, "S, |1⟩"),
        (z_gate, x_gate, 2, 2, "Z, |1⟩"), # Higher precision than necessary
        (t_gate, i_gate, 3, 0, "T, |0⟩"),
        (t_gate, x_gate, 3, 1, "T, |1⟩"),
        (s_gate, x_gate, 3, 2, "S, |1⟩"), # Higher precision than necessary
        (z_gate, x_gate, 3, 4, "Z, |1⟩"), # Higher precision than necessary
    ]

    for (U, P, n, expected, msg) in tests:
        # Repeat several times to catch probabilistic failures
        for _ in range(10):
            actual = fun(U, P, n)
            if actual != expected:
                raise Exception(f"Incorrect eigenphase for (U, |ψ⟩, n) = ({msg}, {n}): expected {expected}, got {actual}")