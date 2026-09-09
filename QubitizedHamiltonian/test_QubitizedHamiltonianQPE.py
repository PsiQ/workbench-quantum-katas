from itertools import product
from IPython.display import HTML
import numpy as np
from psiqdk.algorithms import PrepareNaive, SelectNaive, NaiveAdd, QPE, QFT
from psiqdk.workbench import QPU, Qubits, Qubrick
from psiqdk.workbench.filter_presets import BIT_DEFAULT
from psiqdk.workbench.qre import resource_estimator
from pytest import mark
from warnings import catch_warnings

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_QubitizedHamiltonianQPE as ref
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

#####################################################################################################################################

def print_colored_matrix(matrix: list[list[float]], w: list[float]) -> None:
    """Print the matrix with elements color-coded based on which element of w they are."""
    colors = ["#AB3377", "#66CCEE", "#228833"]
    html = ["<table>"]
    for row in matrix:
        html.append("<tr>")
        for val in row:
            color = colors[w.index(val)] if val in w else "lightgray"
            html.append(f"<td style='padding: 5px 10px; font-weight: bold; color: {color};'>{val}</td>")
        html.append("</tr>")
    html.append("</table>")
    display(HTML("".join(html)))


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
    for j, c in product(range(2 ** n_bits_index), [0, 1]):        
        # Reset QPU and allocate the registers
        qpu.reset(n_bits_index + n_bits_target + 1)
        index = Qubits(n_bits_index, "index", qpu)
        target = Qubits(n_bits_target, "target", qpu)
        ctrl = Qubits(1, "ctrl", qpu)

        # Prepare quantum input for index and ctrl (target is 0)
        index.write(j)
        ctrl.write(c)

        # Load the data
        data_loader.compute(index, target, data, ctrl)

        # Check the results
        res_index = index.read()
        res_target = target.read()
        res_ctrl = ctrl.read()

        prefix = f"Error for j={j}, ctrl={c}: "
        if res_index != j or res_ctrl != c:
            raise Exception(f"{prefix}data loading shouldn't change the state of the register {'index' if res_index != j else 'ctrl'}")
        if c == 0 and res_target != 0:
            raise Exception(f"{prefix}controlled data loading shouldn't change the state of the target register if control qubit is |0⟩")
        if c == 1:
            if j < len(data) and res_target != data[j]:
                raise Exception(f"{prefix}the state of the target register is {res_target}, expected {data[j]}")
            if j >= len(data) and res_target != 0:
                raise Exception(f"{prefix}the state of the target register is {res_target}, expected 0 for indices outside of data")


#####################################################################################################################################

@mark.parametrize("qbk_class", [ref.DataLoader] if ref_available else [])
def test_dataloader(qbk_class):
    for data in [
        [1, 2, 3],
        [4, 0, 2, 1],
        [0, 1, 2, 3],
        [3],
        [1, 0]
    ]:
        global log_message
        log_message = f"Testing loading {data=}"
        check_data_loader(qbk_class, data)


#####################################################################################################################################

def check_adder(
    n_bits_a: int,                     # Number of bits in a
    n_bits_b: int,                     # Number of bits in b
    qbk_class: type[Qubrick],          # Data loader qubrick
) -> None:
    # For naive operations, we don't need to allocate any auxiliary qubits
    n_qubits = n_bits_a + n_bits_b + 1

    qpu = QPU(filters=BIT_DEFAULT)
    adder = qbk_class()
    for input_a, input_b, input_c in product(range(2 ** n_bits_a), range(2 ** n_bits_b), [0, 1]):
        # Reset QPU and allocate the registers of the required type
        qpu.reset(n_qubits)
        a = Qubits(n_bits_a, "a", qpu)
        b = Qubits(n_bits_b, "b", qpu)
        ctrl = Qubits(1, "ctrl", qpu)

        # Prepare quantum inputs
        a.write(input_a)
        b.write(input_b)
        ctrl.write(input_c)

        # Run the adder
        adder.compute(a, b, ctrl)

        # Compare the results of classical and quantum computations
        res_a = a.read()
        res_b = b.read()
        res_c = ctrl.read()

        prefix = f"Error for a={input_a}, b={input_b}, ctrl={input_c}: "
        if res_b != input_b or res_c != input_c:
            raise Exception(f"{prefix}in-place addition shouldn't change the state of the register {'b' if res_b != input_b else 'ctrl'}")
        if input_c == 0:
            if res_a != input_a:
                raise Exception(f"{prefix}controlled addition shouldn't change the state of the register a if control qubit is |0⟩")
        else:
            # Evaluate classical function on the classical input
            res_expected = (input_a + input_b) % (2 ** n_bits_a)
            if res_a != res_expected:
                raise Exception(f"{prefix}expected result a={res_expected}, got {res_a}")


####################################################################################################

@mark.parametrize("qbk_class", [ref.Adder] if ref_available else [])
def test_adder(qbk_class):
    for n in range(1, 5):
        for m in range(1, n):
            global log_message
            log_message = f"Testing {n=}, {m=}"
            check_adder(n, m, qbk_class)


####################################################################################################

class GeneralizedHamiltonianEncoding(Qubrick):
    """Qubitization walk operator (block encoding + reflection) for a hopping
    Hamiltonian specified by aligned (coefficient, offset) term pairs.

    The offsets and the addends-register width are passed in, so the
    same class works for any number of terms / any chain size.
    """

    def __init__(self, prepare, qrom, adder, offsets, addends_size, **kwargs):
        self.prepare = prepare
        self.qrom = qrom
        self.adder = adder
        self.offsets = offsets
        self.addends_size = addends_size
        super().__init__(**kwargs)

    def _compute(self, state_reg, coefficient_reg, add_subtract_reg, ctrl=0):
        # PREPARE
        self.prepare.compute(coefficient_reg, cond=ctrl)
        add_subtract_reg.had(cond=ctrl)

        addends = self.alloc_temp_qreg(self.addends_size, "addends")
        # QROM
        with self.qrom.computed(coefficient_reg, addends, self.offsets, ctrl=ctrl):
            add_subtract_reg.x(cond=ctrl)     # Makes SELECT self-inverse
            # Apply adder on the |0⟩ branch and subtractor on the |1⟩ branch
            self.adder.compute(state_reg, addends, ctrl=(add_subtract_reg == 0) | ctrl)
            self.adder.compute(state_reg, addends, ctrl=(add_subtract_reg == 1) | ctrl, dagger=True)
        addends.release()

        # UnPREPARE
        add_subtract_reg.had(cond=ctrl)
        self.prepare.uncompute()

        # reflection for qubitization
        (~coefficient_reg | ~add_subtract_reg).reflect(ctrl=ctrl)
        if ctrl:
            ctrl.reflect()


def build_qpu(
    state_size: int,
    coefficients: list[float],
    offsets: list[int],
    bits_of_precision: int=5,
    prepare: Qubrick | None=None,
    qrom: Qubrick | None=None,
    adder: Qubrick | None=None,
    buffer: int=10,
):
    """Build a QPU holding a full QPE with the qubitized Hamiltonian.

    Args:
        state_size: The number of qubits in the system register (the chain has 2**state_size sites).
        coefficients: Term weights $w_j$; the default PREPARE loads square roots of these.
        offsets: Integer offsets loaded by the QROM and added to / subtracted from the
            state register. One offset per coefficient.
        bits_of_precision: The size of the QPE phase register.
        prepare: The state preparation Qubrick. Defaults to PrepareNaive.
        qrom: The data lookup Qubrick. Defaults to SelectNaive.
        adder: The adder Qubrick. Defaults to NaiveAdd.
        buffer: The number of spare qubits for decompositions that rely on auxiliary qubits.
    """
    assert len(coefficients) == len(offsets), "The numbers of coefficients and offsets should be the same"
    assert all(int(d) == d and d >= 0 for d in offsets), "Each element of offsets must be a non-negative integer"
    assert max(offsets) < 2 ** state_size, "Each offset should fit in the state register"

    if prepare is None:
        prepare = PrepareNaive(coefficients)
    if qrom is None:
        qrom = SelectNaive()
    if adder is None:
        adder = NaiveAdd()

    coefficient_size = (len(coefficients) - 1).bit_length()   # Number of QROM address bits
    addends_size = (max(offsets) - 1).bit_length() + 1        # Number of bits to hold the largest offset
    ctrl_size = 1

    num_qubits = state_size + coefficient_size + ctrl_size + bits_of_precision + addends_size + buffer

    qpu = QPU(num_qubits=num_qubits, filters=[">>buffer>>"])
    state_reg = Qubits(state_size, "state", qpu)
    coefficients_reg = Qubits(coefficient_size, "coefficients_reg", qpu)
    add_subtract_reg = Qubits(1, "add_subtract_reg", qpu)
    phases_reg = Qubits(bits_of_precision, "phases_reg", qpu)

    qft = QFT()
    qft.compute(state_reg)

    encoding = GeneralizedHamiltonianEncoding(prepare, qrom, adder, offsets, addends_size)

    qpe = QPE(unitary=encoding, bits_of_precision=bits_of_precision)
    qpe.compute(
        state_reg,
        phases_reg,
        coefficient_reg=coefficients_reg,
        add_subtract_reg=add_subtract_reg,
    )
    return qpu


####################################################################################################

@mark.parametrize("get_adder", [ref.get_adder_optimize_av] if ref_available else [])
def test_get_adder_optimize_av(get_adder):
    AV_THRESHOLD = 230_000
    qpu = build_qpu(
        state_size=8, coefficients=[0.1] * 8, offsets=list(range(1, 9)),
        bits_of_precision=5, buffer=20, adder=get_adder()
    )    
    av = float(resource_estimator(qpu).resources()["active_volume"])

    assert av < AV_THRESHOLD, f"Active volume {av:,.1f} is above {AV_THRESHOLD:,}"
    print(f"Active volume {av:,.1f} is below {AV_THRESHOLD:,}!")


@mark.parametrize("get_adder", [ref.get_adder_optimize_av_highwater] if ref_available else [])
def test_get_adder_optimize_av_highwater(get_adder):
    AV_THRESHOLD_2 = 250_000
    HW_THRESHOLD_2 = 26
    qpu = build_qpu(
        state_size=8, coefficients=[0.1] * 8, offsets=list(range(1, 9)),
        bits_of_precision=5, buffer=20, adder=get_adder()
    )    
    resources = resource_estimator(qpu).resources()
    av, highwater = float(resources["active_volume"]), int(resources["qubit_highwater"])

    assert av < AV_THRESHOLD_2, f"Active volume {av:,.1f} is above {AV_THRESHOLD_2:,}"
    assert highwater <= HW_THRESHOLD_2, f"Qubit highwater {highwater} exceeds {HW_THRESHOLD_2}"
    print(f"Active volume {av:,.1f} is below {AV_THRESHOLD_2:,}!")
    print(f"Qubit highwater {highwater} is less or equal to {HW_THRESHOLD_2}!")


####################################################################################################

@mark.parametrize("qbk_class", [ref.MockedRoutine] if ref_available else [])
def test_mockedroutine(qbk_class):
    for N in [8, 64, 1024]:
        global log_message
        log_message = f"Testing N={N}"

        qpu = QPU(num_qubits=64, filters=[">>buffer>>"])
        qbk_class(N=N, qc=qpu).compute()  # no register arguments -> need to pass qc= explicitly
        resources = resource_estimator(qpu).resources()

        expected_av = N ** 2
        if resources["active_volume"] != expected_av:
            raise Exception(
                f"Active volume should equal N**2: expected {expected_av}, "
                f"got {resources['active_volume']}"
            )

        expected_highwater = int(np.ceil(np.log2(N)))
        if resources["qubit_highwater"] != expected_highwater:
            raise Exception(
                f"Qubit highwater should equal ceil(log2 N): expected {expected_highwater}, "
                f"got {resources['qubit_highwater']}"
            )
