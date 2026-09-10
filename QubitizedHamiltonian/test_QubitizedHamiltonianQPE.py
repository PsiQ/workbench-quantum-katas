from itertools import product
from IPython.display import HTML
from math import sqrt
import numpy as np
from random import seed, uniform
from typing import Callable
from psiqdk.workbench import QPU, Qubits, Qubrick
from psiqdk.workbench.filter_presets import BIT_DEFAULT
from psiqdk.workbench.qre import resource_estimator
from pytest import approx, mark
from warnings import catch_warnings
from functools import partial

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

import matplotlib.pyplot as plt

def plot_landscape(landscape, xrange, yrange, parameter, xlabel, ylabel):
    fig, ax = plt.subplots()
    im = ax.imshow(landscape, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xrange)))
    ax.set_xticklabels(xrange)
    ax.set_yticks(range(len(yrange)))
    ax.set_yticklabels(yrange)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{parameter.capitalize()} landscape")
    fig.colorbar(im, ax=ax, label=parameter)

    plt.show()
