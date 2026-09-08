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

