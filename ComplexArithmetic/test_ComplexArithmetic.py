import math as m
import random as r
from warnings import catch_warnings

from pytest import approx, mark

try:
    from importnb import Notebook

    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_ComplexArithmetic as ref
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


# Generates a random complex number in Cartesian form
def prep_random_cartesian():
    real = (r.random() - 0.5) * r.randint(0, 100)
    imag = (r.random() - 0.5) * r.randint(0, 100)
    return (real, imag)


# Generates a random complex number in polar form
def prep_random_polar():
    rad = r.random() * r.randint(0, 100)
    theta = (r.random() - 0.5) * m.pi
    return (rad, theta)


# ------------------------------------------------------
# Assert that checks if the result is a tuple of length 2
def assert_tuple(result):
    if result == None:
        return ValueError("Your function must return a value!")
    if not type(result) is tuple:
        return TypeError(
            "Your function must return a tuple, returned " + type(result).__name__ + "."
        )
    if result[0] == ... or result[1] == ...:
        return RuntimeError(
            "You should not return an ellipsis (...) as part of your answer."
        )
    if len(result) != 2:
        return ValueError(
            "Your function must return a tuple of length 2, but returned tuple is of length "
            + str(len(result))
        )


# Assert that verifies the output is a valid complex tuple, and checks that it matches expected output
def assert_cartesian(expected, actual, message):
    if actual != approx(expected):
        return message


# Assert that verifies the output is a valid polar tuple, and checks that it matches expected output
def assert_polar(expected, actual, message):
    (ar, atheta) = actual
    if ar == 0:
        if ar != approx(expected[0]):
            return message
        if not (-m.pi < atheta <= m.pi):
            return "Even for 0 + 0i, the phase must be between -pi and pi."
        return
    if actual != approx(expected):
        return message


# ------------------------------------------------------
# Formats a complex number in Cartesian form neatly
def format_cartesian(x):
    return (
        "{0:.3f}".format(x[0])
        + (" + " if x[1] >= 0 else " - ")
        + "{0:.3f}i".format(abs(x[1]))
    )


# Formats a complex number in polar form neatly
def format_polar(x):
    return "{0:.3f} * e^({1:.3f}i)".format(x[0], x[1])


# ------------------------------------------------------


def imaginary_power_ref(n):
    return 1 if n % 4 == 0 else -1


@mark.parametrize("fun", [ref.imaginary_power] if ref_available else [])
def test_imaginary_power(fun):
    global log_message
    for i in range(-25, 25):
        n = 2 * i
        log_message = f"Testing i^{n}"
        expected = imaginary_power_ref(n)
        actual = fun(n)
        if actual == None:
            raise ValueError(f"Your function must return a value!")
        if expected != actual:
            raise ValueError(
                f"Result of exponentiation doesn't seem to match expected value: expected i^{n} = {expected}, got {actual}"
            )


# ------------------------------------------------------
def complex_add_ref(x, y):
    return (x[0] + y[0], x[1] + y[1])


@mark.parametrize("fun", [ref.complex_add] if ref_available else [])
def test_complex_add(fun):
    global log_message
    for i in range(25):
        x = prep_random_cartesian()
        y = prep_random_cartesian()
        expected = complex_add_ref(x, y)
        log_message = f"Testing {x=} {y=}"
        actual = fun(x, y)
        error = assert_tuple(actual)
        if error is not None:
            raise error
        error = assert_cartesian(
            expected,
            actual,
            ValueError(
                "Sum doesn't seem to match expected value: expected ("
                + format_cartesian(x)
                + ") + ("
                + format_cartesian(y)
                + ") = "
                + format_cartesian(expected)
                + ", got "
                + format_cartesian(actual)
            ),
        )
        if error is not None:
            raise error

# ------------------------------------------------------

# ------------------------------------------------------
def complex_mult_ref(x, y):
    return ((x[0] * y[0]) - (x[1] * y[1]), (x[0] * y[1]) + (x[1] * y[0]))

@mark.parametrize("fun", [ref.complex_mult] if ref_available else [])
def test_complex_mult(fun):
    global log_message
    for i in range(25):
        x = prep_random_cartesian()
        y = prep_random_cartesian()
        expected = complex_mult_ref(x, y)
        log_message = f"Testing {x=} {y=}"
        actual = fun(x, y)
        error = assert_tuple(actual)
        if error is not None:
            raise error
        error = assert_cartesian(expected, actual,
                               ValueError("Product doesn't seem to match expected value:\nexpected ("
                               + format_cartesian(x)
                               + ") * ("
                               + format_cartesian(y)
                               + ") = "
                               + format_cartesian(expected)
                               + ", got "
                               + format_cartesian(actual)))
        if error is not None:
            raise error

# ------------------------------------------------------
def conjugate_ref(x):
    return (x[0], -x[1])

@mark.parametrize("fun", [ref.conjugate] if ref_available else [])
def test_conjugate(fun):
    global log_message
    for i in range(25):
        x = prep_random_cartesian()
        log_message = f"Testing {x=}"
        expected = conjugate_ref(x)
        actual = fun(x)
        error = assert_tuple(actual)
        if error is not None:
            raise error
        error = assert_cartesian(expected, actual,
                               ValueError("Conjugate doesn't seem to match expected value:\nexpected conjugate of "
                               + format_cartesian(x)
                               + " to be "
                               + format_cartesian(expected)
                               + ", got "
                               + format_cartesian(actual)))
        if error is not None:
            raise error

# ------------------------------------------------------
def complex_div_ref(x, y):
    ybar = conjugate_ref(y)
    numer = complex_mult_ref(x, ybar)
    denom = complex_mult_ref(y, ybar)[0]
    return complex_mult_ref(numer, (1 / denom, 0))

@mark.parametrize("fun", [ref.div_test] if ref_available else [])
def test_complex_div(fun):
    global log_message
    for i in range(25):
        x = prep_random_cartesian()
        y = (0, 0)
        while y == (0, 0):
            y = prep_random_cartesian()
        log_message = f"Testing {x=} {y=}"
        expected = complex_div_ref(x, y)
        actual = fun(x, y)
        error = assert_tuple(actual)
        if error is not None:
            raise error
        error = assert_cartesian(expected, actual,
                               ValueError("Quotient doesn't seem to match expected value:\nexpected ("
                               + format_cartesian(x)
                               + ") / ("
                               + format_cartesian(y)
                               + ") = "
                               + format_cartesian(expected)
                               + ", got "
                               + format_cartesian(actual)))
        if error is not None:
            raise error

# ------------------------------------------------------
def modulus_ref(x):
    return m.sqrt(complex_mult_ref(x, conjugate_ref(x))[0])

@mark.parametrize("fun", [ref.modulus] if ref_available else [])
def test_modulus(fun):
    global log_message
    for i in range(25):
        x = prep_random_cartesian()
        log_message = f"Testing {x=}"
        expected = modulus_ref(x)
        actual = fun(x)
        if actual is None:
            raise ValueError("Your function must return a value!")
        if not (type(actual) is float or type(actual) is int):
            raise TypeError("Your function must return a number, returned " + type(actual).__name__ + ".")
        if actual != approx(expected):
            raise ValueError("Modulus doesn't seem to match expected value: expected |"
                  + format_cartesian(x)
                  + f"| = {expected:.3f}, got {actual:.3f}")

# ------------------------------------------------------
def complex_exp_ref(x):
    realpow = m.e ** x[0]
    return (realpow * m.cos(x[1]), realpow * m.sin(x[1]))

@mark.parametrize("fun", [ref.complex_exp] if ref_available else [])
def test_complex_exp(fun):
    global log_message
    for i in range(25):
        x = prep_random_cartesian()
        log_message = f"Testing {x=}"
        expected = complex_exp_ref(x)
        actual = fun(x)
        error = assert_tuple(actual)
        if error is not None:
            raise error
        error = assert_cartesian(expected, actual,
                               ValueError("Result of exponentiation doesn't seem to match expected value:\nexpected e^("
                               + format_cartesian(x)
                               + ") = "
                               + format_cartesian(expected)
                               + ", got "
                               + format_cartesian(actual)))
        if error is not None:
            raise error

# ------------------------------------------------------
def complex_exp_real_ref(r, x):
    if r == 0: return (0, 0)
    lnr = m.log(r)
    return complex_exp_ref(complex_mult_ref((lnr, 0), x))

@mark.parametrize("fun", [ref.complex_exp_real] if ref_available else [])
def test_complex_exp_real(fun):
    global log_message
    for i in range(25):
        base = r.random() * r.randint(1, 100)
        if i == 0:
            base = 0
        x = prep_random_cartesian()
        log_message = f"Testing r={base} {x=}"
        expected = complex_exp_real_ref(base, x)
        actual = fun(base, x)
        error = assert_tuple(actual)
        if error is not None:
            raise error
        error = assert_cartesian(expected, actual,
                               ValueError("Result of exponentiation doesn't seem to match expected value:\n"
                               + f"expected {base:.3f}^("
                               + format_cartesian(x)
                               + ") = "
                               + format_cartesian(expected)
                               + ", got "
                               + format_cartesian(actual)))
        if error is not None:
            raise error

# ------------------------------------------------------
def polar_convert_ref(x):
    return (modulus_ref(x), m.atan2(x[1], x[0]))

@mark.parametrize("fun", [ref.polar_convert] if ref_available else [])
def test_polar_convert(fun):
    global log_message
    for i in range(25):
        x = prep_random_cartesian()
        if i == 0:
            x = (0, 0)
        log_message = f"Testing {x=}"
        expected = polar_convert_ref(x)
        actual = fun(x)
        error = assert_tuple(actual)
        if error is not None:
            raise error
        error = assert_polar(expected, actual,
                           ValueError("Polar conversion doesn't seem to match expected value:\nexpected "
                           + format_cartesian(x)
                           + " to be converted to "
                           + format_polar(expected)
                           + ", got "
                           + format_polar(actual)))
        if error is not None:
            raise error

# ------------------------------------------------------
def cartesian_convert_ref(x):
    return (x[0] * m.cos(x[1]), x[0] * m.sin(x[1]))

@mark.parametrize("fun", [ref.cartesian_convert] if ref_available else [])
def test_cartesian_convert(fun):
    global log_message
    for i in range(25):
        x = prep_random_polar()
        log_message = f"Testing {x=}"
        expected = cartesian_convert_ref(x)
        actual = fun(x)
        error = assert_tuple(actual)
        if error is not None:
            raise error
        error = assert_cartesian(expected, actual,
                               ValueError("Cartesian conversion doesn't seem to match expected value:\nexpected "
                               + format_polar(x)
                               + " to be converted to "
                               + format_cartesian(expected)
                               + ", got "
                               + format_cartesian(actual)))
        if error is not None:
            raise error

# ------------------------------------------------------
def polar_mult_ref(x, y):
    angle = x[1] + y[1]
    if angle > m.pi: angle -= 2 * m.pi
    elif angle <= -m.pi: angle += 2 * m.pi
    return (x[0] * y[0], angle)

@mark.parametrize("fun", [ref.polar_mult] if ref_available else [])
def test_polar_mult(fun):
    global log_message
    for i in range(25):
        x = prep_random_polar()
        y = prep_random_polar()
        if i == 0:
            x = (3, 2)
            y = x
        elif i == 1:
            x = (3, -2)
            y = x
        log_message = f"Testing {x=} {y=}"
        expected = polar_mult_ref(x, y)
        actual = fun(x, y)
        error = assert_tuple(actual)
        if error is not None:
            raise error
        error = assert_polar(expected, actual,
                           ValueError("Product doesn't seem to match expected value:\nexpected ("
                           + format_polar(x)
                           + ") * ("
                           + format_polar(y)
                           + ") = "
                           + format_polar(expected)
                           + ", got "
                           + format_polar(actual)))
        if error is not None:
            raise error

# ------------------------------------------------------
def complex_exp_arbitrary_ref(x, y):
    xp = polar_convert_ref(x)
    return complex_mult_ref(complex_exp_real_ref(xp[0], y), complex_exp_ref(complex_mult_ref((0, xp[1]), y)))

@mark.parametrize("fun", [ref.complex_exp_arbitrary] if ref_available else [])
def test_complex_exp_arbitrary(fun):
    global log_message
    for i in range(25):
        x = prep_random_cartesian()
        y = prep_random_cartesian()
        if i == 0:
            x = (0, 0)
        log_message = f"Testing {x=} {y=}"
        expected = complex_exp_arbitrary_ref(x, y)
        actual = fun(x, y)
        error = assert_tuple(actual)
        if error is not None:
            raise error
        error = assert_cartesian(expected, actual,
                               ValueError("Result of exponentiation doesn't seem to match expected value:\nexpected ("
                               + format_cartesian(x)
                               + ")^("
                               + format_cartesian(y)
                               + ") = "
                               + format_cartesian(expected)
                               + ", got "
                               + format_cartesian(actual)))
        if error is not None:
            raise error
