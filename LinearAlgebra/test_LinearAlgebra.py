import random as r
from cmath import sqrt
from warnings import catch_warnings

from pytest import approx, mark

try:
    from importnb import Notebook
    # Ignore warnings about invalid syntax when importing LaTeX cells
    with catch_warnings(action="ignore", category=SyntaxWarning):
        with Notebook():
            import Workbook_LinearAlgebra as ref
    ref_available = True
except ImportError:
    pytest.fail("import error")
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

# Generates a random number from -5 to 5
def randnum():
    return (r.random() - 0.5) * r.randint(1, 10)

# Generates a random complex number
def randcomplex():
    return randnum() + randnum() * 1j

# Height (number of rows) is the first dimension for matrices
# Generates a random matrix populated with complex numbers
def gen_complex_matrix(h = -1, w = -1):
    if h == -1: h = r.randint(1, 5)
    if w == -1: w = r.randint(1, 5)
    ans = []
    for i in range(h):
        temp = []
        for j in range(w):
            temp.append(randcomplex())
        ans.append(temp)
    return ans

# ------------------------------------------------------
# Generates a neat message mixing strings and matrices
def gen_matrix_message(matrices, strings):
    # Calculate the length of every string
    strlengths = []
    for s in strings:
        strlengths.append(len(s))
    
    # Find tallest matrix
    hmax = 0
    for mat in matrices:
        h = len(mat)
        if h > hmax: hmax = h
    
    # Calculate the rows where each matrix would be positioned
    starts = []
    ends = []
    lengths = []
    for mat in matrices:
        h = len(mat)
        start = (hmax + 1 - h) // 2
        starts.append(start)
        ends.append(start + h - 1)
        lengths.append((14 if isinstance(mat[0][0], complex) else 7) * len(mat[0]) + 3)
    
    # Start building the string
    middle = hmax // 2
    ans = ""
    for i in range(hmax):
        for j in range(len(matrices)):
            if i == middle:
                ans += strings[j]
            else:
                ans += ' ' * strlengths[j]
            
            if starts[j] <= i <= ends[j]:
                row = i - starts[j]
                ans += '| '
                ans += format_row(matrices[j][row])
                ans += '|'
            else:
                ans += ' ' * lengths[j]
        
        if i == middle:
            ans += strings[-1]
        else:
            ans += ' ' * strlengths[-1]
        
        ans += '\n'
    
    return ans

# Formats the row of a matrix to be evenly spaced
def format_row(row):
    ans = ""
    for num in row:
        if num == ...:
            ans += "     ...     "
            continue
        
        if num.real >= 0:
            ans += ' '
        
        ar = abs(num.real)
        ai = abs(num.imag)
        if ar < 10:
            ans += "{0:.3f}".format(num.real)
        elif ar < 100:
            ans += "{0:.2f}".format(num.real)
        else:
            ans += str(round(num.real))
        
        if num.imag >= 0:
            ans += '+'
        
        if ai < 10:
            ans += "{0:.3f}".format(num.imag)
        elif ai < 100:
            ans += "{0:.2f}".format(num.imag)
        else:
            ans += str(round(num.imag))
        
        ans += 'i '
    return ans

def gen_labeled_message(matrices, labels):
    ans = ""
    for i in range(len(matrices)):
        ans += format_matrix(matrices[i], labels[i]) + '\n'
    return ans

def format_matrix(matrix, label):
    if matrix == ...:
        return label + "..."
    n = len(matrix)
    lsize = len(label)
    ans = ""
    middle = n // 2
    for i in range(n):
        if i == middle:
            ans += label
        else:
            ans += ' ' * lsize
        ans += '| ' + format_row(matrix[i]) + '|\n'
    return ans

# ------------------------------------------------------
# Checks that two matrices are (approximately) equal to each other
def matrix_equal(act, exp):
    if act == ... or exp == ...:
        return False
    
    h = len(act)
    w = len(act[0])
    # Check that sizes match
    if h != len(exp) or w != len(exp[0]):
        return False
    
    for i in range(h):
        # Check that the length of each row matches the expectation
        if w != len(act[i]):
            return False
        for j in range(w):
            if act[i][j] == ... or act[i][j] != approx(exp[i][j]):
                return False
    return True

# ------------------------------------------------------
# Makes a copy of the target matrix
def matrix_copy(mat):
    ans = []
    for row in mat:
        ans.append(row[:])
    return ans

# Creates an n by m matrix filled with 0s
def create_empty_matrix(n, m):
    ans = []
    for i in range(n):
        ans.append([0] * m)
    return ans

# ------------------------------------------------------
def matrix_add_ref(a, b):
    n = len(a)
    m = len(a[0])
    ans = create_empty_matrix(n, m)
    for i in range(n):
        for j in range(m):
            ans[i][j] = a[i][j] + b[i][j]
    
    return ans

@mark.parametrize("fun", [ref.matrix_add] if ref_available else [])
def test_matrix_add(fun):
    global log_message
    for i in range(10):
        a = gen_complex_matrix()
        b = gen_complex_matrix(len(a), len(a[0]))
        expected = matrix_add_ref(a, b)
        # log_message = f"Testing\n{format_matrix(a, "a")}\n{format_matrix(b,"b")}"
        log_message = f"Testing\n{a=}\n{b=}"
        actual = fun(a, b)
        if actual == None:
            raise ValueError("Your function must return a value!")
        if not matrix_equal(actual, expected):
            raise ValueError("Unexpected results of addition: \n"
                  + gen_labeled_message([a, b, expected, actual],
                                        ["A: ", "B: ", "Expected: ", "You returned: "])
                  + "Try again!")

# ------------------------------------------------------
def scalar_mult_ref(x, a):
    ans = []
    for row in a:
        temp = []
        for elem in row:
            temp.append(elem * x)
        ans.append(temp)
    return ans

@mark.parametrize("fun", [ref.scalar_mult] if ref_available else [])
def test_scalar_mult(fun):
    global log_message
    for i in range(10):
        a = gen_complex_matrix()
        x = randcomplex()
        expected = scalar_mult_ref(x, a)
        log_message = f"Testing {x=} {a=}"
        actual = fun(x, a)
        if actual == None:
            raise ValueError("Your function must return a value!")
        if not matrix_equal(actual, expected):
            raise ValueError(f"Unexpected results of scalar multiplication: \nScalar: {x:.3f}\n\n"
                  + gen_labeled_message([a, expected, actual],
                                        ["A: ", "Expected: ", "You returned: "])
                  + "Try again!")

# ------------------------------------------------------
def matrix_mult_ref(a, b):
    h = len(a)
    common = len(a[0]) # = len(b)
    w = len(b[0])
    ans = create_empty_matrix(h, w)
    for i in range(h):
        for j in range(w):
            for k in range(common):
                ans[i][j] += a[i][k] * b[k][j]
    return ans

@mark.parametrize("fun", [ref.matrix_mult] if ref_available else [])
def test_matrix_mult(fun):
    global log_message
    for i in range(10):
        a = gen_complex_matrix()
        b = gen_complex_matrix(len(a[0]))
        expected = matrix_mult_ref(a, b)
        log_message = f"Testing {a=} {b=}"
        actual = fun(a, b)
        if actual == None:
            raise ValueError("Your function must return a value!")
        if not matrix_equal(actual, expected):
            raise ValueError("Unexpected results of matrix multiplication: \n"
                  + gen_labeled_message([a, b, expected, actual],
                                        ["A: ", "B: ", "Expected: ", "You returned: "])
                  + "Try again!")

# ------------------------------------------------------
def matrix_inverse_ref(m):
    a = m[0][0]
    b = m[0][1]
    c = m[1][0]
    d = m[1][1]
    det = (a * d) - (b * c)
    return [[d / det, -b / det], [-c / det, a / det]]

@mark.parametrize("fun", [ref.matrix_inverse] if ref_available else [])
def test_matrix_inverse(fun):
    global log_message
    for i in range(10):
        a = None
        det = 0
        while det == 0:
            a = gen_complex_matrix(2,2)
            det = (a[0][0] * a[1][1]) - (a[0][1] * a[1][0])
        expected = matrix_inverse_ref(a)
        log_message = f"Testing {a=}"
        actual = fun(a)
        if actual == None:
            raise ValueError("Your function must return a value!")
        if not matrix_equal(actual, expected):
            raise ValueError("Inverse doesn't seem to match expected:\n"
                  + gen_labeled_message([a, expected, actual],
                                        ["A: ", "Expected: ", "You returned: "])
                  + "Try again!")

# ------------------------------------------------------
def transpose_ref(a):
    ans = []
    n = len(a)
    m = len(a[0])
    for i in range(m):
        row = []
        for j in range(n):
            row.append(a[j][i])
        ans.append(row)
    return ans

@mark.parametrize("fun", [ref.transpose] if ref_available else [])
def test_transpose(fun):
    global log_message
    for i in range(10):
        a = gen_complex_matrix()
        expected = transpose_ref(a)
        log_message = f"Testing {a=}"
        actual = fun(a)
        if actual == None:
            raise ValueError("Your function must return a value!")
        if not matrix_equal(actual, expected):
            raise ValueError("Unexpected result of a transpose:\n"
                  + gen_labeled_message([a, expected, actual],
                                        ["A: ", "Expected: ", "You returned: "])
                  + "Try again!")

# ------------------------------------------------------
def conjugate_ref(a):
    ans = []
    for row in a:
        temp = []
        for num in row:
            temp.append(num.real - (num.imag * 1j))
        ans.append(temp)
    return ans

@mark.parametrize("fun", [ref.conjugate] if ref_available else [])
def test_conjugate(fun):
    global log_message
    for i in range(10):
        a = gen_complex_matrix()
        expected = conjugate_ref(a)
        log_message = f"Testing {a=}"
        actual = fun(a)
        if actual == None:
            raise ValueError("Your function must return a value!")
        if not matrix_equal(actual, expected):
            raise ValueError("Unexpected result of matrix conjugate:\n"
                  + gen_labeled_message([a, expected, actual],
                                        ["A: ","Expected: ","You returned: "])
                  + "Try again!")

# ------------------------------------------------------
def adjoint_ref(a):
    return conjugate_ref(transpose_ref(a))

@mark.parametrize("fun", [ref.adjoint] if ref_available else [])
def test_adjoint(fun):
    global log_message
    for i in range(10):
        a = gen_complex_matrix()
        expected = adjoint_ref(a)
        log_message = f"Testing {a=}"
        actual = fun(a)
        if actual == None:
            raise ValueError("Your function must return a value!")
        if not matrix_equal(actual, expected):
            raise ValueError("Unexpected result of adjoint operation:\n"
                  + gen_labeled_message([a, expected, actual],
                                        ["A: ","Expected: ","You returned: "])
                  + "Try again!")

# ------------------------------------------------------
# Projection operator - returns (<v,w>/<v,v>)*v
def projection(v,w):
    return scalar_mult_ref(inner_prod_ref(w,v) / inner_prod_ref(v,v), v)

# Appends a vector as the next column in a matrix
def append_vector(m,v):
    for i in range(len(m)):
        m[i].append(v[i][0])

# Generates a unitary matrix via the Gram-Schmidt process
def gen_unitary_matrix(n = -1):
    if n == -1: n = r.randint(1, 5)
    temp = gen_complex_matrix(n, n)
    
    # Split the generated matrix into vectors
    vectors = []
    for i in range(n):
        v = []
        for j in range(n):
            v.append([temp[j][i]])
        vectors.append(v)
    
    vectors2 = []
    for i in range(n):
        v = matrix_copy(vectors[i])
        for j in range(i-1,-1,-1):
            v = matrix_add_ref(v, scalar_mult_ref(-1, projection(vectors2[j], vectors[i])))
        vectors2.append(v)
    
    ans = []
    for i in range(n): ans.append([])
    for v in vectors2:
        append_vector(ans, normalize_ref(v))
    
    return ans

edge_unitary_matrices = [[[0, 0], [0, 0]], [[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), 1/sqrt(2)]]]

def is_matrix_unitary_ref(a):
    n = len(a)
    prod = matrix_mult_ref(a, adjoint_ref(a))
    for i in range(n):
        for j in range(n):
            if i == j:
                if prod[i][j] != approx(1): return False
            else:
                if prod[i][j] != approx(0): return False
    return True

@mark.parametrize("fun", [ref.is_matrix_unitary] if ref_available else [])
def test_is_matrix_unitary(fun):
    global log_message
    for testId in range(12):
        a = []
        # The first two tests are edge cases, after that unitary and non-unitary matrices alternate
        if testId < 2:
            a = edge_unitary_matrices[testId]
        elif testId % 2 == 0:
            a = gen_unitary_matrix()
        else:
            n = r.randint(1,5)
            a = gen_complex_matrix(n,n)
        expected = is_matrix_unitary_ref(a)
        log_message = f"Testing {a=}"
        actual = fun(a)
        if actual == None:
            raise ValueError("Your function must return a value!")
        if actual != expected:
            raise ValueError("Unexpected result:\n"
                  + gen_matrix_message([a],
                                       ["Matrix ", (" is " if expected else " is not ")
                                                   + "unitary, but misidentified as " 
                                                   + ("unitary" if actual else "not unitary")]))

# ------------------------------------------------------
def inner_prod_ref(v, w):
    return matrix_mult_ref(adjoint_ref(v), w)[0][0]

@mark.parametrize("fun", [ref.inner_prod] if ref_available else [])
def test_inner_prod(fun):
    global log_message
    for i in range(10):
        v = gen_complex_matrix(w = 1)
        w = gen_complex_matrix(len(v), 1)
        expected = inner_prod_ref(v, w)
        log_message = f"Testing {v=} {w=}"
        actual = fun(v, w)
        if type(actual) == list:
            raise TypeError("You should return a number, not a matrix")
        if actual == None or actual == ...:
            raise ValueError("Your function must return a value!")
        if actual != approx(expected):
            raise ValueError("Unexpected result of inner product:\n"
                  + gen_labeled_message([v, w], ["V: ", "W: "])
                  + f"Expected: {expected:.3f}\n\n"
                  + f"You returned: {actual:.3f}\n\nTry again!")

# ------------------------------------------------------
def normalize_ref(v):
    return scalar_mult_ref(1 / sqrt(inner_prod_ref(v,v).real), v)

@mark.parametrize("fun", [ref.normalize] if ref_available else [])
def test_normalize(fun):
    global log_message
    for i in range(10):
        v = None
        norm = 0
        while norm == 0:
            v = gen_complex_matrix(w = 1)
            norm = inner_prod_ref(v, v)
        expected = normalize_ref(v)
        log_message = f"Testing {v=}"
        actual = fun(v)
        if actual == None:
            raise ValueError("Your function must return a value!")
        if not matrix_equal(actual, expected):
            raise ValueError("Unexpected result of normalization:\n"
                  + gen_labeled_message([v, expected, actual], ["V: ", "Expected: ", "You returned: "])
                  + "Try again!")

# ------------------------------------------------------
def outer_prod_ref(v, w):
    return matrix_mult_ref(v, adjoint_ref(w))

@mark.parametrize("fun", [ref.outer_prod] if ref_available else [])
def test_outer_prod(fun):
    global log_message
    for i in range(10):
        v = gen_complex_matrix(w = 1)
        w = gen_complex_matrix(w = 1)
        expected = outer_prod_ref(v, w)
        log_message = f"Testing {v=} {w=}"
        actual = fun(v, w)
        if actual == None:
            raise ValueError("Your function must return a value!")
        if not matrix_equal(actual, expected):
            raise ValueError("Unexpected result of outer product:\n"
                  + gen_labeled_message([v, w, expected, actual],
                                        ["V: ", "W: ", "Expected: ", "You returned: "])
                  + "Try again!")

# ------------------------------------------------------
def tensor_product_ref(a, b):
    n = len(a)
    m = len(a[0])
    k = len(b)
    l = len(b[0])
    
    ans = []
    for i in range(n * k):
        row = [0] * m * l
        ans.append(row)
    
    for i in range(n):
        for j in range(m):
            for o in range(k):
                for p in range(l):
                    ans[i * k + o][j * l + p] = a[i][j] * b[o][p]
    
    return ans

@mark.parametrize("fun", [ref.tensor_product] if ref_available else [])
def test_tensor_product(fun):
    global log_message
    for i in range(10):
        (a, b) = ([[1, 1, 1]], [[1, 1, 1]])
        while len(a[0]) * len(b[0]) >= 8:
            a = gen_complex_matrix()
            b = gen_complex_matrix()
        expected = tensor_product_ref(a, b)
        log_message = f"Testing {a=} {b=}"
        actual = fun(a, b)
        if actual == None:
            raise ValueError("Your function must return a value!")
        if not matrix_equal(actual, expected):
            raise ValueError("Unexpected result of tensor product:\n"
                  + gen_labeled_message([a, b, expected, actual],
                                        ["A: ", "B: ", "Expected: ", "You returned: "])
                  + "Try again!")
