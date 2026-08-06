# Introduction

The Quantum Katas are a collection of self-paced tutorials and programming problems to help you learn fault-tolerant quantum computing and PsiQDK programming.

## Prerequisites

To use the Quantum Katas, you need to install [PsiQuantum Quantum Development Kit](https://construct.psiquantum.com/docs/), the Python toolkit for developing fault-tolerant quantum algorithms. You can find the complete installation guide and requirements in [PsiQDK documentation](https://construct.psiquantum.com/docs/psiqdk/installation.html).

You'll also need to install `pytest` and `scipy` for several of the advanced katas.

```bash
pip install psiqdk pytest scipy
```

## User workflow

Each kata is a Jupyter Notebook that includes the theory on one topic, interspersed with demos and programming problems designed to help you practice and internalize this topic.

You can run demos to see the results of their execution and modify them to check your understanding of how the outputs will change. Demos don't affect the verification of programming problems solutions, though you can use functions defined in demos as part of your solution if you want to.

Each programming problem offers the problem description, followed by a Python code cell which provides the signature of the function you need to implement. Your goal is to fill the blank (marked with `...`) with Python code that solves the problem.

> You can define helper functions outside the given function if you need to. You shouldn't modify the signature of the given function in a way that would make it impossible to call it with the original signature, for example, by adding required arguments.

Once you have written the code, you can verify it by executing the code cell. The `@problem` decorator will call the test that covers this problem and show you the results.

If you can't come up with a solution on your own, you can look up the explained solutions in the workbook file in the same folder - a Jupyter Notebook that includes theoretical explanations of the solutions and any details of code implementations that might not be obvious for a first-time user, as well as Python code cells with reference solutions to each problem.

## Learning path

### Single-qubit systems

* [The qubit](./Qubit/Qubit.ipynb). Learn what a qubit is and how to represent its quantum state.
* [Single-qubit gates](./SingleQubitGates/SingleQubitGates.ipynb). Learn what a quantum gate is and about the most common single-qubit gates.
* [Measurements in single-qubit systems](./SingleQubitMeasurements/SingleQubitMeasurements.ipynb). Learn what quantum measurement is and how to use it in single-qubit systems.

### Multi-qubit systems

* [Multi-qubit systems](./MultiQubitSystems/MultiQubitSystems.ipynb). Learn to represent quantum states of multi-qubit systems.
* [Multi-qubit gates](./MultiQubitGates/MultiQubitGates.ipynb). Learn about the most common multi-qubit gates, including controlled variants of single-qubit gates.
* [Multi-qubit measurements](./MultiQubitMeasurements/MultiQubitMeasurements.ipynb). Learn to use measurements for multi-qubit systems.
* [Preparing simple quantum states](./PreparingSimpleStates/PreparingSimpleStates.ipynb). Learn to prepare simple superposition states.
* [Distinguishing orthogonal quantum states](./DistinguishingStates/DistinguishingStates.ipynb). Learn to distinguish orthogonal quantum states using measurements.

### Quantum oracles and simple oracle algorithms

* [Quantum oracles](./Oracles/Oracles.ipynb). Learn to implement classical functions as equivalent quantum oracles.
* [Marking oracles](./MarkingOracles/MarkingOracles.ipynb). Practice implementing marking oracles for a variety of classical functions.
* [Deutsch algorithm](./DeutschAlgorithm/DeutschAlgorithm.ipynb). Learn Deutsch algorithm - the simplest algorithm relying on quantum oracles.
* [Deutsch–Jozsa algorithm](./DeutschJozsaAlgorithm/DeutschJozsaAlgorithm.ipynb). Learn about Bernstein–Vazirani and Deutsch–Jozsa algorithms.

### Grover's search algorithm

* [Grover's search algorithm](./GroversSearch/GroversSearch.ipynb). Learn about Grover's search algorithm and the practical considerations of applying it to solving problems.
* [Solving SAT problems using Grover's algorithm](./SolvingSATWithGrover/SolvingSATWithGrover.ipynb). Explore Grover's search algorithm, using SAT problems as an example. Learn to implement quantum oracles based on the problem description instead of a hard-coded answer.
* [Solving graph coloring problems using Grover's algorithm](./GraphColoringGrover/GraphColoringGrover.ipynb). Continue the exploration of Grover's search algorithm, using graph coloring problems as an example.

### Quantum arithmetic

* [Quantum arithmetic data types](./ArithmeticDataTypes/ArithmeticDataTypes.ipynb). Learn to represent signed and unsigned integers as quantum data types and to implement naive arithmetic operations on them.
* [Quantum adders](./Adders/Adders.ipynb). Learn to implement different quantum adders.

### Building up to Shor's algorithm

* [Quantum Fourier transform](./QFT/QFT.ipynb). Learn to implement quantum Fourier transform and to use it to perform simple state transformations.
* [Quantum phase estimation](./QPE/QPE.ipynb). Learn about quantum phase estimation problem and algorithms to solve it.
* [Shor's integer factoring algorithm](./ShorsFactoring/ShorsFactoring.ipynb). Learn about Shor's algorithm for factoring integers.

### Quantum chemistry

* [Finding the ground state energy of a molecule](./GroundStateEnergy/GroundStateEnergyTheory.ipynb). Learn about the problem of finding ground state energy and the workflow of converting it into a task that can be solved using quantum computing.

### Miscellaneous

* [Preparing arbitrary quantum states](./ArbitraryStatePreparation/ArbitraryStatePreparation.ipynb). Learn to prepare arbitrary quantum states (states without simple internal structure).
* [Distinguishing non-orthogonal quantum states](./DistinguishingNonOrthogonalStates/DistinguishingNonOrthogonalStates.ipynb). Learn about different approaches to the problem of distinguishing non-orthogonal states.
* [Quantum Approximate Optimization Algorithm](./MaxCutQAOA/MaxCutQAOA.ipynb). Learn to solve MaxCut problem using quantum approximate optimization algorithm (QAOA).

## Developer workflow

To add a new problem to the kata, for example, a problem called "Oracle 123" in the Marking Oracles kata, you need to:

1. Add the problem description and function template to the main notebook (`MarkingOracles.ipynb`).
   The function name should match the problem title and follow the naming patterns and signature style of previous problems in the kata (`oracle_123`).
2. Add the solution description and code cell with the "reference solution" to the workbook (`Workbook_MarkingOracles.ipynb`).
   The reference solution should have the same name as the function in the main notebook (`oracle_123`).
   Notice that the workbooks don't use the `@problem` decorator; to run the test on the reference solution, you need to copy it to the main notebook or use pytest.
3. Add the test to the Python file with the testing harness (`test_MarkingOracles.py`).
   The test should have the same name as the function in the main notebook with a prefix `test_` (`test_oracle_123`).
   It should take the solution function as the argument with the reference solution as the default value (see examples of existing tests for the syntax). The actual solution will be provided to the test via the `@problem` decorator; the default value is used for testing all reference solutions via pytest.
4. Verify that the empty function template fails the test and the reference solution passes it.
   You can do that by copying the reference solution from the workbook to the main notebook by hand and executing the code cell.
   Alternatively, you can test all reference solutions at once using pytest. To do this, you need to install the package `importnb` (`pip install importnb`) and run pytest in the kata folder. If this package is not installed, pytest will skip testing, since this package is used to import the reference solutions from the workbook.

   > Note that if you're using importnb to test all reference solutions, you need to install pytest 8.0.2 or earlier; later pytest versions are not compatible with the importnb plugin.

## Getting help and contributing

At this point, the Quantum Katas project is in open development, but there is no easy way for us to accept major external contributions. We are working towards bringing it into full open source mode.

In the meantime, we welcome bug reports and discussions! Please [file an issue](https://github.com/PsiQ/workbench-quantum-katas/issues/new/) to report a bug, or [start a discussion](https://github.com/PsiQ/workbench-quantum-katas/discussions/new/choose) to ask a question or discuss a topic.
