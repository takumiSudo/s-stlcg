stlcg-plus-plus (stlcg++)
======

A toolbox to compute the robustness of STL formulas using computation graphs. This is the PyTorch version of the `STLCG++`, an updated version of the original [`STLCG` toolbox originally implemented in PyTorch](https://github.com/StanfordASL/stlcg/tree/dev).
A JAX version of `STLCG++` can be found at [`STLJAX`](https://github.com/UW-CTRL/stljax)


## Installation

Requires Python 3.10+

Install the repo:

```
pip install git+https://github.com/UW-CTRL/stlcg-plus-plus.git
```

## Description
Please take a look at [README at `STLJAX`](https://github.com/UW-CTRL/stljax) for more details about the latest changes and updates to this toolbox.
We aim to have the Jax and PyTorch libraries to essentially mirror each other in terms of functionality and usage, aside from specific Jax/PyTorch syntaxes and conventions.

In the future, we will have documentation in a single location. But for the meantime, please refer to the [README at `STLJAX`](https://github.com/UW-CTRL/stljax).


## Usage
`demo.ipynb` is an IPython jupyter notebook that showcases the basic functionality of the toolbox:
* Setting up signals for the formulas, including the use of Expressions and Predicates
* Defining STL formulas and visualizing them
* Evaluating STL robustness, and robustness trace
* Gradient descent on STL parameters and signal parameters.



## Feedback
If there are any issues with the code, please make file an issue, or make a pull request.

