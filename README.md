# pripy
<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and licence info here --->
![GitHub contributors](https://img.shields.io/github/contributors/jcranney/pripy) ![License](https://img.shields.io/github/license/jcranney/pripy) 
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pripy) ![PyPI version](https://img.shields.io/pypi/v/pripy) 

pripy is a python package aimed at providing phase retrieval algorithms, primarily for use in adaptive optics systems and simulators. These algorithms must have a common and minimal API, in order to accelerate the testing of various algorithms on a given optical system.


## Prerequisites

* A recent version of Python 3 installed on any Windows/Linux/MacOS machine,

## Installing pripy

To install the most recent stable version of pripy, simply:

```bash
pip install pripy
```

To install the latest development version from this git repo, instead do:

```bash
git clone https://github.com/jcranney/pripy
cd pripy
pip install -e .
```

## Using pripy

To use pripy, follow the provided examples, e.g., using Gerchberg-Saxton in the sandbox AO environment (no external simulator required):

```bash
cd examples
ipython -i sandbox_gs.py   # run the sandbox Gerchberg-Saxton example
```

## Contributing to pripy
pripy is in its infancy and welcomes collaborative input. To contribute to pripy, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the branch: `git push`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contributors

* [@jcranney](https://github.com/jcranney) 
* [@mvandam](https://github.com/mvandam)
* [@ndoucet](https://github.com/ndoucet)

## Contact

If you want to contact me you can reach me at jesse.cranney@anu.edu.au.

## License

This project uses the following license: [MIT License](https://github.com/jcranney/pripy/blob/main/LICENSE).