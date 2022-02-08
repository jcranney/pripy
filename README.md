# PRiPy
Phase Retrieval in Python.

This package intends to implement common phase retrieval algorithms for Adaptive Optics (AO), for use in Python-based AO simulators such as COMPASS and CEO. Ideally, these algorithms should have a common and minimal API, in order to accelerate the testing of various algorithms on a given optical system.

## TODO

 - Sandbox examples (numpy AND cupy compatible),
 - CEO examples.
 - COMPASS examples,

Early implementation goals:
 - **FF**: Fast and Furious Wavefront Sensing,
 - **LIFT**: LInearised Focal-plane Technique,
 - **GS**: classical Gerchberg Saxton algorithm for phase estimation,
 - **TAME**: Taylor-based Moving horizon Estimation,

Medium-term goals:
 - COMPASS API,
 - CEO API.
 - PyPI for `pip install`

## Installation
```bash
git clone git@github.com/jcranney/pripy.git
cd pripy
pip install .
```
