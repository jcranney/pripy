## Deployment Instructions
The idea, probably not the best way to go about it, is to build the distribution files from a clean docker container. In principle, any unit tests can also be done from this container, similar to running things from a clean virtual environment.

```bash
docker build -t . deploypripy
docker run -it deploypripy
```

This will ask for a `PyPI` username and password, which any authorised user can provide. If the program exits successfully, the new version of the pripy should be deployed to PyPI, ready to be installed on any machine via `pip`.