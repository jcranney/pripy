git clone https://github.com/jcranney/pripy &&
cd pripy &&
python -m build &&
python -m twine upload --repository pypi dist/pripy-*