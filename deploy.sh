#m2r README.md
rm -rf rgl.egg-info
rm -rf dist
python setup.py sdist
twine upload dist/* --verbose