sudo -H pip uninstall --yes gdtw

rm -rf build
python setup.py clean build install --force

python test/test.py