#!/bin/bash

# exit on first error
set -e 

# manylinux2014_x86_64
yum update -y
yum install -y centos-release-scl
# yum list available rh-python*

# yum install -y rh-python38
# scl enable rh-python38 bash
# python3.8 --version
# python3.8 -m pip install --upgrade pip
# python3.8 -m pip install setuptools wheel auditwheel
# python3.8 -m pip install numpy
# python3.8 /github/workspace/setup.py bdist_wheel

yum install -y rh-python35
scl enable rh-python35 bash
source scl_source enable rh-python35
python3.5 --version
python3.5 -m pip install --upgrade pip
python3.5 -m pip install setuptools wheel auditwheel
python3.5 -m pip install numpy
python3.5 /github/workspace/setup.py bdist_wheel

# yum list python*
# yum install -y python3 python3-devel python3-pip
# python3 -m pip install --upgrade pip
# python3 -m pip install setuptools wheel auditwheel
# python3 -m pip install numpy
# python3 /github/workspace/setup.py bdist_wheel

# yum install -y python3.8 python3.8-devel python3.8-pip
# python3.8 -m pip install --upgrade pip
# python3.8 -m pip install setuptools wheel auditwheel
# python3.8 -m pip install numpy
# python3.8 /github/workspace/setup.py bdist_wheel

# manylinux1_x86_64
# yum install -y python3 
# yum install -y python3-pip
# python -m pip install --upgrade pip 
# python -m pip install setuptools wheel auditwheel
# python /github/workspace/setup.py bdist_wheel

# ls -al /opt/python/
# PYBIN="/opt/python/cp39-cp39/bin"
# "${PYBIN}/python" -m pip install --upgrade pip
# "${PYBIN}/pip" install setuptools wheel auditwheel numpy
# "${PYBIN}/python" /github/workspace/setup.py bdist_wheel

# upload
ls -al /github/workspace/dist/
latest_wheel=$(ls -t /github/workspace/dist/*linux*.whl | head -1)
echo $latest_wheel
auditwheel repair $latest_wheel -w /github/workspace/dist/
