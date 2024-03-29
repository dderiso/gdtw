#!/bin/bash

# exit on first error
set -e 

curl https://pyenv.run | bash
exec $SHELL
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

echo "pyenv install 3.7.10"
pyenv install 3.7.10
pyenv global 3.7.10

python --version
python3.7 --version
python3.7 -m pip install --upgrade pip
python3.7 -m pip install setuptools wheel auditwheel
python3.7 -m pip install numpy
python3.7 /github/workspace/setup.py bdist_wheel



# manylinux2014_x86_64
# yum update -y
# yum install -y centos-release-scl
# yum list available rh-python*

# yum install -y rh-python38
# scl enable rh-python38 bash
# python3.8 --version
# python3.8 -m pip install --upgrade pip
# python3.8 -m pip install setuptools wheel auditwheel
# python3.8 -m pip install numpy
# python3.8 /github/workspace/setup.py bdist_wheel

# yum install -y rh-python35
# scl enable rh-python35 bash
# # source scl_source enable rh-python35
# ls /opt/rh/rh-python35

# export PATH=/opt/rh/rh-python35/root/usr/bin:$PATH
# ls /opt/rh/rh-python35/root/usr/bin/

# python3.5 --version
# python3.5 -m pip install --upgrade pip
# python3.5 -m pip install setuptools wheel auditwheel
# python3.5 -m pip install numpy
# python3.5 /github/workspace/setup.py bdist_wheel

# yum install -y https://rpms.remirepo.net/enterprise/remi-release-7.rpm
# yum-config-manager --enable remi
# yum install -y python39
# python3.9 --version
# python3.9 -m pip install --upgrade pip
# python3.9 -m pip install setuptools wheel auditwheel
# python3.9 -m pip install numpy
# python3.9 /github/workspace/setup.py bdist_wheel

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
