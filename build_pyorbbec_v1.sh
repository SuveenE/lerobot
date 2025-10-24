git clone https://github.com/orbbec/pyorbbecsdk.git
cd pyorbbecsdk
pip3 install -r requirements.txt
mkdir build
cd build
cmake -Dpybind11_DIR=`pybind11-config --cmakedir` ..
make -j4
make install