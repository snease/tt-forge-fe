

git submodule update --init --recursive -f
source env/activate

cmake -B env/build env
cmake --build env/build

cmake -G Ninja -B build .
cmake --build build
