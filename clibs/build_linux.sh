# gcc -shared -o libretrieval.so -fPIC retrieve.c
# mv libretrieval.so ../rgl/graph_retrieval/

# g++ retrieve.cpp -o retrieve

# # rely on python version
# c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) retrieve.cpp -o libretrieval$(python3-config --extension-suffix)
#!/bin/bash

versions=(3.7 3.8 3.9 3.10 3.11 3.12 3.13)
# for py_ver in "${versions[@]}"; do
#     conda create -n build_py${py_ver} python=${py_ver} -y
# done
# for py_ver in "${versions[@]}"; do
#     pypath=~/miniconda3/envs/build_py${py_ver}/bin/python3
#     pyconfig=~/miniconda3/envs/build_py${py_ver}/bin/python3-config
#     $pypath -m pip install pybind11
#     c++ -O3 -Wall -shared -std=c++11 -fPIC $($pypath -m pybind11 --includes) retrieve.cpp -o libretrieval$($pyconfig --extension-suffix)
# done
for py_ver in "${versions[@]}"; do
    conda env remove -n build_py${py_ver} -y
done

mv *.so ../rgl/graph_retrieval/