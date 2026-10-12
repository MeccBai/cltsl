# CUDA CMake project

Configured for the RTX 2080 Ti (SM 75) and CUDA Toolkit 12.8 on this instance.
The demo reduces 1,025 values using a 100-thread block, exercising a partial
warp and a short final output tile.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/cuda_demo
```

Expected results are `output[0]=320400.0` and `output[1]=205425.0`.
CMake auto-detects `/usr/local/cuda/bin/nvcc` even when CUDA is not in `PATH`.
For another GPU, change `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt`.
