## Accelerating Approximate Triangle Counting on CUDA

### Set GPU in Multi-GPU environment
- `export CUDA_VISIBLE_DEVICES=1` in bash before running

### Tools
- `cmake --version` >= 3.24
- `C++17`
- `CUDA Version X.X`

### Datasets
- `cd ./datasets`
- `make` to download all datasets
- `cd XYZ/; make` or `make XYZ` to download a specific dataset

### To compile
```
make
cd build/
```
Or by hand,
```
mkdir build && cd build;
cmake ..
make
```

### To run
```
cd build
# Exact
./bin/tc -m ../datasets/chesapeake/chesapeake.mtx
# Approximate 
./bin/tc -m ../datasets/chesapeake/chesapeake.mtx -s 0.5 
```
### CLI
```
./build$ bin/tc --help
Accelerating Approximate Triangle Counting
Usage:
  bin/tc [OPTION...]

      --help          Print help
  -v, --validate      CPU validation using Node-Iterator (default: true)
  -m, --market arg    Matrix file
  -r, --reduce        Compute a single triangle count for the entire graph 
                      (default: true)
  -s, --sparsify arg  Do approximate triangle counting, and set edge 
                      sparsification percentage (0-1) (default: false)
```