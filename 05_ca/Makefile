CXX = g++
# CXXFLAGS = -g -Wall -Wno-unused-local-typedefs -Og -pipe -std=c++17
CXXFLAGS = -Wall -Wno-unused-local-typedefs -O3 -pipe -std=c++17
HPXAPPFLAGS = `pkg-config --cflags hpx_application`
HPXAPPLIBS = `pkg-config --libs hpx_application` -lhpx_iostreams
HPXCMPFLAGS = `pkg-config --cflags hpx_component`
HPXCMPLIBS = `pkg-config --libs hpx_component` -lhpx_iostreams
# CUDA 8.0
# CUDAFLAGS = -std=c++11 -arch=sm_61 -g -G
# CUDALIBS = -L/usr/local/cuda-8.0/lib64 -lcudart
# CUDA 9.0
# CUDAFLAGS = -g -G -std=c++14 -arch=sm_61
CUDAFLAGS = -O2 -std=c++14 -arch=sm_61
CUDALIBS = -L/usr/local/cuda-9.0/lib64 -lcudart
# CUDAXFLAGS = -Xcompiler '-fPIC -pipe -g -Og'
CUDAXFLAGS = -Xcompiler '-fPIC -pipe -O3'
DIR = build
EXEC = $(DIR)/ca
OBJ := CellularAutomaton.o CUDACellularAutomaton.o
OBJ := $(addprefix $(DIR)/, $(OBJ))

all: $(EXEC)

$(EXEC): $(DIR)/libkernels.so main.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) $(HPXAPPFLAGS) $(wordlist 2, $(words $^), $^) -o $@ $(HPXAPPLIBS) $(CUDALIBS) -L$(DIR) -lkernels -D HPX_APPLICATION

$(DIR)/kernel_wrappers.o: Kernel/kernel_wrappers.cu Kernel/kernel_wrappers.hpp Kernel/kernels.hpp
	mkdir -p $(DIR)
	nvcc $(CUDAFLAGS) $(CUDAXFLAGS) -dc $< -o $@

$(DIR)/CellularAutomaton.o: CellularAutomaton/CellularAutomaton.cpp CellularAutomaton/CellularAutomaton.hpp
	mkdir -p $(DIR)
	$(CXX) $(CXXFLAGS) $(HPXAPPFLAGS) -c $< -o $@ -D HPX_COMPONENT_NAME=CellularAutomaton

$(DIR)/CUDACellularAutomaton.o: CellularAutomaton/CUDACellularAutomaton.cpp CellularAutomaton/CUDACellularAutomaton.hpp
	$(CXX) $(CXXFLAGS) $(HPXAPPFLAGS) -c $< -o $@ -D HPX_COMPONENT_NAME=CUDACellularAutomaton

$(DIR)/gpu_code.o: $(DIR)/kernel_wrappers.o
	nvcc $(CUDAFLAGS) $(CUDAXFLAGS) -dlink $^ -o $@

$(DIR)/libkernels.so: $(DIR)/gpu_code.o $(DIR)/kernel_wrappers.o
	$(CXX) -shared -fPIC $(CXXFLAGS) $^ -o $@ $(CUDALIBS)

$(DIR)/test_CUDA_CA: $(DIR)/libkernels.so CellularAutomaton/testCUDACellularAutomaton.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) $(HPXAPPFLAGS) $(wordlist 2, $(words $^), $^) -o $@ $(HPXAPPLIBS) $(CUDALIBS) -L$(DIR) -lkernels -D HPX_APPLICATION

clean:
	rm -fr $(DIR)

.PHONY: all clean
