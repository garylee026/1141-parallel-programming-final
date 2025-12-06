# Makefile for CUDA-accelerated CNN

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70 -Xcompiler -Wall

# CPU version (for comparison)
CC = gcc
CFLAGS = -O2 -Wall

# Targets
GPU_TARGET = mnist_gpu
CPU_TARGET = mnist_cpu

# Source files
GPU_SRCS = mnist_gpu.cu cnn_gpu.cu
CPU_SRCS = sample/mnist.c sample/cnn.c

# Default target
all: $(GPU_TARGET)

# GPU version
$(GPU_TARGET): $(GPU_SRCS) cnn_gpu.h
	$(NVCC) $(NVCC_FLAGS) -o $(GPU_TARGET) $(GPU_SRCS) -lm

# CPU version (for comparison)
cpu: $(CPU_SRCS) sample/cnn.h
	$(CC) $(CFLAGS) -o $(CPU_TARGET) $(CPU_SRCS) -lm

# Run GPU version
run_gpu: $(GPU_TARGET)
	./$(GPU_TARGET) train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte

# Run CPU version
run_cpu: cpu
	./$(CPU_TARGET) train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte

# Clean
clean:
	rm -f $(GPU_TARGET) $(CPU_TARGET)

.PHONY: all cpu run_gpu run_cpu clean