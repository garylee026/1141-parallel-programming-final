/*
  cnn_gpu_optimized.cu
  Optimized CUDA-accelerated Convolutional Neural Network
  Key optimization: Minimize CPU-GPU transfers, keep data on GPU
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cnn_gpu.h"

#define DEBUG_LAYER 0
#define BLOCK_SIZE 256
#define TILE_SIZE 16

/*  Misc. functions
 */

/* rnd(): uniform random [0.0, 1.0] */
static inline double rnd()
{
    return ((double)rand() / RAND_MAX);
}

/* nrnd(): normal random (std=1.0) */
static inline double nrnd()
{
    return (rnd()+rnd()+rnd()+rnd()-2.0) * 1.724; /* std=1.0 */
}

/* tanh_g(y): hyperbolic tangent gradient */
__host__ __device__ static inline double tanh_g(double y)
{
    return 1.0 - y*y;
}

/* relu(x): ReLU */
__host__ __device__ static inline double relu(double x)
{
    return (0 < x)? x : 0;
}

/* relu_g(y): ReLU gradient */
__host__ __device__ static inline double relu_g(double y)
{
    return (0 < y)? 1 : 0;
}



/*  Layer
 */

/* Layer_create(lprev, ltype, depth, width, height, nbiases, nweights)
   Creates a Layer object for internal use.
*/
static Layer* Layer_create(
    Layer* lprev, LayerType ltype,
    int depth, int width, int height,
    int nbiases, int nweights)
{
    Layer* self = (Layer*)calloc(1, sizeof(Layer));
    if (self == NULL) return NULL;

    self->lprev = lprev;
    self->lnext = NULL;
    self->ltype = ltype;
    self->lid = 0;
    if (lprev != NULL) {
        assert (lprev->lnext == NULL);
        lprev->lnext = self;
        self->lid = lprev->lid+1;
    }
    self->depth = depth;
    self->width = width;
    self->height = height;

    /* Nnodes: number of outputs. */
    self->nnodes = depth * width * height;
    self->outputs = (double*)calloc(self->nnodes, sizeof(double));
    self->gradients = (double*)calloc(self->nnodes, sizeof(double));
    self->errors = (double*)calloc(self->nnodes, sizeof(double));

    self->nbiases = nbiases;
    self->biases = (double*)calloc(self->nbiases, sizeof(double));
    self->u_biases = (double*)calloc(self->nbiases, sizeof(double));

    self->nweights = nweights;
    self->weights = (double*)calloc(self->nweights, sizeof(double));
    self->u_weights = (double*)calloc(self->nweights, sizeof(double));

    // Allocate GPU memory
    cudaMalloc(&self->d_outputs, self->nnodes * sizeof(double));
    cudaMalloc(&self->d_gradients, self->nnodes * sizeof(double));
    cudaMalloc(&self->d_errors, self->nnodes * sizeof(double));
    cudaMalloc(&self->d_biases, self->nbiases * sizeof(double));
    cudaMalloc(&self->d_u_biases, self->nbiases * sizeof(double));
    cudaMalloc(&self->d_weights, self->nweights * sizeof(double));
    cudaMalloc(&self->d_u_weights, self->nweights * sizeof(double));
    
    // Initialize GPU arrays to zero
    cudaMemset(self->d_u_biases, 0, self->nbiases * sizeof(double));
    cudaMemset(self->d_u_weights, 0, self->nweights * sizeof(double));

    return self;
}

/* Layer_destroy(self)
   Releases the memory.
*/
void Layer_destroy(Layer* self)
{
    assert (self != NULL);

    free(self->outputs);
    free(self->gradients);
    free(self->errors);

    free(self->biases);
    free(self->u_biases);
    free(self->weights);
    free(self->u_weights);

    // Free GPU memory
    cudaFree(self->d_outputs);
    cudaFree(self->d_gradients);
    cudaFree(self->d_errors);
    cudaFree(self->d_biases);
    cudaFree(self->d_u_biases);
    cudaFree(self->d_weights);
    cudaFree(self->d_u_weights);

    free(self);
}

/* Layer_dump(self, fp)
   Shows the debug output.
*/
void Layer_dump(const Layer* self, FILE* fp)
{
    assert (self != NULL);
    Layer* lprev = self->lprev;
    fprintf(fp, "Layer%d ", self->lid);
    if (lprev != NULL) {
        fprintf(fp, "(lprev=Layer%d) ", lprev->lid);
    }
    fprintf(fp, "shape=(%d,%d,%d), nodes=%d\n",
            self->depth, self->width, self->height, self->nnodes);
    
    // Copy from GPU to display
    cudaMemcpy(self->outputs, self->d_outputs, self->nnodes * sizeof(double), cudaMemcpyDeviceToHost);
    
    {
        int i = 0;
        for (int z = 0; z < self->depth; z++) {
            fprintf(fp, "  %d:\n", z);
            for (int y = 0; y < self->height; y++) {
                fprintf(fp, "    [");
                for (int x = 0; x < self->width; x++) {
                    fprintf(fp, " %.4f", self->outputs[i++]);
                }
                fprintf(fp, "]\n");
            }
        }
    }

    switch (self->ltype) {
    case LAYER_FULL:
        /* Fully connected layer. */
        assert (lprev != NULL);
        cudaMemcpy(self->biases, self->d_biases, self->nbiases * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(self->weights, self->d_weights, self->nweights * sizeof(double), cudaMemcpyDeviceToHost);
        
        fprintf(fp, "  biases = [");
        for (int i = 0; i < self->nnodes; i++) {
            fprintf(fp, " %.4f", self->biases[i]);
        }
        fprintf(fp, "]\n");
        fprintf(fp, "  weights = [\n");
        {
            int k = 0;
            for (int i = 0; i < self->nnodes; i++) {
                fprintf(fp, "    [");
                for (int j = 0; j < lprev->nnodes; j++) {
                    fprintf(fp, " %.4f", self->weights[k++]);
                }
                fprintf(fp, "]\n");
            }
        }
        fprintf(fp, "  ]\n");
        break;

    case LAYER_CONV:
        /* Convolutional layer. */
        assert (lprev != NULL);
        cudaMemcpy(self->biases, self->d_biases, self->nbiases * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(self->weights, self->d_weights, self->nweights * sizeof(double), cudaMemcpyDeviceToHost);
        
        fprintf(fp, "  stride=%d, kernsize=%d\n",
                self->conv.stride, self->conv.kernsize);
        {
            int k = 0;
            for (int z = 0; z < self->depth; z++) {
                fprintf(fp, "  %d: bias=%.4f, weights = [", z, self->biases[z]);
                for (int j = 0; j < lprev->depth * self->conv.kernsize * self->conv.kernsize; j++) {
                    fprintf(fp, " %.4f", self->weights[k++]);
                }
                fprintf(fp, "]\n");
            }
        }
        break;

    default:
        break;
    }
}

// CUDA kernel for fully connected layer forward pass
__global__ void kernel_feedForw_full(
    const double* __restrict__ inputs,
    const double* __restrict__ weights,
    const double* __restrict__ biases,
    double* __restrict__ outputs,
    double* __restrict__ gradients,
    int out_size, int in_size, bool is_output_layer)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < out_size) {
        double sum = biases[i];
        int w_base = i * in_size;
        for (int j = 0; j < in_size; j++) {
            sum += inputs[j] * weights[w_base + j];
        }
        
        if (!is_output_layer) {
            double y = tanh(sum);
            outputs[i] = y;
            gradients[i] = tanh_g(y);
        } else {
            outputs[i] = sum;  // Will be processed by softmax
        }
    }
}

// Optimized softmax: compute on CPU for small output (10 classes)
// This avoids multiple kernel launches and transfers for small arrays
static void compute_softmax_cpu(Layer* self)
{
    int size = self->nnodes;
    
    // Copy outputs from GPU
    cudaMemcpy(self->outputs, self->d_outputs, size * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Find max
    double max_val = self->outputs[0];
    for (int i = 1; i < size; i++) {
        if (self->outputs[i] > max_val) {
            max_val = self->outputs[i];
        }
    }
    
    // Exp and sum
    double total = 0.0;
    for (int i = 0; i < size; i++) {
        self->outputs[i] = exp(self->outputs[i] - max_val);
        total += self->outputs[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        self->outputs[i] /= total;
        self->gradients[i] = 1.0;
    }
    
    // Copy back to GPU
    cudaMemcpy(self->d_outputs, self->outputs, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(self->d_gradients, self->gradients, size * sizeof(double), cudaMemcpyHostToDevice);
}

/* Layer_feedForw_full(self)
   Performs feed forward updates - OPTIMIZED VERSION
*/
static void Layer_feedForw_full(Layer* self)
{
    assert (self->ltype == LAYER_FULL);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    const int out_size = self->nnodes;
    const int in_size = lprev->nnodes;
    
    // Data is already on GPU from previous operations
    bool is_output = (self->lnext == NULL);
    
    int numBlocks = (out_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_feedForw_full<<<numBlocks, BLOCK_SIZE>>>(
        lprev->d_outputs, self->d_weights, self->d_biases,
        self->d_outputs, self->d_gradients, out_size, in_size, is_output);

    if (is_output) {
        // For small output layers (like 10 classes), CPU softmax is faster
        // than multiple kernel launches
        compute_softmax_cpu(self);
    }
}

// CUDA kernel for fully connected backward pass - errors propagation
__global__ void kernel_feedBack_full_errors(
    const double* __restrict__ errors,
    const double* __restrict__ gradients,
    const double* __restrict__ weights,
    double* __restrict__ prev_errors,
    int out_size, int in_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < in_size) {
        double err = 0.0;
        for (int i = 0; i < out_size; i++) {
            double dnet = errors[i] * gradients[i];
            err += weights[i * in_size + j] * dnet;
        }
        prev_errors[j] = err;
    }
}

// CUDA kernel for weight updates accumulation
__global__ void kernel_feedBack_full_updates(
    const double* __restrict__ errors,
    const double* __restrict__ gradients,
    const double* __restrict__ prev_outputs,
    double* __restrict__ u_weights,
    double* __restrict__ u_biases,
    int out_size, int in_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < out_size) {
        double dnet = errors[i] * gradients[i];
        int w_base = i * in_size;
        for (int j = 0; j < in_size; j++) {
            atomicAdd(&u_weights[w_base + j], dnet * prev_outputs[j]);
        }
        atomicAdd(&u_biases[i], dnet);
    }
}

static void Layer_feedBack_full(Layer* self)
{
    assert (self->ltype == LAYER_FULL);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    const int out_size = self->nnodes;
    const int in_size = lprev->nnodes;

    // Clear previous layer errors on GPU
    cudaMemset(lprev->d_errors, 0, lprev->nnodes * sizeof(double));
    
    int numBlocks_in = (in_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_feedBack_full_errors<<<numBlocks_in, BLOCK_SIZE>>>(
        self->d_errors, self->d_gradients, self->d_weights,
        lprev->d_errors, out_size, in_size);
    
    int numBlocks_out = (out_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_feedBack_full_updates<<<numBlocks_out, BLOCK_SIZE>>>(
        self->d_errors, self->d_gradients, lprev->d_outputs,
        self->d_u_weights, self->d_u_biases, out_size, in_size);
}

// Helper to calculate layer index
__host__ __device__ static inline int layer_index(int depth, int width, int height, int z, int y, int x)
{
    return (z * height + y) * width + x;
}

// CUDA kernel for convolutional forward pass
__global__ void kernel_feedForw_conv(
    const double* __restrict__ inputs,
    const double* __restrict__ weights,
    const double* __restrict__ biases,
    double* __restrict__ outputs,
    double* __restrict__ gradients,
    int in_depth, int in_width, int in_height,
    int out_depth, int out_width, int out_height,
    int kernsize, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_depth * out_width * out_height;
    
    if (idx < total) {
        int z1 = idx / (out_width * out_height);
        int temp = idx % (out_width * out_height);
        int y1 = temp / out_width;
        int x1 = temp % out_width;
        
        int y0 = stride * y1 - padding;
        int x0 = stride * x1 - padding;
        
        double v = biases[z1];
        int qbase = z1 * in_depth * kernsize * kernsize;
        
        for (int z0 = 0; z0 < in_depth; z0++) {
            int pbase = z0 * in_width * in_height;
            
            for (int dy = 0; dy < kernsize; dy++) {
                int y = y0 + dy;
                if (y >= 0 && y < in_height) {
                    int p = pbase + y * in_width;
                    int q = qbase + dy * kernsize;
                    
                    for (int dx = 0; dx < kernsize; dx++) {
                        int x = x0 + dx;
                        if (x >= 0 && x < in_width) {
                            v += inputs[p + x] * weights[q + dx];
                        }
                    }
                }
            }
            qbase += kernsize * kernsize;
        }
        
        v = relu(v);
        outputs[idx] = v;
        gradients[idx] = relu_g(v);
    }
}

/* Layer_feedForw_conv(self)
   Performs feed forward updates.
*/
static void Layer_feedForw_conv(Layer* self)
{
    assert (self->ltype == LAYER_CONV);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    const int kernsize = self->conv.kernsize;
    const int stride = self->conv.stride;
    const int padding = self->conv.padding;

    int total = self->nnodes;
    int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    kernel_feedForw_conv<<<numBlocks, BLOCK_SIZE>>>(
        lprev->d_outputs, self->d_weights, self->d_biases,
        self->d_outputs, self->d_gradients,
        lprev->depth, lprev->width, lprev->height,
        self->depth, self->width, self->height,
        kernsize, stride, padding);
}

// CUDA kernel for convolutional backward pass
__global__ void kernel_feedBack_conv(
    const double* __restrict__ errors,
    const double* __restrict__ gradients,
    const double* __restrict__ inputs,
    const double* __restrict__ weights,
    double* __restrict__ prev_errors,
    double* __restrict__ u_weights,
    double* __restrict__ u_biases,
    int in_depth, int in_width, int in_height,
    int out_depth, int out_width, int out_height,
    int kernsize, int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_depth * out_width * out_height;
    
    if (idx < total) {
        int z1 = idx / (out_width * out_height);
        int temp = idx % (out_width * out_height);
        int y1 = temp / out_width;
        int x1 = temp % out_width;
        
        int y0 = stride * y1 - padding;
        int x0 = stride * x1 - padding;
        
        double dnet = errors[idx] * gradients[idx];
        int qbase = z1 * in_depth * kernsize * kernsize;
        
        for (int z0 = 0; z0 < in_depth; z0++) {
            int pbase = z0 * in_width * in_height;
            
            for (int dy = 0; dy < kernsize; dy++) {
                int y = y0 + dy;
                if (y >= 0 && y < in_height) {
                    int p = pbase + y * in_width;
                    int q = qbase + dy * kernsize;
                    
                    for (int dx = 0; dx < kernsize; dx++) {
                        int x = x0 + dx;
                        if (x >= 0 && x < in_width) {
                            atomicAdd(&prev_errors[p + x], weights[q + dx] * dnet);
                            atomicAdd(&u_weights[q + dx], dnet * inputs[p + x]);
                        }
                    }
                }
            }
            qbase += kernsize * kernsize;
        }
        
        atomicAdd(&u_biases[z1], dnet);
    }
}

static void Layer_feedBack_conv(Layer* self)
{
    assert (self->ltype == LAYER_CONV);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    // Clear errors on GPU
    cudaMemset(lprev->d_errors, 0, lprev->nnodes * sizeof(double));

    const int kernsize = self->conv.kernsize;
    const int stride = self->conv.stride;
    const int padding = self->conv.padding;

    int total = self->nnodes;
    int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    kernel_feedBack_conv<<<numBlocks, BLOCK_SIZE>>>(
        self->d_errors, self->d_gradients, lprev->d_outputs,
        self->d_weights, lprev->d_errors, self->d_u_weights, self->d_u_biases,
        lprev->depth, lprev->width, lprev->height,
        self->depth, self->width, self->height,
        kernsize, stride, padding);
}

/* Layer_setInputs(self, values)
   Sets the input values.
*/
void Layer_setInputs(Layer* self, const double* values)
{
    assert (self != NULL);
    assert (self->ltype == LAYER_INPUT);
    assert (self->lprev == NULL);

    // Copy input to GPU (only transfer at the beginning of pipeline)
    cudaMemcpy(self->d_outputs, values, self->nnodes * sizeof(double), cudaMemcpyHostToDevice);

    /* Start feed forwarding - all operations stay on GPU */
    Layer* layer = self->lnext;
    while (layer != NULL) {
        switch (layer->ltype) {
        case LAYER_FULL:
            Layer_feedForw_full(layer);
            break;
        case LAYER_CONV:
            Layer_feedForw_conv(layer);
            break;
        default:
            break;
        }
        layer = layer->lnext;
    }
    
    // Sync once after entire forward pass completes
    cudaDeviceSynchronize();
}

/* Layer_getOutputs(self, outputs)
   Gets the output values.
*/
void Layer_getOutputs(const Layer* self, double* outputs)
{
    assert (self != NULL);
    // Copy from GPU only when needed
    cudaMemcpy(outputs, self->d_outputs, self->nnodes * sizeof(double), cudaMemcpyDeviceToHost);
}

/* Layer_getErrorTotal(self)
   Gets the error total.
*/
double Layer_getErrorTotal(const Layer* self)
{
    assert (self != NULL);
    // Copy errors from GPU
    double* errors = (double*)malloc(self->nnodes * sizeof(double));
    cudaMemcpy(errors, self->d_errors, self->nnodes * sizeof(double), cudaMemcpyDeviceToHost);
    
    double total = 0;
    for (int i = 0; i < self->nnodes; i++) {
        double e = errors[i];
        total += e*e;
    }
    free(errors);
    return (total / self->nnodes);
}

/* Layer_learnOutputs(self, values)
   Learns the output values.
*/
void Layer_learnOutputs(Layer* self, const double* values)
{
    assert (self != NULL);
    assert (self->ltype != LAYER_INPUT);
    assert (self->lprev != NULL);
    
    // Compute errors on CPU (small array)
    cudaMemcpy(self->outputs, self->d_outputs, self->nnodes * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < self->nnodes; i++) {
        self->errors[i] = (self->outputs[i] - values[i]);
    }
    
    // Copy errors to GPU once
    cudaMemcpy(self->d_errors, self->errors, self->nnodes * sizeof(double), cudaMemcpyHostToDevice);

    /* Start backpropagation - all operations stay on GPU */
    Layer* layer = self;
    while (layer != NULL) {
        switch (layer->ltype) {
        case LAYER_FULL:
            Layer_feedBack_full(layer);
            break;
        case LAYER_CONV:
            Layer_feedBack_conv(layer);
            break;
        default:
            break;
        }
        layer = layer->lprev;
    }
    
    // Sync once after entire backward pass completes
    cudaDeviceSynchronize();
}

// CUDA kernel for weight updates
__global__ void kernel_update_params(
    double* __restrict__ params,
    double* __restrict__ u_params,
    int size, double rate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        params[i] -= rate * u_params[i];
        u_params[i] = 0.0;
    }
}

/* Layer_update(self, rate)
   Updates the weights - OPTIMIZED VERSION
*/
void Layer_update(Layer* self, double rate)
{
    if (self->nbiases > 0) {
        int numBlocks = (self->nbiases + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_update_params<<<numBlocks, BLOCK_SIZE>>>(
            self->d_biases, self->d_u_biases, self->nbiases, rate);
    }
    
    if (self->nweights > 0) {
        int numBlocks = (self->nweights + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_update_params<<<numBlocks, BLOCK_SIZE>>>(
            self->d_weights, self->d_u_weights, self->nweights, rate);
    }
    
    if (self->lprev != NULL) {
        Layer_update(self->lprev, rate);
    } else {
        // Sync once after all weight updates complete (at root layer)
        cudaDeviceSynchronize();
    }
}

/* Layer_create_input(depth, width, height)
   Creates an input Layer with size (depth x weight x height).
*/
Layer* Layer_create_input(int depth, int width, int height)
{
    return Layer_create(
        NULL, LAYER_INPUT, depth, width, height, 0, 0);
}

/* Layer_create_full(lprev, nnodes, std)
   Creates a fully-connected Layer.
*/
Layer* Layer_create_full(Layer* lprev, int nnodes, double std)
{
    assert (lprev != NULL);
    Layer* self = Layer_create(
        lprev, LAYER_FULL, nnodes, 1, 1,
        nnodes, nnodes * lprev->nnodes);
    assert (self != NULL);

    for (int i = 0; i < self->nweights; i++) {
        self->weights[i] = std * nrnd();
    }
    
    // Copy initialized weights to GPU
    cudaMemcpy(self->d_weights, self->weights, self->nweights * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(self->d_biases, self->biases, self->nbiases * sizeof(double), cudaMemcpyHostToDevice);

    return self;
}

/* Layer_create_conv(lprev, depth, width, height, kernsize, padding, stride, std)
   Creates a convolutional Layer.
*/
Layer* Layer_create_conv(
    Layer* lprev, int depth, int width, int height,
    int kernsize, int padding, int stride, double std)
{
    assert (lprev != NULL);
    assert ((kernsize % 2) == 1);
    assert ((width-1) * stride + kernsize <= lprev->width + padding*2);
    assert ((height-1) * stride + kernsize <= lprev->height + padding*2);

    Layer* self = Layer_create(
        lprev, LAYER_CONV, depth, width, height,
        depth, depth * lprev->depth * kernsize * kernsize);
    assert (self != NULL);

    self->conv.kernsize = kernsize;
    self->conv.padding = padding;
    self->conv.stride = stride;

    for (int i = 0; i < self->nweights; i++) {
        self->weights[i] = std * nrnd();
    }
    
    // Copy initialized weights to GPU
    cudaMemcpy(self->d_weights, self->weights, self->nweights * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(self->d_biases, self->biases, self->nbiases * sizeof(double), cudaMemcpyHostToDevice);

    return self;
}

void Layer_print_profile(void)
{
    // Profiling removed - use nvprof or nsys instead:
    // nvprof ./mnist_gpu_opt <args>
    // nsys profile --stats=true ./mnist_gpu_opt <args>
}