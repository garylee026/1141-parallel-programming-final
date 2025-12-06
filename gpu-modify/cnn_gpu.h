/*
  cnn_gpu.h
  CUDA-accelerated Convolutional Neural Network
*/

#ifndef CNN_GPU_H
#define CNN_GPU_H

/*  LayerType
 */
typedef enum _LayerType {
    LAYER_INPUT = 0,
    LAYER_FULL,
    LAYER_CONV
} LayerType;


/*  Layer
 */
typedef struct _Layer {

    int lid;                    /* Layer ID */
    struct _Layer* lprev;       /* Previous Layer */
    struct _Layer* lnext;       /* Next Layer */

    int depth, width, height;   /* Shape */

    int nnodes;                 /* Num. of Nodes */
    double* outputs;            /* Node Outputs (CPU) */
    double* gradients;          /* Node Gradients (CPU) */
    double* errors;             /* Node Errors (CPU) */

    int nbiases;                /* Num. of Biases */
    double* biases;             /* Biases (trained) (CPU) */
    double* u_biases;           /* Bias updates (CPU) */

    int nweights;               /* Num. of Weights */
    double* weights;            /* Weights (trained) (CPU) */
    double* u_weights;          /* Weight updates (CPU) */

    // GPU memory pointers
    double* d_outputs;          /* Node Outputs (GPU) */
    double* d_gradients;        /* Node Gradients (GPU) */
    double* d_errors;           /* Node Errors (GPU) */
    double* d_biases;           /* Biases (GPU) */
    double* d_u_biases;         /* Bias updates (GPU) */
    double* d_weights;          /* Weights (GPU) */
    double* d_u_weights;        /* Weight updates (GPU) */

    LayerType ltype;            /* Layer type */
    union {
        /* Full */
        struct {
        } full;

        /* Conv */
        struct {
            int kernsize;       /* kernel size (>0) */
            int padding;        /* padding size */
            int stride;         /* stride (>0) */
        } conv;
    };

} Layer;

/* Layer_create_input(depth, width, height)
   Creates an input Layer with size (depth x weight x height).
*/
Layer* Layer_create_input(
    int depth, int width, int height);

/* Layer_create_full(lprev, nnodes, std)
   Creates a fully-connected Layer.
*/
Layer* Layer_create_full(
    Layer* lprev, int nnodes, double std);

/* Layer_create_conv(lprev, depth, width, height, kernsize, padding, stride, std)
   Creates a convolutional Layer.
*/
Layer* Layer_create_conv(
    Layer* lprev, int depth, int width, int height,
    int kernsize, int padding, int stride, double std);

/* Layer_destroy(self)
   Releases the memory.
*/
void Layer_destroy(Layer* self);

/* Layer_dump(self, fp)
   Shows the debug output.
*/
void Layer_dump(const Layer* self, FILE* fp);

/* Layer_setInputs(self, values)
   Sets the input values.
*/
void Layer_setInputs(Layer* self, const double* values);

/* Layer_getOutputs(self, outputs)
   Gets the output values.
*/
void Layer_getOutputs(const Layer* self, double* outputs);

/* Layer_getErrorTotal(self)
   Gets the error total.
*/
double Layer_getErrorTotal(const Layer* self);

/* Layer_learnOutputs(self, values)
   Learns the output values.
*/
void Layer_learnOutputs(Layer* self, const double* values);

/* Layer_update(self, rate)
   Updates the weights.
*/
void Layer_update(Layer* self, double rate);

void Layer_print_profile(void);

#endif // CNN_GPU_H
