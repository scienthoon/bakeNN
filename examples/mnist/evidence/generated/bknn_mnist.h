#ifndef BKNN_MNIST_H
#define BKNN_MNIST_H

#include <stdint.h>

#ifndef BKNN_LAYOUT_NHWC
#define BKNN_LAYOUT_NHWC 1u
#endif
#ifndef BKNN_LAYOUT_NC
#define BKNN_LAYOUT_NC 2u
#endif

#define BKNN_MNIST_INPUT_SIZE 784u
#define BKNN_MNIST_INPUT_BYTES 784u
#define BKNN_MNIST_INPUT_RANK 4u
#define BKNN_MNIST_INPUT_DIM_0 1u
#define BKNN_MNIST_INPUT_DIM_1 28u
#define BKNN_MNIST_INPUT_DIM_2 28u
#define BKNN_MNIST_INPUT_DIM_3 1u
#define BKNN_MNIST_INPUT_LAYOUT BKNN_LAYOUT_NHWC
#define BKNN_MNIST_OUTPUT_SIZE 10u
#define BKNN_MNIST_OUTPUT_BYTES 10u
#define BKNN_MNIST_OUTPUT_RANK 2u
#define BKNN_MNIST_OUTPUT_DIM_0 1u
#define BKNN_MNIST_OUTPUT_DIM_1 10u
#define BKNN_MNIST_OUTPUT_LAYOUT BKNN_LAYOUT_NC
#define BKNN_MNIST_ARENA_SIZE 3920u
#define BKNN_MNIST_ARENA_ALIGNMENT 16u
#define BKNN_MNIST_INPUT_SCALE 0x1.0101020000000p-8f
#define BKNN_MNIST_INPUT_ZERO_POINT -128
#define BKNN_MNIST_OUTPUT_SCALE 0x1.57d4ee0000000p-3f
#define BKNN_MNIST_OUTPUT_ZERO_POINT 29

/* input, output, and arena must not overlap. arena may be NULL when ARENA_SIZE is zero. */
void bknn_mnist_infer(
    uint8_t *restrict arena,
    const int8_t *restrict input,
    int8_t *restrict output);

#endif
