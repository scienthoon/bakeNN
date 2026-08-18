#include "bknn_mnist.h"
#include "bknn_mnist_kernels.h"
#include "bknn_mnist_weights.h"

void bknn_mnist_infer(
    uint8_t *restrict arena,
    const int8_t *restrict input,
    int8_t *restrict output) {
    (void)arena;
    bknn_mnist_conv2d_s8(
        input,
        bknn_mnist_constant_1,
        bknn_mnist_constant_0,
        bknn_mnist_op0_multiplier, bknn_mnist_op0_shift,
        (int8_t *)(void *)(arena + 0u),
        28u, 28u, 1u,
        28u, 28u, 4u,
        1u,
        3u, 3u,
        1u, 1u,
        1u, 1u,
        1u, 1u,
        -128, -128,
        -128, 127);

    bknn_mnist_max_pool2d_s8(
        (const int8_t *)(void *)(arena + 0u),
        (int8_t *)(void *)(arena + 3136u),
        28u, 28u, 4u,
        14u, 14u,
        2u, 2u, 2u, 2u,
        0u, 0u,
        -128, 127);

    bknn_mnist_conv2d_s8(
        (const int8_t *)(void *)(arena + 3136u),
        bknn_mnist_constant_3,
        bknn_mnist_constant_2,
        bknn_mnist_op2_multiplier, bknn_mnist_op2_shift,
        (int8_t *)(void *)(arena + 0u),
        14u, 14u, 4u,
        14u, 14u, 8u,
        1u,
        3u, 3u,
        1u, 1u,
        1u, 1u,
        1u, 1u,
        -128, -128,
        -128, 127);

    bknn_mnist_max_pool2d_s8(
        (const int8_t *)(void *)(arena + 0u),
        (int8_t *)(void *)(arena + 1568u),
        14u, 14u, 8u,
        7u, 7u,
        2u, 2u, 2u, 2u,
        0u, 0u,
        -128, 127);

    /* flatten: flatten_view; storage aliases max_pool2d_1. */

    bknn_mnist_linear_s8(
        (const int8_t *)(void *)(arena + 1568u),
        bknn_mnist_constant_5,
        bknn_mnist_constant_4,
        bknn_mnist_op5_multiplier,
        bknn_mnist_op5_shift,
        output,
        392u, 10u,
        -128, 29,
        -128, 127);
}
