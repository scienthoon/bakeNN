#ifndef BKNN_MNIST_KERNELS_H
#define BKNN_MNIST_KERNELS_H

#include <stdint.h>
#include <stddef.h>

int32_t bknn_mnist_q31_high_mul(int32_t a, int32_t b);
int32_t bknn_mnist_q31_round_div_pot(int32_t value, int32_t exponent);
int32_t bknn_mnist_q31_requantize(int32_t value, int32_t multiplier, int32_t shift);
int8_t bknn_mnist_q31_clamp_s8(int64_t value, int32_t minimum, int32_t maximum);

void bknn_mnist_conv2d_s8(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    const int32_t *multiplier,
    const int32_t *shift,
    int8_t *output,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t output_height,
    size_t output_width,
    size_t output_channels,
    size_t groups,
    size_t kernel_height,
    size_t kernel_width,
    size_t stride_height,
    size_t stride_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t pad_top,
    size_t pad_left,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t activation_min,
    int32_t activation_max);

void bknn_mnist_max_pool2d_s8(
    const int8_t *input,
    int8_t *output,
    size_t input_h, size_t input_w, size_t channels,
    size_t output_h, size_t output_w,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,
    size_t pad_top, size_t pad_left,
    int32_t activation_min, int32_t activation_max);

void bknn_mnist_linear_s8(
    const int8_t *input,
    const int8_t *weight,
    const int32_t *bias,
    const int32_t *multiplier,
    const int32_t *shift,
    int8_t *output,
    size_t input_count,
    size_t output_count,
    int32_t input_zero_point,
    int32_t output_zero_point,
    int32_t activation_min,
    int32_t activation_max);

#endif
