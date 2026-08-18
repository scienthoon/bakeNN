#include "bknn_mnist_kernels.h"

#include <limits.h>

int32_t bknn_mnist_q31_high_mul(int32_t a, int32_t b) {
    if (a == INT32_MIN && b == INT32_MIN) {
        return INT32_MAX;
    }
    const int64_t product = (int64_t)a * (int64_t)b;
    const int64_t nudge =
        product >= 0 ? (INT64_C(1) << 30) : INT64_C(1) - (INT64_C(1) << 30);
    return (int32_t)((product + nudge) / (INT64_C(1) << 31));
}

int32_t bknn_mnist_q31_round_div_pot(int32_t value, int32_t exponent) {
    if (exponent == 0) {
        return value;
    }
    const uint64_t magnitude =
        value < 0 ? (uint64_t)(-(int64_t)value) : (uint64_t)value;
    uint64_t quotient = magnitude >> (uint32_t)exponent;
    const uint64_t remainder =
        magnitude & ((UINT64_C(1) << (uint32_t)exponent) - UINT64_C(1));
    if (remainder >= (UINT64_C(1) << (uint32_t)(exponent - 1))) {
        ++quotient;
    }
    return value < 0 ? (int32_t)(-(int64_t)quotient) : (int32_t)quotient;
}

int32_t bknn_mnist_q31_requantize(int32_t value, int32_t multiplier, int32_t shift) {
    const int32_t left_shift = shift > 0 ? shift : 0;
    const int32_t right_shift = shift < 0 ? -shift : 0;
    const int64_t shifted64 =
        (int64_t)value * (INT64_C(1) << (uint32_t)left_shift);
    const int32_t shifted = (int32_t)shifted64;
    return bknn_mnist_q31_round_div_pot(bknn_mnist_q31_high_mul(shifted, multiplier), right_shift);
}

int8_t bknn_mnist_q31_clamp_s8(int64_t value, int32_t minimum, int32_t maximum) {
    if (value < minimum) {
        value = minimum;
    } else if (value > maximum) {
        value = maximum;
    }
    return (int8_t)value;
}

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
    int32_t activation_max) {
    for (size_t output_y = 0; output_y < output_height; ++output_y) {
        for (size_t output_x = 0; output_x < output_width; ++output_x) {
            const size_t group_input_channels = input_channels / groups;
            const size_t group_output_channels = output_channels / groups;
            for (size_t output_channel = 0; output_channel < output_channels; ++output_channel) {
                const size_t group = output_channel / group_output_channels;
                const size_t input_channel_base = group * group_input_channels;
                int32_t accumulator = bias[output_channel];
                for (size_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
                    const int64_t input_y = (int64_t)output_y * (int64_t)stride_height
                        + (int64_t)kernel_y * (int64_t)dilation_height - (int64_t)pad_top;
                    for (size_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                        const int64_t input_x = (int64_t)output_x * (int64_t)stride_width
                            + (int64_t)kernel_x * (int64_t)dilation_width - (int64_t)pad_left;
                        for (size_t local_input_channel = 0;
                             local_input_channel < group_input_channels;
                             ++local_input_channel) {
                            const size_t input_channel =
                                input_channel_base + local_input_channel;
                            int32_t input_value = input_zero_point;
                            if (input_y >= 0 && input_x >= 0
                                && (uint64_t)input_y < (uint64_t)input_height
                                && (uint64_t)input_x < (uint64_t)input_width) {
                                const size_t input_index =
                                    ((size_t)input_y * input_width + (size_t)input_x) * input_channels
                                    + input_channel;
                                input_value = input[input_index];
                            }
                            const size_t weight_index =
                                ((output_channel * kernel_height + kernel_y) * kernel_width + kernel_x)
                                * group_input_channels + local_input_channel;
                            accumulator += (input_value - input_zero_point) * (int32_t)weight[weight_index];
                        }
                    }
                }
                const int32_t scaled = bknn_mnist_q31_requantize(
                    accumulator, multiplier[output_channel], shift[output_channel]);
                const size_t output_index =
                    (output_y * output_width + output_x) * output_channels + output_channel;
                output[output_index] = bknn_mnist_q31_clamp_s8(
                    (int64_t)scaled + output_zero_point, activation_min, activation_max);
            }
        }
    }
}

void bknn_mnist_max_pool2d_s8(
    const int8_t *input,
    int8_t *output,
    size_t input_h, size_t input_w, size_t channels,
    size_t output_h, size_t output_w,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,
    size_t pad_top, size_t pad_left,
    int32_t activation_min, int32_t activation_max) {
    for (size_t output_y = 0; output_y < output_h; ++output_y) {
        for (size_t output_x = 0; output_x < output_w; ++output_x) {
            for (size_t channel = 0; channel < channels; ++channel) {
                int32_t result = INT32_MIN;
                for (size_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
                    const int64_t input_y = (int64_t)output_y * (int64_t)stride_h
                        + (int64_t)kernel_y - (int64_t)pad_top;
                    if (input_y < 0 || input_y >= (int64_t)input_h) {
                        continue;
                    }
                    for (size_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
                        const int64_t input_x = (int64_t)output_x * (int64_t)stride_w
                            + (int64_t)kernel_x - (int64_t)pad_left;
                        if (input_x < 0 || input_x >= (int64_t)input_w) {
                            continue;
                        }
                        const size_t input_index =
                            (((size_t)input_y * input_w + (size_t)input_x) * channels) + channel;
                        const int32_t candidate = (int32_t)input[input_index];
                        if (candidate > result) {
                            result = candidate;
                        }
                    }
                }
                if (result < activation_min) {
                    result = activation_min;
                } else if (result > activation_max) {
                    result = activation_max;
                }
                const size_t output_index =
                    ((output_y * output_w + output_x) * channels) + channel;
                output[output_index] = (int8_t)result;
            }
        }
    }
}

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
    int32_t activation_max) {
    for (size_t channel = 0; channel < output_count; ++channel) {
        int32_t accumulator = bias[channel];
        const int8_t *channel_weight = weight + channel * input_count;
        for (size_t index = 0; index < input_count; ++index) {
            accumulator +=
                ((int32_t)input[index] - input_zero_point)
                * (int32_t)channel_weight[index];
        }
        const int32_t scaled =
            bknn_mnist_q31_requantize(accumulator, multiplier[channel], shift[channel]);
        output[channel] = bknn_mnist_q31_clamp_s8(
            (int64_t)scaled + output_zero_point, activation_min, activation_max);
    }
}
