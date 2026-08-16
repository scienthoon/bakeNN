// tvm target: cmsis-nn 
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlpack/dlpack.h>
#include <arm_nnfunctions.h>
#include <arm_nn_types.h>
#include <arm_nn_math_types.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_0(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_, uint8_t* global_const_workspace_2_var, uint8_t* global_workspace_3_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_1(int8_t* input_, int8_t* output_, uint8_t* global_const_workspace_4_var, uint8_t* global_workspace_5_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_2(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_, uint8_t* global_const_workspace_6_var, uint8_t* global_workspace_7_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_3(int8_t* input_, int8_t* output_, uint8_t* global_const_workspace_8_var, uint8_t* global_workspace_9_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_4(int8_t* input_, int8_t* filter_, int32_t* bias_, int8_t* output_, uint8_t* global_const_workspace_10_var, uint8_t* global_workspace_11_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_0(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_, uint8_t* global_const_workspace_2_var, uint8_t* global_workspace_3_var) {
  cmsis_nn_context context= {NULL,0};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {1,1};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_dw_conv_params conv_params = {128, -128, 4, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,28,28,1};
  cmsis_nn_dims filter_dims = {1,3,3,4};
  cmsis_nn_dims bias_dims = {1,1,1,4};
  cmsis_nn_dims output_dims = {1,28,28,4};
  arm_cmsis_nn_status status = arm_depthwise_conv_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  switch (!status) {
  case ARM_CMSIS_NN_SUCCESS: break;
  case ARM_CMSIS_NN_ARG_ERROR: return -1;
  case ARM_CMSIS_NN_NO_IMPL_ERROR: return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_1(int8_t* input_, int8_t* output_, uint8_t* global_const_workspace_4_var, uint8_t* global_workspace_5_var) {
  cmsis_nn_context context= {NULL,0};
  cmsis_nn_tile stride = {2,2};
  cmsis_nn_tile padding = {0,0};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_pool_params pool_params = {stride, padding, activation};
  cmsis_nn_dims input_dims = {1,28,28,4};
  cmsis_nn_dims filter_dims = {1,2,2,1};
  cmsis_nn_dims output_dims = {1,14,14,4};
  arm_cmsis_nn_status status = arm_max_pool_s8(&context, &pool_params, &input_dims, input_, &filter_dims, &output_dims, output_);
  switch (!status) {
  case ARM_CMSIS_NN_SUCCESS: break;
  case ARM_CMSIS_NN_ARG_ERROR: return -1;
  case ARM_CMSIS_NN_NO_IMPL_ERROR: return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_2(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_, uint8_t* global_const_workspace_6_var, uint8_t* global_workspace_7_var) {
  void* context_buffer_0_let = (&(global_workspace_7_var[3920]));
  cmsis_nn_context context= {context_buffer_0_let,144};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {1,1};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_conv_params conv_params = {128, -128, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,14,14,4};
  cmsis_nn_dims filter_dims = {8,3,3,4};
  cmsis_nn_dims bias_dims = {1,1,1,8};
  cmsis_nn_dims output_dims = {1,14,14,8};
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  switch (!status) {
  case ARM_CMSIS_NN_SUCCESS: break;
  case ARM_CMSIS_NN_ARG_ERROR: return -1;
  case ARM_CMSIS_NN_NO_IMPL_ERROR: return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_3(int8_t* input_, int8_t* output_, uint8_t* global_const_workspace_8_var, uint8_t* global_workspace_9_var) {
  cmsis_nn_context context= {NULL,0};
  cmsis_nn_tile stride = {2,2};
  cmsis_nn_tile padding = {0,0};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_pool_params pool_params = {stride, padding, activation};
  cmsis_nn_dims input_dims = {1,14,14,8};
  cmsis_nn_dims filter_dims = {1,2,2,1};
  cmsis_nn_dims output_dims = {1,7,7,8};
  arm_cmsis_nn_status status = arm_max_pool_s8(&context, &pool_params, &input_dims, input_, &filter_dims, &output_dims, output_);
  switch (!status) {
  case ARM_CMSIS_NN_SUCCESS: break;
  case ARM_CMSIS_NN_ARG_ERROR: return -1;
  case ARM_CMSIS_NN_NO_IMPL_ERROR: return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_4(int8_t* input_, int8_t* filter_, int32_t* bias_, int8_t* output_, uint8_t* global_const_workspace_10_var, uint8_t* global_workspace_11_var) {
  cmsis_nn_context context= {NULL,0};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_fc_params fc_params = {128, 0, 29, activation};
  cmsis_nn_per_tensor_quant_params quant_params = {2121357441, -10};
  cmsis_nn_dims input_dims = {1,1,1,392};
  cmsis_nn_dims filter_dims = {392,1,1,10};
  cmsis_nn_dims bias_dims = {1,1,1,10};
  cmsis_nn_dims output_dims = {1,1,1,10};
  arm_cmsis_nn_status status = arm_fully_connected_s8(&context, &fc_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  switch (!status) {
  case ARM_CMSIS_NN_SUCCESS: break;
  case ARM_CMSIS_NN_ARG_ERROR: return -1;
  case ARM_CMSIS_NN_NO_IMPL_ERROR: return -1;
  }
  return 0;
}

