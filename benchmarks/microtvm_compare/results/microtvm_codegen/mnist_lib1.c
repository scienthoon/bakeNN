// tvm target: c -keys=arm_cpu,cpu -mcpu=cortex-m4
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_mnist___tvm_main__(int8_t* value_buffer_var, int8_t* linear_buffer_var, uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_0(int8_t*, int8_t*, int32_t*, int32_t*, int32_t*, int8_t*, uint8_t*, uint8_t*);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_1(int8_t*, int8_t*, uint8_t*, uint8_t*);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_2(int8_t*, int8_t*, int32_t*, int32_t*, int32_t*, int8_t*, uint8_t*, uint8_t*);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_3(int8_t*, int8_t*, uint8_t*, uint8_t*);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_4(int8_t*, int8_t*, int32_t*, int8_t*, uint8_t*, uint8_t*);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_mnist___tvm_main__(int8_t* value_buffer_var, int8_t* linear_buffer_var, uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var) {
  void* constant_5_let = (&(global_const_workspace_0_var[4464]));
  void* constant_6_let = (&(global_const_workspace_0_var[3920]));
  void* constant_2_let = (&(global_const_workspace_0_var[4512]));
  void* constant_3_let = (&(global_const_workspace_0_var[4496]));
  void* constant_4_let = (&(global_const_workspace_0_var[4480]));
  void* constant_11_let = (&(global_const_workspace_0_var[4400]));
  void* constant_13_let = (&(global_const_workspace_0_var[4208]));
  void* constant_1_let = (&(global_const_workspace_0_var[4528]));
  void* constant_9_let = (&(global_const_workspace_0_var[4304]));
  void* constant_7_let = (&(global_const_workspace_0_var[4368]));
  void* constant_8_let = (&(global_const_workspace_0_var[4336]));
  void* constant_10_let = (&(global_const_workspace_0_var[4432]));
  void* constant_0_let = (&(global_const_workspace_0_var[4256]));
  void* constant_12_let = (&(global_const_workspace_0_var[0]));
  void* sid_7_let = (&(global_workspace_1_var[0]));
  void* sid_15_let = (&(global_workspace_1_var[0]));
  void* sid_8_let = (&(global_workspace_1_var[3136]));
  void* sid_16_let = (&(global_workspace_1_var[1568]));
  if (tvmgen_default_cmsis_nn_main_0(value_buffer_var, constant_0_let, constant_1_let, constant_3_let, constant_5_let, sid_7_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_1(sid_7_let, sid_8_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_2(sid_8_let, constant_6_let, constant_7_let, constant_9_let, constant_11_let, sid_15_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_3(sid_15_let, sid_16_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_4(sid_16_let, constant_12_let, constant_13_let, linear_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  return 0;
}

