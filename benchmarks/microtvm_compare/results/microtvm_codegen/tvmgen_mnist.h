#ifndef TVMGEN_MNIST_H_
#define TVMGEN_MNIST_H_
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Input tensor value size (in bytes) for TVM module "mnist" 
 */
#define TVMGEN_MNIST_VALUE_SIZE 784
/*!
 * \brief Output tensor linear size (in bytes) for TVM module "mnist" 
 */
#define TVMGEN_MNIST_LINEAR_SIZE 10
/*!
 * \brief Input tensor pointers for TVM module "mnist" 
 */
struct tvmgen_mnist_inputs {
  void* value;
};

/*!
 * \brief Output tensor pointers for TVM module "mnist" 
 */
struct tvmgen_mnist_outputs {
  void* linear;
};

/*!
 * \brief entrypoint function for TVM module "mnist"
 * \param inputs Input tensors for the module 
 * \param outputs Output tensors for the module 
 */
int32_t tvmgen_mnist_run(
  struct tvmgen_mnist_inputs* inputs,
  struct tvmgen_mnist_outputs* outputs
);
/*!
 * \brief Workspace size for TVM module "mnist" 
 */
#define TVMGEN_MNIST_WORKSPACE_SIZE 4064

#ifdef __cplusplus
}
#endif

#endif // TVMGEN_MNIST_H_
