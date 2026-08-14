#include "model_data.h"

#include <cinttypes>
#include <cstddef>
#include <cstdint>

extern "C" {
#if defined(__has_include)
#if __has_include(<zephyr/kernel.h>)
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/timing/timing.h>
#else
#include <kernel.h>
#include <sys/printk.h>
#include <timing/timing.h>
#endif
#endif
}

#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#if defined(BAKENN_USE_CMSIS_NN)
#include <tensorflow/lite/micro/kernels/fully_connected.h>
namespace tflite {
TfLiteRegistration Register_CONV_2D_CMSIS_NN();
}
#endif
#include <tensorflow/lite/schema/schema_generated.h>

namespace {

// An 8 KiB probe reported 564 used bytes, but this 2021 TFLM allocator could
// not re-create the graph in used+16 bytes: the smaller arena changed tail
constexpr std::size_t kWarmupRuns = 8u;
constexpr std::size_t kMeasuredRuns = 101u;

alignas(16) std::uint8_t tensor_arena[2048u];
std::uint64_t measured_cycles[kMeasuredRuns];

void SortCycles(std::uint64_t *values, std::size_t count) {
  for (std::size_t index = 1u; index < count; ++index) {
    const std::uint64_t value = values[index];
    std::size_t position = index;
    while (position > 0u && values[position - 1u] > value) {
      values[position] = values[position - 1u];
      --position;
    }
    values[position] = value;
  }
}

}  // namespace

int main() {
  static tflite::MicroErrorReporter error_reporter;
#if defined(BAKENN_USE_CMSIS_NN)
  printk("TFLM_BACKEND=CMSIS_NN\n");
#else
  printk("TFLM_BACKEND=REFERENCE\n");
#endif
  const tflite::Model *model = tflite::GetModel(tflm_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printk("TFLM_ERROR schema=%d expected=%d\n", model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  static tflite::MicroMutableOpResolver<2> resolver;
#if defined(BAKENN_USE_CMSIS_NN)
  if (resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8()) !=
      kTfLiteOk) {
    printk("TFLM_ERROR cmsis_nn_fc_resolver\n");
    return 1;
  }
#else
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    printk("TFLM_ERROR resolver\n");
    return 1;
  }
#endif
#if defined(BAKENN_USE_CMSIS_NN)
  if (resolver.AddConv2D(tflite::Register_CONV_2D_CMSIS_NN()) != kTfLiteOk) {
    printk("TFLM_ERROR cmsis_nn_conv_resolver\n");
    return 1;
  }
#else
  if (resolver.AddConv2D() != kTfLiteOk) {
    printk("TFLM_ERROR conv_resolver\n");
    return 1;
  }
#endif
  static tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, tflm_model_arena_size, &error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printk("TFLM_ERROR allocate arena_bytes=%u\n", tflm_model_arena_size);
    return 1;
  }

  TfLiteTensor *input = interpreter.input(0);
  TfLiteTensor *output = interpreter.output(0);
  for (std::size_t index = 0u; index < tflm_model_input_count; ++index) {
    input->data.int8[index] = static_cast<std::int8_t>(input->params.zero_point);
  }

  timing_init();
  timing_start();
  timing_t start = timing_counter_get();
  TfLiteStatus status = interpreter.Invoke();
  timing_t end = timing_counter_get();
  const std::uint64_t first_cycles = timing_cycles_get(&start, &end);
  if (status != kTfLiteOk) {
    printk("TFLM_ERROR first_invoke\n");
    return 1;
  }

  for (std::size_t run = 0u; run < kWarmupRuns; ++run) {
    if (interpreter.Invoke() != kTfLiteOk) {
      printk("TFLM_ERROR warmup\n");
      return 1;
    }
  }
  for (std::size_t run = 0u; run < kMeasuredRuns; ++run) {
    start = timing_counter_get();
    status = interpreter.Invoke();
    end = timing_counter_get();
    if (status != kTfLiteOk) {
      printk("TFLM_ERROR invoke run=%u\n", static_cast<unsigned>(run));
      return 1;
    }
    measured_cycles[run] = timing_cycles_get(&start, &end);
  }
  timing_stop();
  SortCycles(measured_cycles, kMeasuredRuns);

  std::size_t stack_unused = 0u;
  const int stack_status = k_thread_stack_space_get(k_current_get(), &stack_unused);
  printk("TFLM target=nrf52840dk runs=%u first_cycles=%" PRIu64
         " median_cycles=%" PRIu64 " p95_cycles=%" PRIu64
         " tensor_arena_bytes=%u tensor_arena_used_bytes=%u"
         " stack_unused_bytes=%zu stack_status=%d\n",
         static_cast<unsigned>(kMeasuredRuns), first_cycles, measured_cycles[50],
         measured_cycles[95], tflm_model_arena_size,
         static_cast<unsigned>(interpreter.arena_used_bytes()), stack_unused, stack_status);
  std::uint32_t output_checksum = 2166136261u;
  for (std::size_t index = 0u; index < tflm_model_output_count; ++index) {
    output_checksum ^= static_cast<std::uint8_t>(output->data.int8[index]);
    output_checksum *= 16777619u;
  }
  printk("TFLM_OUTPUT_FNV1A=0x%08x first", output_checksum);
  const std::size_t preview = tflm_model_output_count < 8u ? tflm_model_output_count : 8u;
  for (std::size_t index = 0u; index < preview; ++index) {
    printk(" %d", static_cast<int>(output->data.int8[index]));
  }
  printk("\n");
  return 0;
}
