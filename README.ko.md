# BakeNN

[English](README.md) | 한국어

[![CI](https://github.com/scienthoon/bakeNN/actions/workflows/ci.yml/badge.svg)](https://github.com/scienthoon/bakeNN/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

BakeNN은 이미 학습된 FP32 PyTorch 모델과 대표 캘리브레이션 샘플을,
모델이 고정된 MCU 펌웨어를 위한 모델 특화 독립 실행형 C11 라이브러리로
컴파일합니다. 생성된 라이브러리는 힙을 사용하지 않으며, 타깃에서는
TFLite, FlatBuffers 또는 인터프리터가 필요하지 않습니다.

## 설치

```bash
python -m pip install "bakenn[torch]"
```

Python 3.10--3.13을 지원합니다. `torch`는 PyTorch 프런트엔드에만
필요하며, 생성된 펌웨어에는 Python이나 프레임워크 의존성이 없습니다.

선택 사항인 호스트 전용 완전 양자화 TFLite 임포터를 사용하려면 다음과
같이 설치합니다.

```bash
python -m pip install "bakenn[tflite]"
```

BakeNN 임포터의 차등 테스트에서 사용하는 선택적 LiteRT 참조 오라클은
`bakenn[tflite,tflite-verify]`로 설치할 수 있으며, 모델 컴파일에는
필요하지 않습니다.

이 임포터는 근삿값을 허용하지 않고 엄격하게 동작합니다. 특히 TFLite
AveragePool 노드는 LiteRT 내장 참조 커널이 사용하는 버전 지정 원시 코드
반올림 프로필을 유지합니다. BakeNN은 이를 자체 중앙 정렬 AveragePool
산술로 재해석하지 않습니다. 지원되는 임포트는 LiteRT, Python 정수 참조,
생성된 C를 대상으로 바이트 단위로 검증됩니다.

## 10줄 빠른 시작

```python
import bakenn

model.eval()
compiled = bakenn.compile_torch_ptq(
    model, example_input, calibration_samples,
    "build/classifier", name="classifier",
)
input_q = bakenn.quantize_input(compiled.plan, input_fp32_nhwc)
output_q = bakenn.run_reference(compiled.plan, input_q)
print(compiled.memory_report.to_text())
```

`example_input`은 배치 크기 1인 입력 형상을 고정합니다. 캘리브레이션
샘플은 INT8 스케일과 영점을 선택하는 데 쓰이는 FP32 활성화 범위를
측정합니다. 그 뒤 BakeNN은 정수 그래프를 검증하고 배포 라이브러리를
생성합니다.

`compiled.calibration_report`에는 캘리브레이션된 각 에지에 대해 소비한
샘플 수, FP32 범위, 선택된 qparams가 기록됩니다. 별도의 계층별
FP32-대-역양자화-INT8 오차 보고서(최댓값, 평균, RMS, INT8 끝점 개수)는
`compiled.verify_accuracy(validation_samples)`를 실행해 확인할 수
있습니다. 이 보고서는 PTQ 정확도를 진단합니다. 생성된 C는 이와
독립적으로 정수 참조와 바이트 단위로 일치해야 합니다.

## 생성 산출물

각 컴파일 디렉터리에는 검사 가능하고 결정론적인 다음 산출물이
들어 있습니다.

```text
bknn_classifier.h                 public C ABI, shapes and I/O qparams
bknn_classifier.c                 fixed execution order and static offsets
bknn_classifier_weights.c/.h      quantized constants
bknn_classifier_kernels.c/.h      only the selected kernels
bknn_classifier_manifest.json     graph, qparams, kernel IDs and provenance
bknn_classifier_memory.txt/.json  Flash/SRAM planning report
```

타깃 오버레이는 선택된 고정 버전 CMSIS-NN 또는 ESP-NN 소스 클로저와
해당 라이선스만 추가로 복사합니다. 생성된 라이브러리는 호출자가
소유한 입력·출력·아레나 버퍼를 사용하며 힙을 할당하지 않습니다.

v1 산출물에는 C ABI 버전 1, 매니페스트 스키마 버전 4, 그리고 버전이
지정된 수치 프로필 ID가 선언됩니다. 모델이나 컴파일러가 변경되면 전체
산출물 세트를 다시 생성하고 검증해야 합니다. 서로 다른 컴파일에서 나온
헤더, 소스, 매니페스트 또는 패킹 상수를 섞지 마십시오. 스키마 v4
매니페스트에는 생성 파일 목록과 매니페스트 페이로드 자체의 정규
다이제스트가 모두 포함됩니다.

## 모델 고정형 펌웨어에 BakeNN을 사용하는 이유

- **타깃 인터프리터 불필요:** 생성된 이식 가능한 C11을 새 MCU의 C
  컴파일러로 컴파일하거나, 검증된 타깃 커널 오버레이를 선택할 수
  있습니다.
- **정적 모델 특화:** 형상, 패딩, qparams, 고정소수점 승수, 메모리
  오프셋, 실행 순서가 상수이므로 융합, 버퍼 재사용, 패킹된 가중치,
  좁은 범위의 벤더 커널 호출이 가능합니다.
- **컴파일 시점 메모리 게이트:** 플래시/SRAM에 기록하기 전에 활성화
  아레나, 스크래치, 상수, 정렬 요구량을 알 수 있습니다. CI는 선언된
  Flash/SRAM 예산을 넘는 모델을 거부할 수 있습니다.
- **감사 가능한 펌웨어:** 최종 C 호출, 가중치, 커널 선택, 매니페스트를
  통해 FlatBuffer, 리졸버, 인터프리터, 런타임 텐서 플래너를 재구성하지
  않고도 모델 구조를 확인할 수 있습니다.
- **실패 폐쇄형 변환:** 지원하지 않는 연산, 호환되지 않는 qparams,
  안전하지 않은 누산기 범위, 예산 위반은 호스트에서 실패합니다.
  타깃 측 부동소수점 폴백은 없습니다.

트레이드오프는 의도적입니다. BakeNN v1은 정적 배치 크기 1, 고정 형상,
하나의 공개 입력/출력을 사용하고 TFLM보다 적은 연산자를 지원합니다.
이는 연결된 모델이 바뀔 때 펌웨어를 다시 빌드하는 제품을 위한
제약이며, 동적 모델 런타임을 제공하려는 시도가 아닙니다. 호환성 정책은
[STABILITY.md](STABILITY.md)를 참조하십시오.

각 시스템이 더 유리한 경우를 포함해 TFLM 및 Edge Impulse EON과
출처를 명시하여 비교한 내용은
[docs/COMPARISON.md](docs/COMPARISON.md)를 참조하십시오.

## 물리 벤치마크 요약

다음은 범위가 한정된 물리 측정값이며, BakeNN이 모든 모델이나
MCU에서 더 우수하다는 주장은 아닙니다. 비교한 모든 출력 바이트는
동일했습니다. 전체 프로토콜, 해시, 원시 UART, 제한 사항은
`benchmarks/` 아래에 체크인되어 있습니다.

물리 측정값과 보드 없는 크로스 빌드는 별도로 색인되어 있습니다.

- [물리 보드 근거](benchmarks/physical/README.md)
- [크로스 빌드/툴체인 근거](benchmarks/cross_build/README.md)

### 학습된 MNIST 전체 모델 데모

체크인된 MNIST 근거에는 4 epoch FP32 체크포인트, 클래스별로 균형 잡힌
캘리브레이션 이미지 160개, 생성된 독립 실행형 C, 물리 보드 이미지
100개 말뭉치가 고정되어 있습니다. FP32 정확도는 96.92%이고, 생성된 C의
INT8 정확도는 테스트 이미지 10,000개 전체에서 97.02%이며, Python
INT8과 생성된 C 간 바이트 불일치는 0개입니다.
[근거 매니페스트](examples/mnist/evidence/mnist_evidence.json)와
[재현 지침](examples/mnist/evidence/README.md)을 참조하십시오.

같은 고정 전체 모델은 160 MHz의 물리 원본 ESP32에서도 측정했습니다.
시간을 측정한 101회 호출에서 중앙값 3,165,624 cycle(19.785150 ms)을
기록했습니다. 이어진 이미지 100개 실행에서는 99개를 올바르게
분류했고, 예상 INT8 출력 바이트 1,000개가 모두 일치했습니다. 앱
바이너리는 79,500-byte 내장 검증 말뭉치를 포함해 236,208 B였습니다.
BakeNN 모델 구성 요소 자체는 Flash 8,871 B를 차지했고 3,920-byte
활성화 아레나를 사용했습니다.
[물리 결과, 원시 UART 및 크기 근거](benchmarks/esp32/results/mnist_trained_esp32.md)를
참조하십시오. 이전
[크로스 빌드 기록](benchmarks/cross_build/results/mnist_esp32_cross_build.json)은
두 번째 물리 측정값이 아니라 툴체인 근거로 별도 색인되어 있습니다.

학습된 모델에는 Apache TVM 0.16.0 AOT+USMP+CMSIS-NN과 비교한 동일
그래프 Cortex-M4 결과도 있습니다. 생성된 두 C 경로 모두 출력 바이트
1,000/1,000개가 일치했습니다. BakeNN ELF는 Flash 12,088 B를
연결했고, microTVM은 17,456 B를 연결했으며, 계획된 작업 공간은 둘 다
4,064 B로 동일했습니다. 이는 보드 없는 섹션 크기이며 지연 시간은
아닙니다. [microTVM 근거](benchmarks/microtvm_compare/README.md)를
참조하십시오.

### 원본 ESP32: 학습된 MobileNetV2-0.25

동일한 1 epoch CIFAR-10 체크포인트, 캘리브레이션 세트, 고정 INT8
그래프, 실수 0에 해당하는 입력, 240 MHz 클록, 워밍업 8회, 측정 실행
101회를 사용했습니다.

| 경로 | 중앙 지연 시간 | 앱 바이너리 | 연결된 DRAM |
|---|---:|---:|---:|
| BakeNN portable C | 285.546 ms | 459,088 B | 31,900 B |
| BakeNN + ESP-NN | **97.685 ms** | **465,296 B** | **31,900 B** |
| TFLM + ESP-NN | 98.891 ms | 665,504 B | 95,332 B |

이 산출물에서 BakeNN의 ESP-NN 직접 로워링은 portable C보다 2.92x
빨랐고 TFLM+ESP-NN보다 지연 시간이 1.22% 낮았습니다. TFLM과 비교하면
앱 바이너리는 30.1%, 연결된 DRAM은 66.5% 줄었습니다.
[전체 ESP32 비교](benchmarks/esp32/results/mobilenet_v2_025_cifar10_esp32_tflm_espnn.md)를
참조하십시오.

### nRF52840: 직접 CMSIS-NN 대 TFLM

이 문서에서 MCU 비교 대상은 모바일/Linux LiteRT 런타임이 아니라
TensorFlow Lite for Microcontrollers(TFLM)입니다.

다음 측정값은 IoT-LAB nRF52840DK(Cortex-M4, 64 MHz)에서 동일하게
고정된 `32 -> 16 -> 4` INT8 완전 연결 워크로드를 실행한 결과입니다.
동일한 qparams, 가중치, 바이어스, 입력 바이트, 출력 의미 체계를
사용했습니다. 네 빌드 모두 같은 출력 바이트와 FNV-1a 체크섬을
생성했습니다.

| 빌드 | 중앙 cycle 수 | Flash (text+data) | SRAM (data+bss) |
|---|---:|---:|---:|
| BakeNN direct CMSIS-NN FC | 3,786 | 20,920 B | 8,540 B |
| TFLM + CMSIS-NN FC | 5,418 | 69,640 B | 11,040 B |
| BakeNN portable FC | 8,706 | 20,764 B | 8,540 B |
| TFLM reference FC | 9,342 | 63,176 B | 11,008 B |

이 일치 조건의 FC 워크로드에서 BakeNN의 직접 CMSIS-NN 경로는 같은
CMSIS-NN FC 커널 계열을 쓴 TFLM보다 **cycle 수가 30.1% 적고**,
**연결된 Flash가 70.0% 적으며**, **연결된 정적 SRAM이 22.6% 적었습니다**.
모델 아레나는 16 B였고, TFLM은 1,024 B를 예약하고 580 B를 사용했다고
보고했습니다.

별도의 정적 `1x4x4x1 -> 1x4x4x2` Conv2D 측정에서는 BakeNN portable
C와 TFLM 참조 커널을 사용했습니다.

| 빌드 | 중앙 cycle 수 | Flash (text+data) | SRAM (data+bss) | 아레나 |
|---|---:|---:|---:|---:|
| BakeNN portable Conv2D | 24,610 | 20,332 B | 8,160 B | 0 B |
| TFLM reference Conv2D | 27,441 | 61,760 B | 10,624 B | 2,048 B reserved |

이 Conv2D 실행은 **cycle 수가 10.3% 적고**, **연결된 Flash가 67.1%
적으며**, **연결된 정적 SRAM이 23.2% 적었습니다**. 이는 CMSIS-NN
컨볼루션 비교가 아닙니다. 정확한 툴체인, 프로토콜, 해시, 제한 사항은
[전체 FC 보고서](benchmarks/tflm_compare/results/iotlab_447626_direct_cmsis_fc.md)와
[독립 실행형 Conv2D 보고서](benchmarks/tflm_compare/results/iotlab_447609_conv.md)를
참조하십시오. 이 측정값은 이 보드에서 고정된 워크로드에 대한
근거이며, 보편적인 순위가 아닙니다.

## 정적 AOT 설계의 작동 방식

TFLM은 양자화된 그래프를 FlatBuffer에 저장하고 MicroInterpreter,
연산자 리졸버, 런타임 텐서 메타데이터, 텐서 아레나를 통해 실행합니다.
BakeNN은 그래프, 텐서 수명, qparams, 고정소수점 파라미터, 커널 선택,
메모리 오프셋을 호스트에서 확정한 다음 모델별 직접 C 호출 그래프를
생성합니다.

MCU와 모델이 고정되고 모델이 바뀔 때 펌웨어를 다시 빌드하는 제품에서는
다음과 같은 구체적인 이점이 있습니다.

- **펌웨어에 모델 인터프리터나 FlatBuffer 파서가 없습니다.** 출력은
  모델과 선택된 커널만 포함하는 독립 실행형 C11 라이브러리입니다. 새
  32-bit MCU에서는 먼저 TFLM을 포팅하지 않아도 해당 MCU의 C11
  컴파일러로 portable 폴백을 빌드할 수 있습니다. 타깃별 최적화 커널은
  런타임 요구 사항이 아니라 선택적 오버레이입니다.
- **모델 특화 최적화.** 형상, 패딩, 채널, 승수, 버퍼 주소, 실행 순서가
  컴파일 시점 상수이므로 융합, 수명 기반 버퍼 재사용, 패킹된 가중치,
  좁은 범위의 1x1, 3x3, depthwise, Linear 커널을 사용할 수 있습니다.
  예산이 지정된 부분/전체 언롤링과 일반 Conv 내부/경계 루프 분리는
  문서화된 로드맵 항목이며, 현재의 성능 주장이 아닙니다.
- **컴파일 시점 리소스 강제.** 플래시하기 전에 상수 바이트, 활성화
  아레나, 스크래치, 정렬 요구량을 알 수 있습니다. Flash/SRAM 예산은
  보드에서 발견하는 대신 컴파일과 CI를 실패시킬 수 있습니다. 따라서
  `Flash <= 256 KiB`, `model SRAM <= 48 KiB`, 힙 심볼 없음,
  `alignment <= 16` 같은 제품 게이트를 기계적으로 강제할 수 있습니다.
  생성 모델 검사와 크로스 ELF 검사는 최종 애플리케이션의 스택 및 관련
  없는 전역 변수 측정을 대체하지 않습니다.
- **직접 벤더 커널 호출.** 지원되는 계층은 해당 커널 주변에 TFLM을
  유지하지 않고 CMSIS-NN 또는 ESP-NN을 직접 호출할 수 있습니다. 현재
  CMSIS-NN 어댑터는 ARMv7E-M DSP 타깃에서 FullyConnected, Conv2D,
  DepthwiseConv2D, AveragePool2D, MaxPool2D를 지원합니다. 선택 사용
  방식의 ESP-NN 백엔드는 ESP32-S3의 SIMD Conv2D, DepthwiseConv2D,
  채널별 FullyConnected, 풀링과 원본 ESP32의 Espressif 최적화
  Conv2D/DepthwiseConv2D 경로를 지원합니다.
- **검사 가능한 배포 산출물.** 생성된 C 함수 순서, 가중치, 정적
  오프셋, 커널 ID, qparams, 매니페스트를 직접 감사할 수 있습니다.
  펌웨어 검토 시 FlatBuffer, 인터프리터, op 리졸버, 텐서 플래너,
  런타임 설정 전반에서 모델을 재구성할 필요가 없어, 반복 가능한 산업
  및 안전 검토가 실질적으로 단순해집니다.
- **결정론적 실패.** 지원하지 않는 연산자, 안전하지 않은 누산기 범위,
  호환되지 않는 qparams, 메모리 예산 위반은 호스트 컴파일 중
  실패합니다. 타깃 측 부동소수점 폴백은 없습니다.

### 체크인된 보드 비교에서 더 쉬웠던 점

다음 차이는 이 저장소에서 FC 및 Conv2D 펌웨어를 빌드하고 실행하면서
관찰한 것이며, 가상의 API 비교가 아닙니다.

- **모델 패키징:** BakeNN 컴파일은 모델 C, 가중치, 선택된 커널,
  매니페스트, CMake 소스 목록을 함께 생성했습니다. TFLM 경로에는
  일치하는 `.tflite` FlatBuffer, `model_data.cc`로의 변환, 스키마
  호환성, 별도 C++ 실행기가 필요했습니다.
- **그래프 변경:** BakeNN은 검증된 그래프에서 실행 순서와 필요한
  커널을 도출했습니다. 최초의 최소 TFLM 실행기는 `AddFullyConnected()`만
  포함한 `MicroMutableOpResolver<1>`을 사용했습니다. Conv2D를 추가할
  때는 이를 `MicroMutableOpResolver<2>`로 바꾸고 `AddConv2D()`를
  등록해야 했습니다. 그렇지 않으면 모델 설정 과정에서 연산자를 찾을 수
  없었습니다. 이는 TFLM 커널 버그가 아니라 저희의 선택적 리졸버 구성
  실수였지만, 모델이 바뀌면 애플리케이션의 리졸버 용량과 등록을 서로
  맞게 유지해야 한다는 점을 보여 줍니다.

  ```text
  BakeNN graph contains Linear  -> select/emit a Linear kernel
  BakeNN graph contains Conv2D  -> select/emit a Conv2D kernel
  operation is unused          -> omit it from the artifact
  ```

- **연산자 버전 호환성:** 고정된 Zephyr TFLM은 이 픽스처에서 Conv2D
  연산자 버전 2를 허용했으며, 시도한 버전 1과 3은 거부했습니다.
  BakeNN은 펌웨어를 생성하기 전에 타입이 지정된 IR을 검증하므로
  FlatBuffer 연산자 버전 협상이 없습니다.
- **아레나 크기 결정:** BakeNN은 타깃 빌드 전에 필요한 16 B FC
  아레나와 0 B Conv2D 아레나를 생성했습니다. TFLM에는 호출자가
  선택하는 아레나와 이어지는 런타임 `AllocateTensors()`가
  필요했습니다. 체크인된 실행 전반에서 TFLM은 약 564--580 B를
  사용했다고 보고했지만, 보고된 양보다 조금만 더 예약해서는 그래프를
  안정적으로 재생성하지 못했습니다. 실행기에서는 대신 1,024--2,048 B를
  예약했습니다.
- **CMSIS-NN 통합:** BakeNN의 선택 기능은 고정된 FC 소스 클로저,
  헤더, 라이선스, 필요한 컴파일 정의를 산출물에 복사했습니다. 테스트한
  Zephyr TFLM 통합에는 별도의 CMSIS-NN, CMSIS-Core, TFLM 소스 루트가
  필요했습니다. 구형 래퍼는 `CMSIS/NN/Include`를 예상했으므로
  하네스에서 CMSIS-NN v4 레이아웃용 호환 링크를 만들었습니다. 참조
  래퍼와 CMSIS 래퍼는 충돌하는 등록 심볼을 내보냈으므로 하네스에서
  래퍼 심볼 이름을 로컬로 바꿨습니다. 또한 래퍼의 `ARM_MATH_SUCCESS`
  이름을 CMSIS-NN v4의 `ARM_CMSIS_NN_SUCCESS`에 매핑했습니다.
- **실패 위치:** BakeNN은 지원하지 않는 의미 체계와 안전하지 않은
  메모리를 호스트 컴파일 시점에 거부했습니다. TFLM 실행기에는 스키마
  버전, 리졸버 등록, 텐서 할당, `Invoke()` 실패에 대한 타깃/런타임
  검사도 필요했습니다.

실무적인 차이는 단순히 빌드 명령이 짧은 것보다 컸습니다. TFLM
경로에서는 개발자가 상호 호환되는 모델 형식, 연산자 버전, 리졸버,
런타임, 커널 라이브러리를 조합해야 했습니다. BakeNN은 고정 그래프를
분석해 필요한 실행 코드와 커널 세트를 생성했습니다.

트레이드오프는 의도적입니다. BakeNN은 현재 정적 배치 크기 1, 고정 형상
모델을 대상으로 하며 더 좁은 연산자 범위를 지원합니다. 하나의 펌웨어
런타임이 재컴파일 없이 서로 다른 모델 파일을 받아야 할 때는 더 넓은
연산자 범위를 제공하는 TFLM이 더 적합합니다.

정적 배치 크기 1과 단일 입력/단일 출력 모델 ABI는 동적 런타임에서
기다리는 임시 누락이 아니라 의도적인 제품 제약입니다. BakeNN은 칩과
모델이 고정되고 모델이 펌웨어에 링크되며, 모델 변경 시 일반적으로
펌웨어를 다시 빌드해 배포하는 프로덕션 MCU 펌웨어를 대상으로 합니다.
이 계약을 정적으로 유지함으로써 컴파일러는 배포 전에 텐서 형상, 실행
순서, 버퍼 수명, SRAM 오프셋, 커널 선택을 확정할 수 있습니다. 이것이
인터프리터 없이 결정론적 메모리 검사, 버퍼 재사용, 모델별 C 생성을
가능하게 합니다. 내부 그래프는 여전히 분기, 잔차 연결, 다중 입력
연산자를 포함할 수 있습니다. 단일 입력/단일 출력 제한은 공개 모델
ABI에 적용됩니다.

```text
PyTorch FP32 eval model + calibration samples
        -> torch.export FloatGraph
        -> PTQ QuantizedGraph + legalize/fuse
        -> verified static ExecutionPlan
        -> Python integer reference
        -> model.h + model.c + weights + portable C kernels
```

## 현재 수치 계약

- 활성화: 텐서별 affine `int8`, 범위 `[-128, 127]`
- 가중치: 출력 채널별 symmetric `int8`, 범위 `[-127, 127]`
- 바이어스와 누산기: `int32`, 컴파일 시점 오버플로 증명
- 재양자화: 버전 지정 Q31 이중 반올림 프로필(`bakenn.int8.v1`)
- 정확한 언더플로: 어떤 int32 누산기도 바꿀 수 없을 만큼 작은 비율은
  상수 Q31 결과 `(multiplier=0, shift=0)`로 생성
- 배포 스케일: 유한한 양의 IEEE-754 float32로 한 번 정규화
- 정적 형상 및 배치 크기 1만 지원
- 지원하지 않는 연산과 안전하지 않은 누산기는 컴파일 시점에 실패

현재 구현된 연산 계열은 다음과 같습니다.

- Conv2D(groups 포함), depthwise Conv2D, Conv1D(groups 포함),
  FullyConnected
- 정적으로 브로드캐스트되는 Add/Mul, Clamp/ReLU/ReLU6, Sigmoid,
  HardSigmoid, HardSwish, SiLU, 내부 Requantize
- AveragePool2D/MaxPool2D 및 AveragePool1D/MaxPool1D
- 명시적인 실수 0 Pad2D 및 공간/시간 ReduceMean
- Reshape, Flatten, Squeeze/Unsqueeze 뷰, 정적 Slice/Crop, Concatenate
- nearest 및 Q15 bilinear Resize2D, grouped ConvTranspose2D
- `bakenn.softmax_lut.q15.v1`을 사용하는 마지막 축 rank-two Softmax

이미지 ABI는 정규 NHWC이고 시퀀스 ABI는 정규 NLC입니다. PyTorch
프런트엔드는 NCHW/NCL을 허용하고 호스트에서 가중치와 활성화를
변환합니다. 현재 Pad는 상수인 실수 0만 지원합니다. ReduceMean은
`keepdim=True` 또는 축소된 NC 출력과 함께 NHWC 공간 축이나 NLC 시간
축을 지원합니다. Sigmoid는 고정 출력 qparams
`(scale=1/256, zero_point=-128)`를 사용합니다. 모든 형상은 계속
정적이며, 배치 크기는 1이고, 모델 입력과 출력은 각각 하나입니다.
Resize 출력 크기는 컴파일 중 고정됩니다. Bilinear resize는 버전이
지정된 `bakenn.int8.resize_bilinear.q15.v1` 좌표/반올림 프로필을
사용합니다. ConvTranspose2D는 출력 채널별 가중치와 함께 양의 groups,
정적 stride, dilation, 비대칭 padding, output padding을 지원합니다.
Slice/Crop 경계, 음수 인덱스, 양의 step은 호스트에서 정규화됩니다.
Add/Mul 브로드캐스트 차원은 정적으로 호환되어야 하며, 런타임
브로드캐스트 결정은 없습니다.

현재 모델 수준 호스트 게이트가 다루는 범위는 다음과 같습니다.

- 수정하지 않은 torchvision `mobilenet_v3_small` 및
  `mobilenet_v3_large` 그래프의 FP32 캡처, PTQ, 계획, C 산출물 생성
- 정적 32x32에서 수정하지 않은 torchvision `mobilenet_v2`의 FP32
  캡처, PTQ, 계획, 생성된 C 컴파일, 원시 INT8 바이트 단위 정확 실행
- EfficientNet-Lite 스타일 ReLU6 MBConv 분류기 및 더 어려운
  SiLU/SE-broadcast 프런트엔드 상위 집합인 torchvision EfficientNet-B0
- `keepdim=False` 전역 ReduceMean을 포함하여 수정하지 않은 torchvision
  MNASNet0.5의 FP32 캡처, PTQ, 계획, C 산출물 생성
- grouped ConvTranspose2D, nearest/bilinear resize 또는 정적 center
  crop을 사용하고 skip concatenation 및 정적 아레나 계획을 적용한
  소형 U-Net
- 소형 ResNet bottleneck, DenseNet 스타일 dense concat, Inception
  branch, SqueezeNet Fire, Conv1D-flatten 분류기, temporal residual,
  Softmax MLP 모델의 역양자화 정확도 및 바이트 단위로 정확한 생성 C
  테스트

소형 모델은 ASan/UBSan을 적용한 GCC 및 Clang으로 컴파일하며, C 출력은
Python INT8 참조와 바이트 단위로 정확히 일치합니다. 실데이터 학습
행렬은 1 epoch 학습 후 MNIST 모델 2개와 CIFAR-10 모델 4개를 다룹니다.
정확도와 메모리 결과는
[`examples/training_matrix/RESULTS.md`](examples/training_matrix/RESULTS.md)에
기록되어 있습니다. 이 호스트 행렬은 위에서 설명한 물리 nRF52840
근거와 별개입니다.

한 번의 호출로 실행하는 PyTorch PTQ 경로는 다음과 같습니다.

```python
import bakenn

model.eval()
compiled = bakenn.compile_torch_ptq(
    model,
    example_input,       # one static batch-one FP32 tensor
    calibration_data,    # representative FP32 tensors or batches
    "build/classifier",
    name="classifier",
)

input_q = bakenn.quantize_input(compiled.plan, input_fp32_nhwc)
output_q = bakenn.run_reference(compiled.plan, input_q)
```

모든 컴파일은 결정론적인 사람이 읽을 수 있는 메모리 보고서와 JSON
메모리 보고서도 생성합니다. 이 보고서는 선택된 커널, 활성화 수명, 물리
아레나 재사용, 백엔드 스크래치, 정확한 상수 페이로드를 노출하되 이를
전체 펌웨어 측정값인 것처럼 제시하지 않습니다.

```python
print(compiled.memory_report.to_text())

print(compiled.artifacts.memory_report_text)  # bknn_classifier_memory.txt
print(compiled.artifacts.memory_report_json)  # bknn_classifier_memory.json
```

컴파일 시점 보고서는 AOT 계획으로 알 수 있는 값과, 타깃 ELF/map이
필요한 최종 Flash/load 섹션, 타깃 분석 또는 물리 측정이 필요한 최대
스택/전체 펌웨어 SRAM을 구분합니다. 호출자 소유 입력 및 출력 버퍼는
모델 아레나와 별도로 보고됩니다.

실행 가능한 FP32 PyTorch -> PTQ -> ESP-NN -> 독립 실행형 ESP-IDF
흐름은 [ESP32-S3 엔드투엔드 데모](examples/esp32s3_end_to_end/README.md)를
참조하십시오. 학습된 대표 모델은
[MobileNetV2-0.25 CIFAR-10 예제](examples/mobilenet_v2_cifar10/README.md)를
참조하십시오. 이 예제는 전체 torchvision 토폴로지를 학습하고, FP32와
생성된 C의 INT8 정확도를 측정하고, Python/C 바이트를 검증하고, 같은
그래프를 ESP32-S3용으로 패키징합니다.
[기록된 1 epoch 결과](examples/mobilenet_v2_cifar10/RESULTS.md)는 테스트
이미지 10,000개 전체에서 FP32 21.10%, 생성된 C INT8 20.95%의 정확도를
측정했습니다. 이는 경쟁력 있는 학습 벤치마크가 아니라 엔드투엔드
파이프라인 결과입니다.

같은 파이프라인을 명시적이고 검사 가능한 단계로도 사용할 수 있습니다.

```python
from bakenn.frontends import capture_torch_export

float_graph = capture_torch_export(model, example_input)
qgraph = bakenn.quantize_float_graph(float_graph, calibration_data)
compiled = bakenn.compile(qgraph, "build/classifier")
```

P2 커널 선택은 명시적이고 재현 가능합니다. Portable C가 기본값입니다.
검증된 형상 특화 커널과 가중치 패킹을 결정론적인 기능 우선순위
방식으로 선택하려면 다음과 같이 지정합니다.

```python
compiled = bakenn.compile(
    qgraph,
    "build/classifier",
    backend_options=bakenn.CBackendOptions(
        kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY,
    ),
)
```

첫 일반 슬라이스에는 `optimized.linear_oi2.v1`과 홀수 출력용 변형인
`optimized.linear_oi2_tail.v1`, `optimized.conv2d_1x1_o2.v1`,
`optimized.depthwise_3x3_c2.v1`이 포함됩니다. 지원하지 않는 형상은
portable C로 폴백합니다. 폴백 대신 실패해야 하는 커버리지 감사에는
`REQUIRE_OPTIMIZED`를 사용할 수 있습니다. 산출물 매니페스트에는 모든
선택, 거부 이유, 패킹 레이아웃이 기록됩니다. 일반 후보는 호스트에서
검증된 특화 구현입니다. 최초 측정 타깃 결과는 아래에 별도로
기록되어 있습니다.

`cortex-m4` 프로필에는 실제 Arm DSP intrinsic 후보도 있습니다.

- `cortex_m4.linear_smlad.v1`
- `cortex_m4.conv2d_1x1_smlad.v1`
- `cortex_m4.depthwise_3x3_smlad.v1`
- `cortex_m4.conv2d_3x3_im2col_smlad.v1`
- `cortex_m4.global_average_pool2d_s8.v1`
- `cortex_m4.max_pool2d_2x2_s2.v1`

3x3 Conv 후보는 재사용 가능한 1 출력 픽셀 im2col 스크래치 영역을
사용합니다. Arm 크로스 ELF 테스트는 GNU Arm이 실제 `smlad` 명령을
생성하는지, 최종 ELF에 미해결 심볼, 힙 심볼, 부동소수점 런타임 심볼이
없는지 검증합니다. 패킹된 SMLAD 가중치는 Flash를 늘릴 수 있습니다.
타깃 적격성 자체는 성능 주장이 아닙니다. 측정된 nRF52840 결과는
체크인된 FC/Conv 워크로드만 다룹니다.

선택 사항인 직접 CMSIS-NN 소스 백엔드는 ARMv7E-M DSP 타깃에서
FullyConnected, Conv2D, DepthwiseConv2D, AveragePool2D, MaxPool2D를
지원합니다. 선택된 모델에 필요한 고정 CMSIS-NN v4 소스 클로저만
번들로 포함합니다. Conv와 Depthwise는 affine 입력/출력 오프셋, 출력
채널별 승수/shift 배열, 융합된 활성화 clamp를 CMSIS 래퍼에 직접
전달합니다. 타깃 버퍼 크기 공식과 AveragePool 스크래치는 호스트에서
계산되어 BakeNN의 공유 정적 SRAM 아레나와 매니페스트에 포함됩니다.
기능 검사는 레이아웃, groups/depth multiplier, 차원, stride, dilation,
padding을 다룹니다. AveragePool은 영점과 유효 윈도 개수로 CMSIS
반올림 결과가 `bakenn.int8.v1`과 바이트 단위로 정확히 일치함을 증명할
수 있을 때만 CMSIS를 사용합니다. 지원하지 않는 경우 폴백하거나
`REQUIRE_OPTIMIZED`에서는 실패합니다.

선택 사항인 ESP-NN 소스 백엔드는 두 번째 벤더 오버레이입니다. ESP-NN
1.2.6의 리비전
`c0876179f1cf4b4b9073b4f81cb65c8051ccb476`을 벤더링하고, 해당 식별자를
매니페스트에 기록하며, 고정된 타깃 소스 클로저, 헤더, 라이선스를
생성된 ESP-IDF 구성 요소에 복사합니다. 생성된 프로젝트를 빌드할 때 TFLM을
사용하거나 ESP 구성 요소를 다운로드하지 않습니다.

- `esp32s3`는 정확한 기능 조건자를 만족할 때 ESP-NN Conv2D,
  DepthwiseConv2D, 출력 채널별 FullyConnected, AveragePool2D,
  MaxPool2D를 선택합니다. 필요한 ESP-NN 스크래치와 안전한 FC 스테이징은
  BakeNN이 정적으로 계획한 하나의 스크래치 아레나에 포함됩니다.
- `esp32`는 Espressif의 최적화된 일반 Conv2D 및 DepthwiseConv2D
  구현을 선택합니다. 이 칩에서 ESP-NN은 FC와 풀링을 ANSI C로
  매핑하므로 BakeNN은 이 연산에 대해 자체 검증된 일반 커널을 의도적으로
  유지합니다.
- `esp32c3`에는 고정된 릴리스의 ESP-NN 구현이 없으므로 BakeNN의
  portable/일반 최적화 폴백을 유지합니다.

BakeNN은 ESP-NN의 TFLM 호환 이중 반올림 프로필을 고정하며
`CONFIG_NN_SKIP_NUDGE`를 절대 활성화하지 않습니다. 기능 검사는
지원하지 않는 dilation, depthwise geometry, alignment와 반올림이
`bakenn.int8.v1`과 바이트 단위로 정확하다고 증명할 수 없는
AveragePool 경우를 거부합니다. 그 뒤 `STATIC_PRIORITY`는 폴백하고,
`REQUIRE_OPTIMIZED`는 정확한 이유를 보고합니다. 호스트 테스트는 원본
ESP32 최적화 C를 실행해 BakeNN 정수 참조와 바이트 단위로 비교합니다.
ESP32-S3 래퍼는 공식 ESP-NN ANSI 오라클을 통해 호스트에서 검사하고,
실제 Xtensa 소스는 보드 없는 ESP-IDF CI에서 컴파일합니다. 실제 S3 SIMD
cycle, 캐시 동작, 에너지는 여전히 물리 ESP32-S3에서 측정해야 하며
여기서는 주장하지 않습니다.

타깃 선택은 선택 사항입니다. `portable32`가 기본값으로 유지됩니다.
ARM/RISC-V 프로필은 정확한 ABI/정렬/컴파일러 메타데이터를 추가하고,
ESP 프로필은 ESP-IDF 구성 요소/프로젝트를 생성할 수 있습니다.

```python
compiled = bakenn.compile(graph, "build/m4", target="cortex-m4")
report = bakenn.build_freestanding_elf(
    compiled.artifacts, "cortex-m4", "build/m4/cross"
)

esp_options = bakenn.CBackendOptions(
    kernel_policy=bakenn.KernelPolicy.STATIC_PRIORITY,
    enable_esp_nn=True,
    target=bakenn.ESP32_S3,
)
esp = bakenn.compile(
    graph,
    "build/s3",
    backend_options=esp_options,
    target="esp32s3",
)
project = bakenn.export_esp_idf_project(
    esp.artifacts, "esp32s3", "build/s3/project"
)
```

내장 타깃 ID는 `portable32`, `cortex-m0plus`, `cortex-m4`, `rv32imc`,
`esp32`, `esp32s3`, `esp32c3`입니다. ARM M0+/M4와 RV32IMC에는 독립
실행형 ELF/link-map/symbol-audit 경로가 있습니다. ESP-IDF 패키징과
보드 없는 빌드 CI가 제공되며, 여기에는 ESP32 및 ESP32-S3의 선택 사용
ESP-NN Conv/Depthwise 스모크 그래프와 ESP32-C3의 portable 폴백이
포함됩니다. 현재 물리 cycle 근거는
[`benchmarks/tflm_compare/results/iotlab_447626_direct_cmsis_fc.md`](benchmarks/tflm_compare/results/iotlab_447626_direct_cmsis_fc.md)의
nRF52840DK/Cortex-M4 벤치마크를 다룹니다. 원본 ESP32 MobileNetV2
비교는 [`benchmarks/esp32/results`](benchmarks/esp32/results)에
기록되어 있습니다. ESP32-S3 SIMD cycle과 전체 펌웨어 에너지는 아직
측정되지 않았습니다.
[타깃 계층 계약](docs/TARGETS.md)을 참조하십시오.

`STATIC_PRIORITY`는 기능 조건자와 선언된 우선순위에 따라 선택합니다.
더 낮은 지연 시간, Flash 또는 에너지를 보장하지 않습니다. `MEASURED`는
정규 워크로드, 타깃, 툴체인, 플래그가 정확히 일치하는 물리 비용 항목만
참조하고, 그렇지 않으면 portable C를 사용합니다. `AUTO`는
`STATIC_PRIORITY`의 사용 중단 예정 호환성 별칭이며, 가장 빠른 모드가
아닙니다. Portable이 기본값입니다. 구현된 각 계열의 호스트 스모크
비교는 다음 명령으로 실행할 수 있습니다.

```bash
PYTHONPATH=src python benchmarks/host_linear_compare.py --kernel linear
PYTHONPATH=src python benchmarks/host_linear_compare.py --kernel conv1x1
PYTHONPATH=src python benchmarks/host_linear_compare.py --kernel depthwise3x3
```

호스트 결과는 회귀 근거입니다. 별도의 nRF52840 및 원본 ESP32 결과가
현재의 물리 MCU/TFLM 성능 근거이며, 각 결과는 고정된 모델, 보드,
컴파일러, 프로토콜로 명시적으로 제한됩니다.

프레임워크 프런트엔드는 실제 `torch.export` API를 통해 캡처하고,
PyTorch는 사용할 때만 임포트합니다. 곧바로 BakeNN 소유의 불변 타입을
생성합니다. IR, 플래너, 참조 실행기, 백엔드는 PyTorch를 임포트하지
않습니다. 생성된 펌웨어는 PyTorch, TensorFlow, FlatBuffers,
인터프리터, C++, 동적 할당에 전혀 의존하지 않습니다.

선택 사항인 호스트 전용 TFLite 프런트엔드는 허용된 부분 집합을 즉시
BakeNN 소유 IR로 변환합니다.

```python
compiled = bakenn.compile_tflite(
    "classifier.tflite", "build/classifier", name="classifier"
)
```

현재 TFLite 스키마 버전 3, 정확히 하나인 서브그래프와 공개 입력/출력,
정적 배치 크기 1인 rank-2/3/4 INT8 활성화를 허용하며 다음 builtin
버전을 지원합니다. `CONV_2D` v3, `DEPTHWISE_CONV_2D` v3,
바이어스가 있는 `FULLY_CONNECTED` v4 또는 바이어스가 선택 사항인 v6, `ADD`
v2, `AVERAGE_POOL_2D`/`MAX_POOL_2D` v2, `RESHAPE` v1,
`PAD`/`PADV2` v2입니다. 융합 활성화는 `NONE`, `RELU`, `RELU6`으로
제한됩니다. 지원하지 않는 연산자/버전, 동적·가변·희소 텐서, 외부 버퍼,
커스텀 op, grouped Conv2D, 호환되지 않는 qparams는 호스트 임포트 중
실패합니다. 가중치는 예상된 양자화 축에서 영점 0인 symmetric INT8이어야
합니다. INT32 바이어스 스케일은 입력 스케일과 가중치 스케일의 곱과
같아야 하며, `PADV2`는 affine 실수 0을 인코딩해야 합니다. 선택적 파서
패키지는 생성된 타깃 코드에 TFLite 런타임이나 FlatBuffers 의존성을
추가하지 않습니다. v1 게이트는 일치하는 모델을 LiteRT 내장 참조
리졸버, 임포트한 BakeNN 정수 실행기, 컴파일된 C로 각각 실행하고 출력이
바이트 단위로 정확히 일치할 것을 요구합니다. XNNPACK 같은 호스트
delegate는 1 LSB tie 규칙이 다를 수 있으므로 이 펌웨어 정확성
오라클로 사용하지 않습니다.

캘리브레이션은 배열, 텐서, iterable, 단일 입력
`TensorDataset`/`DataLoader` 배치를 허용합니다. 샘플은 스트리밍 중
스냅샷되므로 로더가 backing buffer를 재사용해도 안전합니다. 다중 필드
`(input, target)` 배치는 모호하므로 거부합니다. 모델 입력만
전달하십시오. complex, object, string, boolean, empty, non-finite
캘리브레이션 데이터는 실패 폐쇄 방식으로 거부됩니다.

Eval BatchNorm은 호스트에서 폴딩됩니다. Dropout과 Identity는 제거되고,
일반적인 안전한 in-place ReLU/Add 표면은 mutation/fan-out 검사 후에만
불변 SSA로 정규화됩니다. 바이어스가 0이 아닌 모든 0 가중치 채널은
생성된 C와 동일한 Q31 산술에서 정확한 INT8 출력이 증명되는, 선언된
상수 채널 스케일 정책을 사용합니다.

의존성이 적은 테스트 스위트는 다음 명령으로 실행합니다.

```bash
PYTHONPATH=src python -m pytest -q
```

엔드투엔드 테스트는 C를 생성하고, 호스트 C 컴파일러로 컴파일해 실행한
뒤, 독립적인 Python 정수 참조와 출력을 바이트 단위로 비교합니다.

CI는 Python 3.10, 3.11, 3.12, 3.13과 GCC 및 Clang 조합으로 의존성이
적은 테스트 스위트를 실행합니다. 프레임워크 행렬은 추가로 Python
3.10에서 Torch 2.9/torchvision 0.24, Python 3.13에서 Torch
2.10/torchvision 0.25를 실행하며, wheel 빌드와 깨끗한 설치 검사도
포함합니다. 이 조합은 v1에서 지원하는 프레임워크 양 끝점입니다. 이전
Torch export IR 방언은 근사 해석하지 않고 거부합니다.

생성된 모델은 호출자 소유의 원시 아레나 포인터를 노출합니다. 보고된
`*_ARENA_SIZE` 바이트를 `*_ARENA_ALIGNMENT`에 맞춰 정확히
할당하십시오. 보고된 크기가 0이면 `NULL`을 전달하십시오. 입력, 출력,
아레나 메모리는 서로 겹치면 안 됩니다. 입력 및 출력 scale/zero-point
매크로는 공개 모델 헤더에 생성됩니다. 헤더에는 I/O rank, 차원, 바이트
수, 정규 레이아웃도 생성됩니다. 헤더의 `restrict` ABI는 입력, 출력,
아레나가 서로 분리되어 있을 것을 요구합니다.

[`benchmarks/tflm_compare`](benchmarks/tflm_compare/README.md)의 오프라인
비교 계약은 최종 ELF Flash, 전체 최대 SRAM, 초기화/추론 cycle, 출력
오차를 기록합니다. 체크인된 템플릿은 여전히 측정되지 않은 것으로
명시되어 있으며, 측정된 FC 및 Conv 보고서는 해당 디렉터리에서 링크되어
있습니다.

**신경망을 펌웨어에 구워 넣으세요.**

사용자용 Python 패키지는 `bakenn`입니다. 생성된 펌웨어 심볼은 짧은
`bknn_` 접두사와 `BKNN_` 매크로를 사용합니다. 버전이 지정된 수치
프로필은 읽기 쉬운 `bakenn.*` 네임스페이스를 유지합니다.

정확한 지원 계약은 [P0 청사진](docs/P0_BLUEPRINT.md)을, 선택,
고정소수점, 패킹 계약은
[P2 커널 아키텍처](docs/P2_KERNEL_ARCHITECTURE.md)를 참조하십시오.
기준선 이후 작업과 승인 게이트의 순서는
[로드맵](docs/ROADMAP.md)에 기록되어 있습니다.

릴리스 및 프로젝트 정책:
[안정성](STABILITY.md),
[결과 재현](REPRODUCING.md),
[제3자 고지](THIRD_PARTY_NOTICES.md),
[변경 기록](CHANGELOG.md),
[기여](CONTRIBUTING.md),
[보안](SECURITY.md).
