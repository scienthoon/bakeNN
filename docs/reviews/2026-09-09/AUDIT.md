# BakeNN 코드 감사 및 수정 기록 — 2026-09-09

## 범위와 결론

기준 커밋: `d2fb3aa9ffcc9cc2c1120e450f3b884732ad2ca5` (`main`, BakeNN 1.0.0).
시작 당시 추적 중인 코드 변경은 없었으며, 기존 `docs/reviews/2026-09-08/`와
`docs/submission/final/`의 사용자 파일은 그대로 보존했다.

PyTorch/TFLite 입력에서 INT8 IR, 양자화, 그래프 검증, 실행 계획, 산출물,
타깃 프로젝트와 CI까지 구조를 검토했다. 실제 결함은 수정 전 재현이나 실패하는
회귀 테스트로 확인한 뒤 수정했다. 기존 전체 테스트 355개가 통과하는 상태에서도
새 경계 조건과 원본 프레임워크 비교에서 오류가 발견됐다.

Trusted Access 활성화 후 C 백엔드·vendor 커널·공간 연산 lowering도 다시 검토했다.
중단 전에 보관했던 변경은 그대로 채택하지 않고 재현 fixture, sanitizer 및 경계의
양성 대조 사례로 다시 확인했다. 이 보고서는 유한한 코드 검토와 테스트 결과이며
모든 입력·환경에서 버그가 없다는 보증은 아니다.

### 직접 재검토에 따른 정정

이전 검토에서 CMSIS의 fully-padded depthwise 제한을 과도하다고 판단해 제거한 것은
잘못이었다. 생성 코드가 호출하는 wrapper는 Cortex-M4에서 DSP im2col 경로를 고르지만,
호스트 빌드는 `ARM_MATH_DSP`가 없어 reference 경로로 내려간다. 따라서 호스트의
sanitizer 통과는 그 DSP 경로가 안전하다는 증거가 아니었다. 고정된 vendor 소스의
buffer-fill 블록을 그대로 추출해 실행하자 2바이트 scratch에 4바이트를 쓰는 ASan
오류가 재현됐다. 제한을 복원했고 정상 경계와 수평 padding은 계속 지원한다.

ESP32 depthwise에도 별도의 int16 좌표 잘림을 재현했다. 출력 높이 32769에서
depth multiplier 1·2 각각 마지막 1·2바이트가 기대값 `1` 대신 `0`이었다.
ESP32-S3 dispatcher의 동일한 generic C 분기까지 선택 조건을 수정했다.

## 구조에 대한 평가

| 단계 | 역할 | 이번 검토의 핵심 |
|---|---|---|
| `frontends/torch_export`, `frontends/tflite` | 원본 모델 의미를 BakeNN 표현으로 변환 | 공유 저장소, 생략 옵션, 이름 충돌, float32 반올림 |
| `quantization` | 캘리브레이션, qparams, 정수 계약, 정확도 비교 | 반복 레이어 보존, 채널 순서, NumPy 정수의 중간 오버플로 |
| `ir`, `passes` | SSA·dtype·shape·수치 검증과 그래프 최적화 | 차원/축/32비트 경계, 부적합 qparams의 조기 거부 |
| `plan`, `reference`, `backend` | 실행 계획, 정수 기준 실행, C 생성 | resize 좌표, vendor ABI/산술 경계, SRAM-aware kernel fallback |
| `artifacts`, `reporting`, `targets` | 무결성 검사, 메모리 보고, 펌웨어 패키징 | 기존 파일 보존, 실패 시 부분 산출물, 측정·빌드 옵션 일치 |
| 패키징·CI·문서 | 설치와 재현, 호환성 약속 | 새 회귀 검사의 CI 연결, 1.x 정책 및 checksum 신뢰 경계 |

프레임워크 의존성을 호스트 입력 단계에 한정하고 정수 계약·C ABI·manifest를
버전 관리하는 분리는 잘 되어 있다. 반면 Python 정수 기준 실행과 생성 C가 동일한
잘못된 plan을 공유하면 둘이 일치해도 원본 모델과는 다를 수 있다. 독립적인
PyTorch/LiteRT 비교가 반드시 함께 있어야 한다.

## 완료한 수정

우선순위 P1은 잘못된 출력, 메모리 접근, 사용자 파일 손실 가능성 때문에 우선 수정한
항목이다. P2는 호환성·오류 처리·검증 신뢰도를 개선한 항목이다. CVSS 점수는 산정하지 않았다.

| ID | 우선순위 | 재현된 문제 | 수정과 검증 |
|---|---|---|---|
| F01 | P1 | TFLite 이름 중복 해소용 이름이 다른 원본 이름과 다시 충돌해 가중치를 합침. LiteRT `[8,16,24,32]`, 기존 BakeNN/C `[16,32,48,64]` | 모든 tensor 이름이 실제로 고유해질 때까지 검사. LiteRT/Python/C 바이트 일치 확인 |
| F02 | P2 | 자동 생성한 zero bias 이름이 뒤에서 등장할 원본 tensor 이름과 충돌 | 전체 원본 이름을 미리 예약하고 합성 이름도 함께 관리 |
| F03 | P1 | TFLite RELU6 경계의 float64 계산 때문에 1 LSB 차이. LiteRT `-125`, BakeNN/C `-126` | float32 나눗셈 후 반올림하고 큰 몫은 먼저 포화 처리. 세 실행 경로 비교 |
| F04 | P2 | shape 상수 입력이 있는 정상 RESHAPE도 options가 없으면 거부 | 상수 입력으로 shape가 완전히 정해지는 인코딩 허용. shape tensor는 rank 1 검사 |
| F05 | P2 | FC v6에서 bias가 있는 노드를 거부하여 bias 유무가 섞인 모델을 읽지 못함 | v4는 bias 필수, v6는 bias 선택으로 처리. LiteRT/C 확인 및 지원 문서 갱신 |
| F06 | P1 | MaxPool1d의 dilation/ceil_mode를 무시. dilation=2 입력 `[1,100,3,4]`에서 PyTorch `3`, BakeNN FP32 `100` | 아직 구현되지 않은 의미는 명시적으로 거부 |
| F07 | P1 | 공유 view/slice/dropout을 통한 in-place 변경이 다른 소비자에게 반영되지 않음. PyTorch `[0,12]`, BakeNN `[-4,12]` | alias 경로 전체의 공유 여부 검사. 안전한 비공유 경로의 정상 동작도 테스트 |
| F08 | P1 | ReduceMean의 NHWC/NLC rank 불일치가 검증을 통과해 C에서 입력 밖을 읽음. 6바이트 입력을 18바이트처럼 취급한 ASan 재현 | layout별 rank와 축의 원래 범위를 검사한 뒤 축 정규화. 잘못된 모델은 파일 생성 전에 거부 |
| F09 | P1 | Pool1D·ConvTranspose2D가 32비트 타깃 ABI 범위를 넘는 매개변수를 허용 | kernel/stride/padding/dilation 등 해당 매개변수의 범위 검증 추가 |
| F10 | P2 | Conv1D·ConvTranspose2D bias scale 곱의 float32 underflow/overflow가 문맥 없는 ValueError로 노출 | 연산 이름·원인이 포함된 GraphValidationError로 일관되게 실패 |
| F11 | P2 | Slice가 지원하지 않는 per-axis 활성화 qparams를 허용 | 다른 활성화 연산처럼 per-tensor qparams 요구 |
| F12 | P1 | legacy Sequential PTQ의 `named_children()`가 동일 레이어의 반복 호출을 제거 | 등록된 실행 슬롯 순서대로 변환. 반복 Linear와 ReLU 보존 |
| F13 | P1 | NCHW↔NCL reshape에서 singleton 차원 존재만 확인하여 채널 재해석을 허용 | 채널 수와 공간/시간 원소 수 보존까지 검증. 정상 squeeze/unsqueeze는 계속 허용 |
| F14 | P2 | Flatten 정확도 보고서가 NCHW/NCL 순서와 channel-last 순서를 직접 비교하여 약 2.0의 거짓 오차를 보고 | flatten 직전 배치 순서에 맞춰 FP32 비교값 재배열 |
| F15 | P1 | NumPy int32 scalar가 고정소수점 중간 곱·절댓값에서도 int32를 유지해 오버플로 | Integral을 Python int로 정규화. 비정수와 비지원 shift는 연산 전에 거부 |
| F16 | P1 | ESP-IDF/Zephyr export가 기존 CMakeLists·main.c를 덮어쓰고 실패 시 부분 프로젝트를 남김 | 비어 있거나 없는 출력만 허용, 임시 디렉터리에서 완성 후 원자적 게시, 오류 시 정리 |
| F17 | P1 | 원본 산출물 안으로 export하거나 symlink 출력 경로를 쓰면 원본 manifest/외부 디렉터리가 변경됨 | 복사 전에 원본·출력 관계와 symlink 확인. 원본/기존 파일 보존 테스트 |
| F18 | P2 | 측정된 `-O3`가 freestanding 빌드의 기본 `-Os`로 덮어써짐. 옵션 없는 측정도 `-Os`를 임의 추가 | 선언된 옵션 또는 측정 당시 옵션 부재를 유지. 측정과 충돌하는 옵션·추가 flags는 쓰기 전에 거부 |
| F19 | P1 | `align_corners=True` bilinear resize의 출력 축 크기가 1이면 첫 좌표 대신 중앙을 선택. 대표값 원본 `1`, 기존 plan/C 약 `5` | singleton aligned 축의 분모를 1로 두고 첫 좌표 선택. 손으로 계산한 Python/C golden 12개와 ASan/UBSan·guard 확인 |
| F20 | P1 | ESP32 최적화 1×1 Conv가 padding을 무시해 padded 입력에서 ASan 경계 밖 읽기 | padding이 있는 ESP32 1×1 후보를 제외하고 portable C로 안전하게 fallback. padding 1·2의 출력 golden 확인 |
| F21 | P1 | CMSIS-NN의 좁은 필드에 큰 좌표·reduction 길이가 잘림. depthwise 높이 32769에서 2바이트 불일치, Conv reduction 65536에서 기대 `127,127` 대신 `0,0` | int16 좌표 원점, uint16 reduction, int32 좌표 산술 한계를 후보 선택 전에 검사. 경계 안은 계속 CMSIS, 경계 밖은 portable C로 실행해 byte-exact 확인 |
| F22 | P1 | vendor requantization에서 `-31` shift의 signed `1 << 31` 및 INT32 극값에 output zero-point를 더할 때 UB/overflow | 증명된 accumulator 범위의 양 끝을 requantize하여 vendor int32 연산 가능 여부 검사. CMSIS/ESP-NN Conv·Depthwise·Linear의 정상/위험 경계와 UBSan 확인 |
| F23 | P2 | portable arena 48바이트는 예산 내인데 static 후보의 scratch로 80바이트가 되어, 배포 가능한 모델도 선택 후 실패 | scratch 크기와 정렬을 함께 계산해 SRAM 내 후보 조합으로 fallback. `REQUIRE_OPTIMIZED`, measured 증거/사유를 보존하고 generated C를 sanitizer로 실행 |
| F24 | P1 | CMSIS DSP depthwise im2col의 kernel 밖 수직 padding이 선언된 scratch보다 많이 memset. 이전 호스트 검사에서는 reference 분기 때문에 누락 | DSP 소스의 실제 buffer-fill 블록으로 ASan 재현. padding/window 제한 복원, 안전한 경계·수평 padding·generic 분기는 유지 |
| F25 | P1 | ESP-NN generic depthwise의 int16 원점이 32768에서 음수로 잘려 출력 손실 | ESP32와 S3의 해당 generic C 분기에만 int16 원점 범위를 검사. ESP32 C 출력 및 S3 선택 조건 검증 |

새 테스트는 `test_frontend_audit.py`, `test_ir_defensive_audit.py`,
`test_ptq_defensive_audit.py`, `test_bilinear_codegen_audit.py`,
`p2/test_backend_defensive_audit.py`, `p2/test_backend_budget_audit.py`,
`p2/test_cmsis_dsp_im2col_audit.py`,
`targets/test_export_safety.py`, `targets/test_build_flags.py`와 기존
`test_fixedpoint.py`에 추가했다.
정상 입력을 잘못 거부하지 않도록 양성 대조 사례도 포함한다.

## 남은 경계와 제외 영역

- kernel selection은 현재 **SRAM arena/scratch** 예산에 대해 후보를 재선택한다.
  후보별 packed constant가 달라지는 Flash 예산은 최종 생성 payload에서 실패시키지만,
  다음 Flash-feasible 후보를 다시 고르는 기능은 아직 없다.
- vendor 경계 수정은 저장소에 고정된 CMSIS-NN 4.0.0과 ESP-NN 1.2.6 소스 계약을
  대상으로 한다. 다른 vendor 버전이나 물리 보드 전체 실행의 안전성을 일반화하지 않는다.
- CMSIS 호스트 sanitizer는 DSP instruction 경로 전체를 실행하지 않는다.
  이번 DSP 재현은 실제 소스에서 추출한 im2col buffer-fill 블록의 검사이며,
  ESP32-S3 새 회귀는 dispatcher 선택 조건 검사다. 타깃 assembly 실행 검증으로
  간주하지 않는다.
- 중단 중 보관한 임시 patch는 감사 이력일 뿐 최종 적용 근거가 아니다. 현재 변경은
  저장소 테스트와 직접 검토한 코드이며, 보관 patch 자체는 배포 산출물이 아니다.

## 동작 변경 시 주의할 점

- 이전에 잘못 컴파일되던 dilation·공유 mutation·잘못된 IR은 이제 명시적인
  오류로 종료된다. 해당 연산의 새 커널 지원을 추가한 것은 아니다.
- 안전하지 않은 vendor 후보는 선택에서 제외하고 정책에 따라 다른 지원 커널을 고른다.
  `REQUIRE_OPTIMIZED`에서 가능한 최적화 후보가 없으면 명확히 실패한다.
- SRAM 예산 fallback은 공유 scratch 크기와 정렬을 대상으로 하며 전체 그래프의
  전역 최소 지연을 계산한다고 주장하지 않는다.
- ESP-IDF/Zephyr 출력 디렉터리가 비어 있지 않으면 export가 거부된다. 기존 앱과
  통합할 때는 새 component 디렉터리에 export하고 빌드 구성을 연결해야 한다.
- 고정소수점 수치 profile, C ABI, manifest schema 및 패키지 버전은 변경하지 않았다.
  수정 효과를 배포 모델에 반영하려면 해당 모델을 재컴파일·검증해야 한다.
- SHA-256 manifest는 자체 무결성 검사이지 서명이 아니다. 이 구분과 신뢰한
  PyTorch 모델 코드만 실행해야 한다는 호스트 경계를 `SECURITY.md`에 명시했다.
  0.1.x/alpha로 남아 있던 지원 문서도 1.x 정책과 일치시켰다.
- 커밋, push, 배포, 외부 보고, 물리 보드 flashing은 수행하지 않았다.

## 추가 권장 작업

이번 변경에 다음 대규모 기능을 임의로 구현하지 않았다. 우선순위에 따른 후속 제안이다.

1. **독립 수치 oracle을 사용하는 경계 조건 CI 확대.** 출력 한 축이 1인 resize,
   반복 모듈, alias, 양자화 tie, 좁은 vendor 필드 경계를 작은 고정 fixture로
   유지하고 PyTorch/LiteRT·정수 기준 실행·C를 각각 비교한다. 무작위 테스트도
   실패 seed를 저장해 작은 회귀 fixture로 축소하는 체계가 유용하다.
2. **Flash-aware 후보 재선택.** packed constant와 생성 코드의 결정적 크기 모델을
   선택 전에 계산해, preferred 후보가 Flash 예산을 넘으면 다음 feasible 후보를
   고른다. 현재는 최종 payload 검사에서 안전하게 실패한다.
3. **호스트 입력의 자원 한도.** TFLite 파일 크기, tensor/op 수, 생성 코드 크기,
   캘리브레이션 샘플 수와 수행 시간의 명시적 한도를 설계한다. 현재 shape 검증이
   host process의 메모리/CPU 격리를 대신하지 않는다.
4. **미지원 의미 사전 진단.** 컴파일 전에 연산 종류와 거부 이유를 수집하는
   검사 API/명령이 있으면 사용자가 첫 오류를 하나씩 수정하며 반복할 필요가 줄어든다.
5. **최종 타깃 회귀와 재현성.** ESP32-S3 등 실제 보드의 출력 비교, 스택 사용량,
   ELF/map/UART 보관을 강화한다. source distribution에서도 전체 테스트를
   재현할 수 있도록 test helper·fixture 포함 정책을 정리한다.
6. **원래 제품 범위 안의 기능 확장.** dilation/ceil pooling, 공유 view mutation의
   정확한 functionalization, QAT는 별도 수치 계약·검증을 갖춘 작업으로 진행한다.
   동적 shape·런타임 모델 교체는 현재 정적 MCU AOT 설계의 작은 버그 수정과 다르다.

## 검증 환경

최종 결과:

| 검사 | 결과 |
|---|---|
| 수정 전 전체 suite | 355 passed, 6 subtests passed |
| 최종 전체 suite | **504 passed, 6 subtests passed**, 실패/skip 없음, 284.09초 |
| 추가된 회귀·정상 대조 사례 | **149개** (기준 suite 355개 대비, 이번 직접 재검토로 17개 증가) |
| 직접 재검토의 backend 관련 검사 | 73 passed; DSP im2col 소스 블록 및 ESP32 실행 회귀 포함 |
| SRAM 선택 독립 전수 비교 | seed 20260909, 500개 후보 집합 × 3개 정책 = 1,500건이 모든 조합을 열거한 결과와 일치 |
| Cortex-M4 padding 경계 ELF | padding 1은 CMSIS, padding 2는 portable로 링크 성공. 미해결·금지 심볼 없음 |
| wheel 및 sdist 빌드 | 성공 |
| `twine check` | 두 패키지 모두 PASSED |
| 새 venv의 wheel 설치 | 설치 경로가 해당 venv의 `site-packages/bakenn`임을 확인 |
| 설치된 패키지 smoke | ESP-IDF project export, manifest 생성, strict C11 컴파일·공유 라이브러리 링크 성공 |
| `git diff --check` | 통과 |

전체 suite 결과 XML은
`/private/tmp/bakenn-direct-review.dERIrc/pytest-final.xml`에 보관했다.
검증용 패키지와 새 venv도 같은 임시 디렉터리에 있다. venv는 의존성 다운로드 없이
기존 시스템 패키지를 공유했으며, BakeNN 자체는 새로 빌드한 wheel에서 가져왔음을
검사했다. 이는 모든 의존성을 새로 설치한 clean-room 테스트와는 다르다.

- macOS arm64, Python 3.13.9, NumPy 2.3.5.
- PyTorch 2.10.0, torchvision 0.25.0, TFLite schema 2.18.0, LiteRT 2.2.0.
- Apple Clang 21.0.0. 로컬 `gcc`도 Apple Clang이므로 독립적인 호스트 GNU GCC
  검증으로 세지 않았다.
- ARM GNU cross compiler 16.2.0, RISC-V GNU cross compiler 15.2.0.
- `idf.py`와 `west`는 현재 PATH에 없으며 물리 MCU 실행은 하지 않았다.
- 새 CI 설정은 로컬 파일 변경이다. 원격 GitHub Actions 실행 결과로 주장하지 않는다.

## 재실행

저장소 루트에서 필요한 extras가 설치된 Python으로:

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m pytest -q tests/test_frontend_audit.py tests/test_ir_defensive_audit.py tests/test_ptq_defensive_audit.py tests/test_bilinear_codegen_audit.py tests/p2/test_backend_defensive_audit.py tests/p2/test_backend_budget_audit.py tests/p2/test_cmsis_dsp_im2col_audit.py tests/targets/test_export_safety.py tests/targets/test_build_flags.py tests/test_fixedpoint.py
python -m build --no-isolation --outdir /path/to/audit-package
```

테스트는 기존 strict C11/ASan/UBSan 및 사용 가능한 ARM/RISC-V cross-build 검사를
포함한다. frontend extras가 없는 환경에서는 일부 테스트가 skip되므로 릴리스 시
각 extras 전용 CI job도 함께 통과해야 한다.
