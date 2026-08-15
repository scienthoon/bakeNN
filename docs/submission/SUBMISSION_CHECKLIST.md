# 오픈소스 개발자대회 제출 체크리스트

마감: 2026년 8월 27일 18:00. 접속 혼잡을 피하기 위해 먼저 제출 완료 후
마감 전 수정하는 방식이 안전하다.

## 현재 준비된 항목

- [x] 공개 소스코드: https://github.com/scienthoon/bakeNN
- [x] Apache-2.0 라이선스와 vendored dependency license/provenance
- [x] 결과보고서 독립 DOCX 초안과 PDF
- [x] 재현 가능한 실보드 benchmark 원본 UART/JSON/Markdown
- [x] FP32→PTQ→ESP32-S3 one-command demo와 3분 시연 대본
- [x] CI, contribution/security/conduct 파일, issue/PR template

## 사용자가 완료해야 하는 외부 항목

- [ ] 운영사무국 공식 결과보고서 양식에 DOCX 초안 내용을 이관한다.
- [ ] 3분 시연 대본으로 화면을 녹화하고 공개/미등록 영상 URL을 만든다.
- [ ] 공식 양식의 소스코드 링크와 시연영상 링크를 채운다.
- [ ] green PR을 main에 병합한 뒤 `v0.1.0` tag와 GitHub Release를 발행한다.
- [ ] 원본과 PDF를 열어 링크·표·페이지 번호를 마지막으로 확인한다.
- [ ] osscontest.kr의 `참가신청 > 접수 및 조회`에서 파일을 업로드한다.
- [ ] `출품작 제출 완료하기`를 누른다.
- [ ] 화면의 제출 상태가 `제출 완료`인지 확인한다.
- [ ] 자동 발송된 제출 완료 안내 메일을 확인한다.

## 시연영상 권장 흐름

1. FP32 `DemoCNN` 코드를 15초간 보여준다.
2. `generate.py`를 실행하고 PTQ·arena·선택된 ESP-NN kernel ID를 보여준다.
3. 생성 `model.c`와 manifest에서 호출 순서·SRAM offset·resource를 보여준다.
4. ESP-IDF build 또는 nRF52840 실측 결과를 보여준다.
5. 동일 FC workload의 BakeNN/TFLM cycles·Flash·SRAM 표와 한계를 설명한다.

영상에서는 ESP32-S3 cycle이 아직 실측되지 않았다는 점과 nRF52840 수치가
동결된 FC/Conv workload에만 적용된다는 점을 명확히 말한다.
