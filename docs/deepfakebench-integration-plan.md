# DeepfakeBench 통합 방향 (services/ai 기준)

이 문서는 `DeepfakeBench` 코드를 `verifake` 레포 안으로 가져올 때,
기존 레포 구조를 깨지 않으면서 `services/ai/` 기준으로 어떻게 배치하는 것이 좋은지 정리한 문서입니다.

---

## 1. 추천 배치 위치

추천 경로:

```text
verifake/
  services/
    ai/
      deepfakebench/
```

즉,
`DeepfakeBench` 원본 성격을 유지하되,
`verifake` 안에서는 AI 서비스 레이어의 하위 엔진으로 취급합니다.

---

## 2. 추천 폴더 구조

```text
verifake/
  docs/
    deepfakebench-integration-plan.md
  infra/
  services/
    ai/
      README.md
      deepfakebench/
        README.md
        LICENSE
        analysis/
        preprocessing/
        training/
```

---

## 3. 구조 메모

- `services/ai/deepfakebench/` = 외부 기반 벤치마크 엔진
- 이후 `services/ai/inference/`, `services/ai/pipelines/` = 우리 서비스용 래퍼/추론 로직
- 외부 코드는 독립된 하위 폴더로 유지

---

## 4. 어떤 부분을 우선 가져오면 좋은가

우선순위 기준으로 보면:

### 1순위
- `training/`
- `preprocessing/`

### 2순위
- `analysis/`

### 3순위
- 필요 시 `README.md`, `LICENSE`

초기에는 전체를 다 활용하기보다,
**전처리 + 학습/탐지 핵심 구조 중심**으로 들여오는 것이 좋습니다.

---

## 5. 현재 단계에서 추천하는 통합 방식

### 단계 1
`services/ai/deepfakebench/` 폴더를 만들고,
외부 코드가 이 위치에 들어온다는 기준을 먼저 고정합니다.

### 단계 2
`services/ai/README.md`를 두어,
이 레이어가 어떤 목적의 코드들을 담는지 설명합니다.

### 단계 3
`DeepfakeBench`는 원본 그대로 가져오되,
`verifake` 서비스용 코드는 그 바깥에 별도로 만듭니다.

예:

```text
services/ai/
  deepfakebench/
  inference/
  common/
```

---

## 6. 지금 바로 하면 좋은 최소 작업

1. `docs/deepfakebench-integration-plan.md` 추가
2. `services/ai/README.md` 추가
3. `services/ai/deepfakebench/` 구조 준비
4. 이후 실제 코드 반입은 별도 커밋으로 진행

즉,
**구조와 기준을 먼저 만들고, 원본 코드는 그 다음에 넣는 방식**이 안전합니다.

---

## 7. 한 줄 결론

`DeepfakeBench` 코드는 **`verifake/services/ai/deepfakebench/` 아래에 두는 구조**로 가져가고,
기존 레포 구조는 유지한 채 AI 엔진 영역만 분리해서 관리합니다.
