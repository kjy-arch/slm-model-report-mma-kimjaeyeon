# SLM 모델 분석 보고서

> **소속:** 병무청 &nbsp;|&nbsp; **이름:** 김재연 &nbsp;|&nbsp; **선택 모델:** `kanana-nano-2.1b-instruct`
>
> 🤗 **Hugging Face 모델 링크:** [huggingface.co/kakaocorp/kanana-nano-2.1b-instruct](https://huggingface.co/kakaocorp/kanana-nano-2.1b-instruct)

---

## 목차

1. [모델 선택 이유](#01-모델-선택-이유)
2. [모델 기본 정보](#02-모델-기본-정보)
3. [학습 데이터와 튜닝 방식](#03-학습-데이터와-튜닝-방식)
4. [파라미터 및 구조적 특징](#04-파라미터-및-구조적-특징)
5. [모델 파일 구성](#05-모델-파일-구성)
6. [활용 가능 업무](#06-활용-가능-업무)
7. [한계와 주의사항](#07-한계와-주의사항)
8. [종합 의견](#08-종합-의견)
9. [참고 링크](#09-참고-링크)

---

## 01. 모델 선택 이유

본 업무는 인터넷 상의 **허위정보·병역면탈 조장정보·불건전정보**를 자동으로 탐지·분류하는 공공행정 목적의 AI 시스템 구축을 목표로 한다. 이 맥락에서 `kanana-nano-2.1b-instruct`를 선택한 이유는 다음과 같다.

- **한국어 최적화:** 카카오가 개발한 한국어 중심 이중언어(한·영) 모델로, 동급(2B대) 모델 중 KMMLU(한국어 지식)·HAE-RAE(한국어 이해) 벤치마크에서 최상위 성능을 기록한다.
- **RAG 친화적 생태계:** 동일 시리즈에서 Base·Instruct·Embedding·Function Call·**RAG 전용** 버전이 모두 공개되어, 웹 검색 결과 텍스트를 입력해 판별하는 검색-증강생성(RAG) 파이프라인 구성이 용이하다.
- **경량·온프레미스 운영 가능:** 2.1B 파라미터(BF16 약 4.2 GB)로 GPU 8 GB 이상 서버에서 독립 운영이 가능하여, 행정 보안망(인터넷 망분리) 환경에 적합하다.
- **국내 기업 개발:** 카카오가 개발·공개하여 국산 AI 활용 정책 방향에 부합하며, 기술 지원 창구(kanana-llm@kakaocorp.com)가 명확하다.
- **명확한 라이선스:** CC-BY-NC-4.0 라이선스로 비상업적 목적의 공공행정 내부 활용이 가능하다 (도입 전 법무 검토 권고).

---

## 02. 모델 기본 정보

| 항목 | 내용 |
|------|------|
| **모델명** | `kanana-nano-2.1b-instruct` |
| **개발 주체** | 카카오(주) — Kakao Corp. (Kanana LLM Team) |
| **모델 규모** | 약 **2.1 B (21억) 파라미터** ｜ BF16 정밀도 ｜ 파일 크기 약 4.17 GB |
| **라이선스** | CC-BY-NC-4.0 (비상업적 사용 허용) |
| **모델 유형** | Instruct (Chat) — SFT + 선호도 최적화(Preference Optimization) 적용 모델 |

---

## 03. 학습 데이터와 튜닝 방식

### ① 사전학습(Pre-training) 데이터

- 한국어·영어 이중언어 웹 코퍼스를 중심으로 구성되며, **카카오 사용자 데이터는 일절 포함되지 않음**을 공식 명시.
- 고품질 데이터 필터링(High-quality data filtering), 단계별 사전학습(Staged pre-training), 깊이 업스케일링(Depth up-scaling), 가지치기 및 지식 증류(Pruning & Distillation) 기법을 통해 동급 모델 대비 현저히 낮은 연산 비용(FLOPs)으로 경쟁력 있는 성능을 달성.

### ② 사후학습(Post-training) 방식

- **SFT (Supervised Fine-Tuning):** 명령 수행 능력 향상을 위한 지도 미세조정. 한국어·영어 Instruction 데이터 학습.
- **선호도 최적화 (Preference Optimization):** 사용자와의 자연스러운 상호작용 품질 향상을 위한 DPO 계열 학습 적용.

### ③ 특화 적응(Adaptation) 버전

- 동일 베이스 모델에서 **Embedding / Function Call / RAG** 전용 모델이 별도로 학습·공개되어, 검색 증강 생성(RAG) 파이프라인 구성 시 RAG 전용 모델 활용을 권장.

---

## 04. 파라미터 및 구조적 특징

| 항목 | 내용 |
|------|------|
| **Architecture** | `LlamaForCausalLM` (Llama 계열 디코더 전용 트랜스포머) |
| **Context Length** | 8,192 tokens (`max_position_embeddings: 8192`) |
| **Hidden Size** | 1,792 (`hidden_size: 1792`) |
| **Layers** | 32 레이어 (`num_hidden_layers: 32`) |
| **Attention Heads** | Q-heads: 24 / KV-heads: 8 (Grouped Query Attention, GQA 적용) |
| **Tokenizer** | Llama 계열 BPE 토크나이저 / Vocab size: **128,256** / 파일 크기 17.2 MB |

---

## 05. 모델 파일 구성

| 파일명 | 크기 | 설명 |
|--------|------|------|
| 📄 `README.md` | 24.8 kB | 모델 카드 (소개·성능·사용법·라이선스) |
| ⚙️ `config.json` | 718 B | 모델 아키텍처 설정 (레이어·헤드·vocab 등) |
| ⚙️ `generation_config.json` | 126 B | 기본 생성 파라미터 (max_new_tokens 등) |
| 🧠 `model.safetensors` | 4.17 GB | 가중치 파일 (Safetensors 단일 샤드, BF16) |
| 🔤 `tokenizer.json` | 17.2 MB | BPE 토크나이저 어휘·병합 규칙 |
| 🔤 `tokenizer_config.json` | 51.1 kB | 채팅 템플릿 포함 토크나이저 설정 |
| 🔤 `special_tokens_map.json` | 444 B | BOS/EOS/PAD 등 특수 토큰 매핑 |
| 📁 `assets/` | — | 로고·성능 그래프 등 이미지 리소스 |

> **참고:** 가중치가 **단일 Safetensors 파일**(model.safetensors)로 제공되어 로딩이 간단. 총 저장 용량 약 4.2 GB, GPU VRAM 8 GB 이상 환경에서 운영 가능.

### 파생 모델 현황 (HF 모델 트리)

- 어댑터(Adapters): 10개 모델 공개
- 파인튜닝(Finetunes): 15개 모델 공개
- 양자화(Quantizations): 12개 모델 공개 (GGUF·GPTQ 등)

---

## 06. 활용 가능 업무

### ① 허위정보·불건전정보 탐지 (본 과제 핵심 업무)

- 웹 크롤링 또는 검색 API를 통해 수집한 텍스트를 모델에 입력, 시스템 프롬프트에 판별 기준을 정의하여 "해당/비해당" 분류 및 근거 출력.
- RAG 버전(`kanana-nano-2.1b-rag`)과 연계 시 검색 결과를 직접 컨텍스트로 주입하는 파이프라인 구성 가능.

### ② 병역면탈 조장정보 판별

- 병무청·법령 기준을 시스템 프롬프트로 입력하고, 게시물·댓글 텍스트의 위법 가능성을 판별하는 자동 1차 스크리닝 활용.

### ③ 공문서·보고서 요약 및 초안 생성

- 8,192 토큰 컨텍스트를 활용한 중간 길이 문서 요약, 회의록 정리, 기안 초안 생성.

### ④ 민원 분류 및 응답 지원

- 민원 텍스트를 카테고리별로 분류하고 FAQ 기반 1차 답변 초안을 생성하는 챗봇 백엔드로 활용.

### ⑤ Function Calling 기반 자동화

- Function Call 전용 버전(`kanana-nano-2.1b-function-call`) 연계 시 외부 API·DB 조회를 자연어 명령으로 자동화.

---

## 07. 한계와 주의사항

- **컨텍스트 길이:** 기본 8,192 토큰으로, 장문 법령·판례 문서 전체 처리에는 한계가 있음. 긴 문서는 청킹(chunking) 분할 후 처리 필요.
- **수학·코딩 성능:** GSM8K(수학) 46.32, HumanEval+(코딩) 63.41로 일반 지식·추론 모델 대비 낮음. 수리 계산이 필요한 업무에는 부적합.
- **허위정보 최종 판단 불가:** AI 출력은 보조 자료로만 활용해야 하며, 법적 효력이 있는 최종 판단은 반드시 담당 공무원이 수행해야 함.
- **라이선스 제약:** CC-BY-NC-4.0은 상업적 이용 금지. 외부 서비스로 공개하거나 유상 제공 시 라이선스 위반 가능성. 도입 전 법무 검토 필수.
- **환각(Hallucination) 위험:** 모든 생성형 AI와 마찬가지로 사실과 다른 정보를 생성할 수 있음. 출력 결과의 교차 검증 체계 마련 필요.
- **Knowledge Cutoff:** 학습 데이터 기준일 이후의 최신 사건·법령 개정 등은 반영되지 않을 수 있음. RAG 연계로 보완 권장.
- **안전 필터 미포함:** 별도 안전 메커니즘을 구현하지 않은 경우 부적절한 내용이 생성될 수 있음. 시스템 프롬프트 또는 후처리 필터 적용 필요.

---

## 08. 종합 의견

`kanana-nano-2.1b-instruct`는 **한국어 처리 성능**과 **경량성**, **국내 기업 개발**이라는 세 가지 요소를 동시에 충족하는 현실적인 공공행정용 SLM이다.

특히 **허위정보·불건전정보 탐지 업무**에서 강점을 보이는 이유는, 동급 2B대 모델 중 **KMMLU·HAE-RAE 한국어 이해 벤치마크 1위**를 기록하며 한국어 문맥 파악 능력이 검증되었고, **RAG 전용 파생 모델**을 통해 실시간 웹 검색 결과를 컨텍스트로 주입하는 구조를 쉽게 구성할 수 있기 때문이다.

운영 측면에서는 **단일 Safetensors 파일(4.2 GB)**로 GPU 8 GB 서버에서 **온프레미스 배포**가 가능해 **망분리 환경의 행정기관**에 적합하다. **vLLM·SGLang** 등 고성능 추론 엔진도 공식 지원한다.

다만 **8K 컨텍스트 한계**와 **CC-BY-NC-4.0 라이선스**는 도입 전 반드시 확인해야 하며, AI 출력은 **담당자의 최종 판단을 보조하는 역할**로 운영 범위를 명확히 해야 한다. 향후 업무 확대나 성능 향상이 필요하면 동일 시리즈의 `kanana-1.5-2.1b-instruct-2505`(**Apache 2.0**, **128K 컨텍스트**)로 마이그레이션하는 경로도 열려 있어 장기적 확장성이 양호하다.

---

## 09. 참고 링크

- 🤗 [Hugging Face 모델 카드 — kakaocorp/kanana-nano-2.1b-instruct](https://huggingface.co/kakaocorp/kanana-nano-2.1b-instruct)
- 📄 [Technical Report — Kanana: Compute-efficient Bilingual Language Models (arXiv:2502.18934)](https://arxiv.org/abs/2502.18934)
- 📦 [Hugging Face Collection — Kanana Nano 2.1B (Base·Instruct·Embedding·Function Call·RAG)](https://huggingface.co/collections/kakaocorp/kanana-nano-21b-67a326cda1c449c8d4172259)

---

## 제출 정보

| 항목 | 내용 |
|------|------|
| **GitHub Repository** | [https://github.com/kjy-arch/slm-model-report-mma-kimjaeyeon](https://github.com/kjy-arch/slm-model-report-mma-kimjaeyeon.git) |
| **파일명** | 병무청_김재연_slm_report.html |
