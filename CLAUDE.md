# DHB-XR Project

## 하네스: Project Page & Simulation Quality

**목표:** Simulation demo videos와 GitHub Pages project page의 품질을 OpenVLA/Octo/FAST 수준으로 끌어올린다.

**에이전트 팀:**
| 에이전트 | 역할 |
|---------|------|
| sim-visualizer | Demo video scripts 개선, matplotlib 애니메이션 품질 향상 |
| page-crafter | HTML project page 레이아웃·내러티브·섹션 강화 |
| qa-critic | 두 에이전트 출력물 리뷰, 구체적 피드백, 최종 승인 |

**스킬:**
| 스킬 | 용도 | 사용 에이전트 |
|------|------|-------------|
| dhb-project-improve | 전체 팀 오케스트레이션 (fan-out → QA → integrate) | 오케스트레이터 |

**실행 규칙:**
- project page 개선, simulation 품질 향상, 비디오 수정, 페이지 레이아웃 개선 요청 시 `dhb-project-improve` 스킬로 에이전트 팀 실행
- 단순 파일 수정/확인은 직접 처리 가능
- 모든 에이전트는 `model: "opus"` 사용
- 중간 산출물: `_workspace/` 디렉토리
- 품질 기준: OpenVLA, Octo, FAST tokenizer, pi0 project pages

**디렉토리 구조:**
```
.claude/
├── agents/
│   ├── sim-visualizer.md
│   ├── page-crafter.md
│   └── qa-critic.md
└── skills/
    └── dhb-project-improve/
        └── SKILL.md
```

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-04-10 | 초기 구성 | 전체 | Project page quality improvement harness |
