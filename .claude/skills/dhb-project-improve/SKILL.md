---
name: dhb-project-improve
description: "Runs the DHB-XR project page improvement team — sim-visualizer, page-crafter, and qa-critic agents working in parallel to improve simulation video quality and project page presentation to match top robotics papers (OpenVLA, Octo, FAST, pi0). Trigger when user asks to: improve the project page, improve simulation results, make the page look better, fix videos, match quality of other papers, run the improvement team, re-run improvements, update the project page, or improve presentation quality."
---

# DHB-XR Project Page Improvement Orchestrator

Runs a 3-agent team to improve simulation video quality and project page presentation
to the standard of top robotics papers (OpenVLA, Octo, FAST tokenizer, pi0).

## Execution Mode: Agent Team (fan-out → QA → integrate)

## Agent Team

| Agent | Type | Role | Output |
|-------|------|------|--------|
| sim-visualizer | general-purpose | Improve demo video scripts and animations | `assets/demo_*_v2.mp4` |
| page-crafter | general-purpose | Improve HTML layout, narrative, sections | `docs/paper/index.html` |
| qa-critic | general-purpose | Review both outputs, request changes | `_workspace/qa_*.md` |

## Reference Benchmark
The target quality bar is set by these project pages (user explicitly requested parity):
- **OpenVLA**: openvla.github.io — side-by-side video grid, explicit success/fail indicators
- **Octo**: octo-models.github.io — Bulma carousel, animated hero, stat numbers
- **FAST tokenizer**: pi.website/research/fast — pipeline visualization, minimal code example
- **pi0**: pi.website/blog/pi0 — retro aesthetic, isometric robot illustrations
- **Nerfies template**: standard Bulma + FontAwesome base

## Workspace
All intermediate files: `dhb_xr/_workspace/`

## Workflow

### Phase 0: Context Check
1. Check if `_workspace/` exists with prior run outputs
2. If yes + partial update requested → restart only that agent
3. If yes + new run → rename to `_workspace_{timestamp}/`, fresh start
4. If no → initial run

### Phase 1: Preparation
1. Inventory existing assets: `ls docs/paper/assets/`
2. Read current HTML: `docs/paper/index.html`
3. Read current video script: `dhb_xr_paper/scripts/gen_demo_videos.py`
4. Create `_workspace/` with subdirs: `00_input/`, `sim/`, `page/`, `qa/`
5. Save asset list to `_workspace/00_input/assets.txt`

### Phase 2: Team Assembly & Parallel Execution

```
TeamCreate(
  team_name: "dhb-improve",
  members: [
    {
      name: "sim-visualizer",
      agent_type: "general-purpose",
      model: "opus",
      prompt: "Read .claude/agents/sim-visualizer.md for your role. 
               Current script: dhb_xr_paper/scripts/gen_demo_videos.py
               Current assets: dhb_xr/docs/paper/assets/demo_*.mp4
               Improve the simulation videos per your agent spec.
               When done, message qa-critic with your file list."
    },
    {
      name: "page-crafter",
      agent_type: "general-purpose",
      model: "opus",
      prompt: "Read .claude/agents/page-crafter.md for your role.
               Current page: dhb_xr/docs/paper/index.html
               Target quality: OpenVLA (openvla.github.io), Octo (octo-models.github.io), 
               FAST (pi.website/research/fast) — fetch these pages and compare.
               Improve the HTML per your agent spec.
               IMPORTANT: coordinate with sim-visualizer for any new video filenames.
               When done, message qa-critic with your change summary."
    },
    {
      name: "qa-critic",
      agent_type: "general-purpose",
      model: "opus",
      prompt: "Read .claude/agents/qa-critic.md for your role.
               Wait for sim-visualizer and page-crafter to complete.
               Then run your full review process.
               Send specific feedback to each agent.
               Write final sign-off to _workspace/qa_final.md"
    }
  ]
)
```

Tasks:
```
TaskCreate(tasks: [
  {title: "Improve simulation videos", assignee: "sim-visualizer",
   description: "Generate v2 demo videos with better visual storytelling. See agent spec."},
  {title: "Improve project page HTML", assignee: "page-crafter",
   description: "Strengthen HTML per agent spec. Benchmark against OpenVLA/Octo/FAST."},
  {title: "QA review — first pass", assignee: "qa-critic",
   depends_on: ["Improve simulation videos", "Improve project page HTML"],
   description: "Review all outputs. Send targeted feedback."},
  {title: "Apply QA feedback — sim", assignee: "sim-visualizer",
   depends_on: ["QA review — first pass"],
   description: "Address qa-critic feedback on videos."},
  {title: "Apply QA feedback — page", assignee: "page-crafter",
   depends_on: ["QA review — first pass"],
   description: "Address qa-critic feedback on HTML."},
  {title: "QA final sign-off", assignee: "qa-critic",
   depends_on: ["Apply QA feedback — sim", "Apply QA feedback — page"],
   description: "Final review. Write qa_final.md. Message orchestrator APPROVED."}
])
```

### Phase 3: Integration (after QA approves)

1. Replace v2 videos with final names (drop `_v2` suffix): `mv assets/demo_*_v2.mp4 assets/demo_*.mp4`
2. Build site: `pixi run mkdocs build` from `dhb_xr/`
3. Verify build succeeds (no errors)
4. Restart local server on `site/` directory

### Phase 4: Deploy
```bash
git add docs/paper/ && git commit -m "Improve project page: videos + layout (harness run)"
git push origin main
```

## Error Handling
- If a video script fails: keep previous video, note in report, continue with page improvements
- If HTML edit breaks build: revert to git HEAD for that file, flag to user
- If QA requests >3 revision rounds: escalate to user with specific blocker

## Test Scenarios
**Normal flow:** "improve the project page quality" → team runs, QA approves, deployed
**Partial re-run:** "the videos still need work" → only sim-visualizer re-runs
**Quality bar check:** "is the page on par with OpenVLA?" → qa-critic runs comparison only
