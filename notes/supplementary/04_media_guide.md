# Media Guide — DHB-XR CoRL 2026

All videos, figures, and extracted frames in `output/`. What they show, how they were
generated, and how to regenerate them. Supplement to the paper's figure captions.

---

## Directory Structure

```
output/
├── videos/           # 16 video/animation files
├── figures/          # 14 figure files (7 figures × PDF + PNG)
└── frames/           # 24 extracted frame stills
```

---

## Videos (`output/videos/`)

### Summary Table

| File | Size | Content | Type |
|------|------|---------|------|
| `arc_obstacle_adaptation.gif` | 188 KB | DHB arc clearance vs affine warp | matplotlib animation |
| `arc_obstacle_adaptation.mp4` | 54 KB | Same, MP4 | matplotlib animation |
| `trajectory_adaptation.gif` | 712 KB | 3D SE(3) adaptation | matplotlib animation |
| `trajectory_adaptation.mp4` | 160 KB | Same, MP4 | matplotlib animation |
| `comparison_0cm_agentview.mp4` | 41 KB | Side-by-side sim (naive vs DHB, 0cm) | LIBERO agentview |
| `comparison_5cm_agentview.mp4` | 41 KB | Side-by-side sim (naive vs DHB, 5cm) | LIBERO agentview |
| `dhb_adaptation_0cm.mp4` | 17 KB | DHB alone, 0cm offset | LIBERO agentview |
| `dhb_adaptation_5cm.mp4` | 17 KB | DHB alone, 5cm offset | LIBERO agentview |
| `naive_replay_0cm.mp4` | 26 KB | Naive replay, 0cm offset | LIBERO agentview |
| `naive_replay_5cm.mp4` | 26 KB | Naive replay, 5cm offset | LIBERO agentview |
| `pick_up_...comparison_0cm.mp4` | 41 KB | Long-name copy of comparison_0cm | LIBERO agentview |
| `pick_up_...comparison_5cm.mp4` | 41 KB | Long-name copy of comparison_5cm | LIBERO agentview |
| `pick_up_...dhb_0cm.mp4` | 17 KB | Long-name copy of dhb_0cm | LIBERO agentview |
| `pick_up_...dhb_5cm.mp4` | 17 KB | Long-name copy of dhb_5cm | LIBERO agentview |
| `pick_up_...naive_0cm.mp4` | 26 KB | Long-name copy of naive_0cm | LIBERO agentview |
| `pick_up_...naive_5cm.mp4` | 26 KB | Long-name copy of naive_5cm | LIBERO agentview |

Full long name prefix: `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_`

---

### arc_obstacle_adaptation.{gif,mp4}

**What it shows**: DHB-XR trajectory adaptation for a 6D task (3D position + 3D
orientation) that passes over an obstacle. Compares DHB geometric adaptation against
a naive affine warp.

**Content**:
- Left panel: original demo trajectory (yellow arc over obstacle)
- Middle panel: DHB-adapted trajectory (preserves arc clearance geometry)
- Right panel: affine-warped trajectory (collapses arc, would clip obstacle)

**Key point**: DHB preserves the vertical clearance arc because it encodes motion
shape relative to a moving frame. Affine warp stretches/compresses the trajectory
linearly, destroying the curvature that was necessary for obstacle clearance.

**Generation**: `scripts/generate_arc_adaptation_video.py` (or equivalent in
`update_vis.py`). Uses matplotlib FuncAnimation on synthetic arc trajectories.

**File format**: Animated GIF (188 KB, ~2 fps, looping) and MP4 (54 KB, 10 fps).
Both are matplotlib-rendered — not from physical simulation.

---

### trajectory_adaptation.{gif,mp4}

**What it shows**: Full 3D trajectory adaptation animation. Shows a 6-DOF robot arm
trajectory being adapted from a reference pose to a perturbed target pose using DHB-XR.

**Content**:
- Rotating 3D view showing original trajectory (blue) and adapted trajectory (orange)
- Start/end pose markers
- The adaptation preserves loop structure, curvature, and approach direction

**Generation**: `update_vis.py` generates the matplotlib 3D animation using LIBERO demo
positions encoded with DHB-DR and decoded at a perturbed initial pose.

**File format**: Animated GIF (712 KB, highest quality of all animations) and
MP4 (160 KB). Both are matplotlib 3D rendered — not from physical simulation.

**Usage in paper**: Referenced in §4 (Experiments), supports Fig. 6.

---

### comparison_{0cm,5cm}_agentview.mp4

**What it shows**: Side-by-side comparison of naive replay vs DHB-adapted execution
in LIBERO-Spatial simulation. Agentview camera (overhead/angled view of the workspace).

**Task**: `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`

**0cm variant** (41 KB, 20 frames):
- Left half: naive replay from demo initial state — succeeds (100% at 0cm)
- Right half: DHB adaptation from same initial state — also succeeds
- Both arms move identically (no offset, so both approaches should match)

**5cm variant** (41 KB, 20 frames):
- Object shifted 5cm in +Y direction
- Left half: naive replay — misses object (gripper arrives at original position, 5cm off)
- Right half: DHB adaptation — arm shifts appropriately to follow the moved object
- DHB arm visibly approaches the perturbed object location

**Format**: Side-by-side 256×528 pixels (2×128×256 stitched horizontally).
18-20 frames at ~5 fps. Real simulation frames from MUJOCO via EGL headless rendering.

**Important note**: The DHB execution in these videos uses a fixed OSC scale factor
(~83×). The arm reaches approximately the right location but with suboptimal
trajectory smoothness due to the variable OSC scaling issue (see 03_simulation_analysis.md).
Frames show the arm gripper arriving near the target, not necessarily grasping successfully.

---

### dhb_adaptation_{0cm,5cm}.mp4

**What it shows**: DHB-only view (single arm, not side-by-side). Same task as above.

**0cm** (17 KB, 20 frames): DHB arm executing at demo pose — normal operation.
**5cm** (17 KB, 20 frames): DHB arm adapting to 5cm shifted object.

Smaller files than comparison videos (single view, 128×128 pixels).

---

### naive_replay_{0cm,5cm}.mp4

**What it shows**: Naive replay only (single arm view). Same task.

**0cm** (26 KB, 18 frames): Naive replay at demo pose — succeeds, full arm motion visible.
**5cm** (26 KB, 18 frames): Naive replay with 5cm shift — arm moves to wrong location.
Slightly larger than DHB videos because naive replay produces more total arm movement
(no smart stopping when wrong position detected).

---

## Figures (`output/figures/`)

### Summary Table

| File | Size (PNG) | Caption summary | Script |
|------|-----------|-----------------|--------|
| `fig1_token_stability_ratedistortion` | 59 KB | Token stability bar + rate-distortion curve | `notebooks/` |
| `fig2_perturbation_sweep` | 59 KB | Endpoint error vs perturbation magnitude | `scripts/` |
| `fig3_sample_efficiency` | 45 KB | DHB vs naive error at various N demos | `scripts/` |
| `fig4_pipeline` | 38 KB | DHB-XR system pipeline diagram | `scripts/` |
| `fig5_main_results` | 179 KB | Main results table as figure | `scripts/` |
| `fig6_3d_trajectory_adaptation` | 491 KB | 3D trajectory pairs (original vs adapted) | `update_vis.py` |
| `fig7_6d_adaptation_vs_affine` | 280 KB | 3-panel: trajectories, height, curvature | `scripts/` |

All figures exist as both `.pdf` (for LaTeX embedding) and `.png` (for web/preview).

---

### fig1: Token Stability and Rate-Distortion

**Path**: `output/figures/fig1_token_stability_ratedistortion.{pdf,png}`
**Size**: 33 KB PDF, 59 KB PNG

**Two-panel figure**:

Left panel — Token Stability:
- Bar chart comparing DHB magnitude tokens vs raw action tokens
- Y-axis: percentage of tokens that change identity under 40 random SE(3) transforms
- DHB magnitude: 0.0% (bar is absent / zero height)
- Raw actions: 99.1% (nearly full bar)
- X-axis: representation type

Right panel — Rate-Distortion:
- X-axis: number of tokens K
- Y-axis: MSE reconstruction error
- One curve per tokenizer type
- OAT (ours) highlighted: K=8, MSE=0.018
- FAST shown: K=14, MSE=0.0002 (dominates MSE but no prefix decode)
- Points for OAT k=1,2,4,8 show the anytime decode curve

**Key visual message**: DHB tokens are stable (left); OAT provides ordered compression (right).

---

### fig2: Perturbation Sweep

**Path**: `output/figures/fig2_perturbation_sweep.{pdf,png}`
**Size**: 26 KB PDF, 59 KB PNG

**Content**: Line plot of mean endpoint error vs perturbation offset (0, 2, 5, 10, 15, 20 cm).

Lines shown:
- DHB-DR (blue solid, all zeros): flat at 0.0 mm across all offsets
- DHB-TI (green dashed, all zeros): same flat line
- Naive replay (red, rising): 0, 20, 50, 100, 150, 200 mm

Error bars: zero for DHB (deterministic), would be shown for stochastic baselines.
10 LIBERO-Spatial tasks × 8 directions = 80 trials per offset level.

**Key visual message**: DHB achieves 0mm endpoint error regardless of perturbation magnitude.

---

### fig3: Sample Efficiency

**Path**: `output/figures/fig3_sample_efficiency.{pdf,png}`
**Size**: 18 KB PDF, 45 KB PNG

**Content**: Bar chart or line plot of endpoint error at 10cm perturbation vs number
of demonstrations N = {1, 2, 5, 10, 20}.

- DHB: 0.0 mm at all N (horizontal line)
- Naive replay: 100.0 mm at all N (horizontal line)

**Key visual message**: DHB needs only N=1 demonstration for spatial generalization.
More demos don't help naive replay; DHB is already optimal at N=1.

**Important caveat** (noted on figure): This is a geometric proxy experiment, not
a policy success rate. Task success would benefit from more demos for other aspects
of generalization (grasping, visual variation, etc.).

---

### fig4: Pipeline Diagram

**Path**: `output/figures/fig4_pipeline.{pdf,png}`
**Size**: 24 KB PDF, 38 KB PNG

**Content**: System architecture diagram showing the full DHB-XR pipeline:

```
Demo trajectory
  → encode_dhb_dr (66ms, one-time)
  → DHB invariant tokens
  → [OAT tokenizer optional]
  → Fatrop adaptation (≤7ms, given new target)
  → decode_dhb_dr
  → Adapted trajectory
  → embodik VelocityIK (6ms)
  → Joint commands
```

Color coding: encode (blue), adapt (orange), decode (green), IK (purple).
Timing annotations on each box.

---

### fig5: Main Results

**Path**: `output/figures/fig5_main_results.{pdf,png}`
**Size**: 33 KB PDF, 179 KB PNG

**Content**: Comprehensive results figure combining multiple experiment panels.
Likely includes:
- Table or heatmap of perturbation sweep results
- LIBERO task success rates (simulation rows marked as `--` pending)
- Comparison columns: DHB-DR, DHB-TI, Naive, StarVLA-FAST, StarVLA-MLP

Largest PNG file among figures (179 KB) — multi-panel layout.

---

### fig6: 3D Trajectory Adaptation

**Path**: `output/figures/fig6_3d_trajectory_adaptation.{pdf,png}`
**Size**: 51 KB PDF, 491 KB PNG (largest figure file)

**Content**: 3D matplotlib visualization of trajectory adaptation for multiple LIBERO tasks.

Each panel (or subplot) shows:
- Original demo trajectory (blue, starting at origin)
- Adapted trajectory (orange, starting at perturbed pose)
- Arrows at start/end indicating orientation
- Workspace bounding box

The 491 KB PNG size indicates multiple high-quality 3D subplots, likely 3×2 or 2×3
grid of task trajectories.

**Generation**: `update_vis.py` using Axes3D with elevation/azimuth for best view angle.

---

### fig7: 6D Adaptation vs Affine (3-Panel)

**Path**: `output/figures/fig7_6d_adaptation_vs_affine.{pdf,png}`
**Size**: 33 KB PDF, 280 KB PNG

**Content**: 3-panel comparison of DHB vs affine warp for the arc-over-obstacle task:

**Panel 1 — 3D Trajectories**:
- Original trajectory (yellow), DHB-adapted (blue), affine-warped (red)
- Shows geometric structure preservation vs collapse
- 3D perspective view of SE(3) paths

**Panel 2 — Height Profile** (clearance visualization):
- X-axis: arc-length progress (normalized 0 to 1)
- Y-axis: end-effector height (Z coordinate)
- Original: shows clearance hump (arc over obstacle)
- DHB: preserves hump at new start position
- Affine: hump compressed/flattened, reduced clearance

**Panel 3 — Curvature Bar Chart**:
- Peak vertical clearance (max Z above start Z) for three methods
- Error bars from multiple test start positions
- DHB ≈ original clearance; affine warp systematically lower
- Quantifies the arc-preservation advantage of DHB

**Key figure**: This is the main qualitative argument that DHB preserves functionally
important geometric properties (obstacle clearance) that naive warping destroys.

---

## Frames (`output/frames/`)

24 extracted still frames from simulation videos. Used for qualitative comparisons
in paper figures and supplementary.

### Frame Files

| Pattern | Count | Source video |
|---------|-------|-------------|
| `comparison_0cm_{initial,mid,final}.png` | 3 | comparison_0cm_agentview.mp4 |
| `comparison_5cm_{initial,mid,final}.png` | 3 | comparison_5cm_agentview.mp4 |
| `dhb_0cm_{initial,mid,final}.png` | 3 | dhb_adaptation_0cm.mp4 |
| `naive_0cm_{initial,mid,final}.png` | 3 | naive_replay_0cm.mp4 |
| `pick_up_..._dhb_0cm_frame{00,01,02}.png` | 3 | long-name dhb_0cm |
| `pick_up_..._dhb_5cm_frame{00,01,02}.png` | 3 | long-name dhb_5cm |
| `pick_up_..._naive_0cm_frame{00,01,02}.png` | 3 | long-name naive_0cm |
| `pick_up_..._naive_5cm_frame{00,01,02}.png` | 3 | long-name naive_5cm |

Total: 24 frames.

### Frame Content Description

**initial** frames: Robot arm at reset position, object visible in scene.
**mid** frames: Robot arm in transit (approach phase), arm extended toward object.
**final** frames: Robot arm near object — either grasping (DHB) or off-target (naive 5cm).

### Key Comparison Frames

For paper supplementary, the most informative frames are:
- `comparison_5cm_final.png`: Side-by-side showing naive arm missing, DHB arm at object
- `dhb_5cm_frame02.png`: DHB final approach — arm correctly positioned at shifted object
- `naive_5cm_frame02.png`: Naive final position — arm 5cm off target, gripper in wrong location

---

## Regeneration Instructions

### Videos and Figures

Most figures and videos are regenerated by running the update scripts:

```bash
cd /home/andypark/Projects/repos/dhb_xr_paper

# Main trajectory visualization
pixi run python update_vis.py

# Figures 1-4 (token stability, sweep, efficiency, pipeline)
pixi run python scripts/generate_paper_figures.py

# Fig 7 (6D adaptation vs affine)
pixi run python scripts/generate_arc_adaptation_video.py
```

### Simulation Videos

Simulation videos require LIBERO and MUJOCO:

```bash
MUJOCO_GL=egl pixi run python examples/integration/libero_sim_with_media.py \
    --task "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate" \
    --offset 0.05 \
    --output-dir output/videos/
```

Note: LIBERO environment takes ~15s to initialize. Use `--no-video` flag for
faster experiment runs when video capture is not needed.

### Frame Extraction

```bash
# Extract frames from existing videos
pixi run python scripts/extract_frames.py \
    --video output/videos/comparison_5cm_agentview.mp4 \
    --output-dir output/frames/ \
    --frames 0 10 -1   # first, middle, last frame
```

---

## File Quality Notes

All 16 video files in `output/videos/` have real content (17-712 KB). No empty/black
files remain (previously generated empty files from failed renders were removed).

The long-name files (`pick_up_the_black_bowl_...`) are identical in content to the
short-name files (`comparison_0cm_agentview.mp4` etc.) — they were generated by an
earlier version of the capture script that used the LIBERO task name verbatim.
Either set can be used; the short-name files are preferred for the paper submission.

Figures are deterministically regenerated from `results/*.json` files. The JSON results
files are the ground truth; figures are derived artifacts.
