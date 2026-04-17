# Draft: SmolVLA-Based DHB Action Head Results and Discussion

## Option A: Paper-Style Prose

### SmolVLA-Based DHB Action Head Validation

We first used SmolVLA as a controlled validation environment for the DHB action head before scaling to larger VLA frameworks. The purpose of this phase was not to maximize final task performance, but to verify that the DHB action head was wired correctly, that the current robot state was incorporated at inference time, and that the downstream execution path through invariant decoding and `embodik` inverse kinematics was reliable.

This validation phase established three positive results. First, the integration bug in which the DHB action head could be queried without the current end-effector state was removed, and regression tests were added so that this failure mode cannot silently reappear. Second, the previous persistent end-effector rotation drift disappeared after routing inference through the adapter-based state-aware path. Third, the `embodik` execution stage proved stable: across the smoke, 50-epoch, and 200-epoch checkpoints, inverse-kinematics errors remained near `0.47 mm` mean and `0.50 mm` max, indicating that the IK solver was not the dominant source of task failure.

However, the SmolVLA experiments also showed a clear limitation. Increasing training duration substantially improved supervised training loss but did not improve closed-loop task success. The `10`-epoch smoke checkpoint achieved `2/3` success at `0 cm` and `1/3` at `3 cm`, while the `50`-epoch and `200`-epoch checkpoints both remained at `1/3` and `1/3`, despite the best loss improving from `0.94095` to `0.06337`. This disconnect indicates that lower invariant-prediction loss alone is not sufficient to improve embodied execution quality in this setting.

Per-demo diagnostics further clarified the failure mode. The apparent seed dependence was actually a demo dependence: `seed 0`, `seed 1`, and `seed 2` corresponded to different underlying demonstrations (`demo_0`, `demo_1`, and `demo_10`). The successful case, `demo_0`, consistently terminated closer to the plate center in the final placement phase, whereas the failing cases, `demo_1` and `demo_10`, systematically under-reached in `Y` and remained slightly high in `Z`. For example, at `0 cm`, the successful `demo_0` run finished at approximately `ee_minus_plate = [0.018, -0.020, 0.072]`, while the failing `demo_1` and `demo_10` runs finished at approximately `[0.001, -0.047, 0.075]` and `[-0.006, -0.055, 0.073]`, respectively. The gripper close and reopen events remained broadly consistent across demos, suggesting that the dominant error is placement margin rather than gross timing failure.

Taken together, these results are useful in two ways. On the positive side, they validate the DHB action head plumbing, confirm that the `embodik` path is a reliable execution benchmark, and show that the remaining error is structured rather than random. On the negative side, they show that simply extending SmolVLA training is not an efficient path to better closed-loop performance. The main remaining limitation is not the invariant representation itself, nor the IK layer, but the policy's ability to reach the correct placement region robustly across specific demonstrations.

For this reason, we treat the SmolVLA phase as complete for debugging and validation purposes and use the best smoke checkpoint as the baseline for subsequent frameworks. Future improvements should focus on stronger placement-sensitive supervision, better checkpoint selection criteria, or improved upstream policy features rather than additional SmolVLA epochs alone.

## Option B: LaTeX-Friendly Draft

```tex
\paragraph{SmolVLA-based DHB action head validation.}
We first used SmolVLA as a controlled validation environment for the DHB action head before scaling to larger VLA frameworks. The goal of this phase was not to maximize task success, but to verify that the DHB action head was wired correctly, that the current robot state was incorporated at inference time, and that the downstream execution path through invariant decoding and \texttt{embodik} inverse kinematics was reliable.

This phase established three positive results. First, the integration bug in which the DHB action head could be queried without the current end-effector state was removed, and regression tests were added so that this failure mode cannot silently reappear. Second, the previous persistent end-effector rotation drift disappeared after routing inference through the adapter-based state-aware path. Third, the \texttt{embodik} execution stage proved stable: across the smoke, 50-epoch, and 200-epoch checkpoints, inverse-kinematics errors remained near 0.47\,mm mean and 0.50\,mm max, indicating that the IK solver was not the dominant source of task failure.

However, increasing training duration substantially improved supervised training loss without improving closed-loop task success. The 10-epoch smoke checkpoint achieved 2/3 success at 0\,cm and 1/3 at 3\,cm, while the 50-epoch and 200-epoch checkpoints both remained at 1/3 and 1/3, despite the best loss improving from 0.94095 to 0.06337. This disconnect indicates that lower invariant-prediction loss alone is not sufficient to improve embodied execution quality in this setting.

Per-demo diagnostics clarified the failure mode. The apparent seed dependence was actually a demo dependence: seed 0, seed 1, and seed 2 corresponded to different underlying demonstrations (\texttt{demo\_0}, \texttt{demo\_1}, and \texttt{demo\_10}). The successful case, \texttt{demo\_0}, consistently terminated closer to the plate center in the final placement phase, whereas the failing cases, \texttt{demo\_1} and \texttt{demo\_10}, systematically under-reached in $Y$ and remained slightly high in $Z$. At 0\,cm, the successful \texttt{demo\_0} run finished at approximately $\texttt{ee\_minus\_plate}=[0.018,-0.020,0.072]$, while the failing \texttt{demo\_1} and \texttt{demo\_10} runs finished at approximately $[0.001,-0.047,0.075]$ and $[-0.006,-0.055,0.073]$, respectively. The gripper close and reopen events remained broadly consistent across demos, suggesting that the dominant error is placement margin rather than gross timing failure.

Taken together, these results validate the DHB action head plumbing, confirm that the \texttt{embodik} path is a reliable execution benchmark, and show that the remaining error is structured rather than random. At the same time, they show that simply extending SmolVLA training is not an efficient path to better closed-loop performance. We therefore treat the SmolVLA phase as complete for debugging and validation purposes and use the best smoke checkpoint as the baseline for subsequent frameworks.
```

## Short Figure Caption Candidate

SmolVLA was sufficient to validate the DHB action head and `embodik` execution stack, but not sufficient to solve the remaining placement-robustness problem. Longer training sharply reduced prediction loss without improving closed-loop success, and per-demo diagnostics showed that the failure cases systematically under-reached in `Y` at final placement.
