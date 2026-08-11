# DHB-XR brand assets

<p style="text-align:center; margin:1rem 0 2rem;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/brand/dhb-xr-wordmark-dark.svg">
    <img src="../assets/brand/dhb-xr-wordmark-light.svg" width="380" alt="DHB-XR">
  </picture>
</p>

The logo uses responsive masters so the mechanical detail never competes with
letter recognition.

| Use | Asset |
| --- | --- |
| README or large light surface | `dhb-xr-wordmark-light.svg` |
| Large dark surface | `dhb-xr-wordmark-dark.svg` |
| Compact light lockup | `dhb-xr-wordmark-compact-light.svg` |
| Compact dark lockup | `dhb-xr-wordmark-compact-dark.svg` |
| Standalone display mark | `dhb-xr-mark-display-light.svg` or `-dark.svg` |
| Standalone mark below 64 px | `dhb-xr-mark-compact-light.svg` or `-dark.svg` |
| MkDocs navigation | `dhb-xr-logo-nav.svg` |
| Browser or package tile | `dhb-xr-favicon.svg` |

Use the display master at 64 px and above. Below 64 px, use the compact master,
which renders the Householder `H` as one color. At 24 px, use the favicon's
simplified geometric construction. Do not add reflection ticks, pose axes, or
extra trajectory echoes to the production mark.

The palette is cobalt (`#3478F6`), orange (`#F97336`), and slate
(`#1E293B` / `#475569`). Dark-surface variants use lighter equivalents chosen
for contrast.
