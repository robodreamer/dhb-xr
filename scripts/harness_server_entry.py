#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "vla-eval",
#     "dhb-xr",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../vendor/vla-evaluation-harness", editable = true }
# dhb-xr = { path = "..", editable = true }
# ///
"""Repo-root entrypoint for the AllenAI harness server.

This wrapper exists because the harness smoke runner executes the configured
script through `uv`. If the script path lives under `src/`, `uv` may treat that
directory as the project root and fail to install `dhb_xr` from the real repo
root. Keeping the entrypoint under `scripts/` avoids that mis-detection.
"""

from dhb_xr.integration.vla.harness_server import DHBModelServer, HAS_HARNESS


def main() -> None:
    if not HAS_HARNESS:
        raise RuntimeError("vla_eval is not installed in the harness environment")

    from vla_eval.model_servers.serve import run_server

    run_server(DHBModelServer)


if __name__ == "__main__":
    main()
