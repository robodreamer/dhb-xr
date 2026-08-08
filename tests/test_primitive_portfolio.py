"""Tests for matched source comparison under identical primitive gates."""

import numpy as np

from dhb_xr.primitives import (
    AnalyticPrimitiveSource,
    EndpointGate,
    EvidenceLevel,
    ObservationEpochGate,
    PrimitiveEvidence,
    PrimitiveFamily,
    PrimitivePortfolio,
    PrimitiveRequest,
    SphereClearanceGate,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
EVIDENCE = PrimitiveEvidence(EvidenceLevel.PLANNED, "portfolio-test")


def _request(epoch: int = 3):
    return PrimitiveRequest(
        "move",
        np.zeros(3),
        IDENTITY,
        np.array([1.0, 0.0, 0.0]),
        IDENTITY,
        samples=51,
        duration_s=1.0,
        observation_epoch=epoch,
    )


def test_via_point_clears_obstacle_that_rejects_direct_minimum_jerk():
    direct = AnalyticPrimitiveSource(PrimitiveFamily.MINIMUM_JERK, EVIDENCE)
    detour = AnalyticPrimitiveSource(
        PrimitiveFamily.VIA_POINT,
        EVIDENCE,
        parameters={
            "positions": [[0.5, 0.35, 0.0]],
            "quaternions_wxyz": [IDENTITY],
            "samples_per_segment": 26,
        },
    )
    portfolio = PrimitivePortfolio(
        (direct, detour),
        (
            ObservationEpochGate(),
            EndpointGate(),
            SphereClearanceGate(
                centers=np.array([[0.5, 0.0, 0.0]]),
                radii_m=np.array([0.15]),
                margin_m=0.02,
            ),
        ),
    )

    result = portfolio.evaluate(_request())
    accepted_sources = [candidate.source_id for candidate in result.accepted]

    assert accepted_sources == ["analytic:via_point"]
    direct_row = next(
        candidate
        for candidate in result.candidates
        if candidate.source_id == "analytic:minimum_jerk"
    )
    assert not direct_row.accepted
    assert direct_row.gates[-1].reason == "TCP sample intersects sphere"


def test_source_failure_is_recorded_without_blocking_other_sources():
    class FailingSource:
        source_id = "failure:test"

        def propose(self, request):
            del request
            raise RuntimeError("expected proposal failure")

    direct = AnalyticPrimitiveSource(PrimitiveFamily.MINIMUM_JERK, EVIDENCE)
    result = PrimitivePortfolio(
        (FailingSource(), direct),
        (ObservationEpochGate(), EndpointGate()),
    ).evaluate(_request())

    assert result.sources[0].error == "RuntimeError: expected proposal failure"
    assert result.sources[1].error is None
    assert len(result.accepted) == 1
