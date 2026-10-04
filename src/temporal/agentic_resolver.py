"""Agentic temporal reasoning component.

A lightweight deterministic agent that decides:
"Do I have enough evidence for this observation?"

If not, it requests previous/next temporal context and uses it to
resolve ambiguous states. This does NOT call an LLM — it uses
rule-based temporal context inspection.

The agent's key insight: single observations are often ambiguous.
By examining what came before and after, the system can make better
decisions about bed exits, state transitions, and uncertain periods.
"""
from __future__ import annotations

from typing import List, Optional, Protocol

from src.models import (
    ActivityState,
    AgentReasoningResult,
    ContextDirection,
    StatePrediction,
)
from src.config import PipelineConfig


# ---------------------------------------------------------------------------
# VLM Adapter Protocol (optional, pluggable)
# ---------------------------------------------------------------------------

class VisionReasoner(Protocol):
    """Protocol for vision-based reasoning (optional VLM adapter)."""
    def analyze_context(
        self,
        observation_index: int,
        observations: list,
        question: str,
    ) -> str:
        ...


# ---------------------------------------------------------------------------
# Rule-based implementation (default — no API keys needed)
# ---------------------------------------------------------------------------

class RuleBasedVisionReasoner:
    """Default deterministic reasoner. No external API required."""

    def analyze_context(
        self,
        observation_index: int,
        observations: list,
        question: str,
    ) -> str:
        return "Rule-based analysis: temporal context used for decision."


class AgenticContextResolver:
    """Resolves uncertain observations using temporal context.

    Implements a deterministic agentic loop:
    1. Check if current evidence is sufficient
    2. If not, inspect previous temporal window
    3. If still uncertain, inspect next temporal window
    4. Combine evidence and reach a conclusion
    """

    def __init__(
        self,
        config: PipelineConfig,
        reasoner: Optional[VisionReasoner] = None,
    ) -> None:
        self._config = config
        self._reasoner = reasoner or RuleBasedVisionReasoner()
        self._results: List[AgentReasoningResult] = []

    @property
    def results(self) -> List[AgentReasoningResult]:
        return list(self._results)

    def resolve(
        self,
        index: int,
        prediction: StatePrediction,
        timestamp: float,
        all_predictions: List[StatePrediction],
        all_timestamps: List[float],
    ) -> AgentReasoningResult:
        """Attempt to resolve an uncertain prediction using context.

        Called for predictions where confidence is low or the state
        is UNKNOWN.
        """
        result = AgentReasoningResult(
            observation_index=index,
            timestamp_sec=timestamp,
            original_state=prediction.state,
            confidence=prediction.confidence,
        )

        # Decide if resolution is needed
        needs_resolution = (
            prediction.state == ActivityState.UNKNOWN
            or prediction.confidence < self._config.state_confidence_threshold
        )

        if not needs_resolution:
            result.resolution = "confirmed"
            result.revised_state = prediction.state
            result.conclusion = prediction.state
            result.evidence_gathered = ["sufficient evidence — no context needed"]
            self._results.append(result)
            return result

        # --- Step 1: Inspect previous context ---
        result.uncertainty_reason = (
            f"state={prediction.state.value}, "
            f"confidence={prediction.confidence:.2f} is insufficient"
        )
        result.requested_context = ContextDirection.BOTH

        prev_states, prev_confs = self._get_context(
            index, all_predictions, all_timestamps, direction="previous"
        )
        next_states, next_confs = self._get_context(
            index, all_predictions, all_timestamps, direction="next"
        )

        evidence = []

        # Analyse previous context
        if prev_states:
            majority = self._majority_state(prev_states, prev_confs)
            evidence.append(
                f"previous context: majority state is {majority.value} "
                f"({len(prev_states)} observations)"
            )
        else:
            majority = None
            evidence.append("no previous context available")

        # Analyse next context
        if next_states:
            next_majority = self._majority_state(next_states, next_confs)
            evidence.append(
                f"next context: majority state is {next_majority.value} "
                f"({len(next_states)} observations)"
            )
        else:
            next_majority = None
            evidence.append("no next context available")

        # --- Step 2: Reach conclusion ---
        if majority is not None and next_majority is not None:
            if majority == next_majority:
                # Surrounding context agrees
                result.revised_state = majority
                result.conclusion = majority
                result.confidence = 0.65
                result.resolution = "revised"
                evidence.append(
                    f"both contexts agree on {majority.value} — adopting"
                )
            else:
                # Context disagrees — this may indicate a transition
                # Keep the higher-confidence context
                if prev_confs and next_confs:
                    avg_prev = sum(prev_confs) / len(prev_confs)
                    avg_next = sum(next_confs) / len(next_confs)
                    chosen = majority if avg_prev >= avg_next else next_majority
                else:
                    chosen = majority or next_majority
                result.revised_state = chosen
                result.conclusion = chosen
                result.confidence = 0.50
                result.resolution = "revised"
                evidence.append(
                    f"contexts disagree; chose {chosen.value} "
                    f"(higher avg confidence)"
                )
        elif majority is not None:
            result.revised_state = majority
            result.conclusion = majority
            result.confidence = 0.55
            result.resolution = "revised"
            evidence.append(
                f"only previous context available; using {majority.value}"
            )
        elif next_majority is not None:
            result.revised_state = next_majority
            result.conclusion = next_majority
            result.confidence = 0.50
            result.resolution = "revised"
            evidence.append(
                f"only next context available; using {next_majority.value}"
            )
        else:
            result.revised_state = ActivityState.UNKNOWN
            result.conclusion = ActivityState.UNKNOWN
            result.confidence = 0.2
            result.resolution = "unresolved"
            evidence.append("no context available — remaining UNKNOWN")

        result.evidence_gathered = evidence
        self._results.append(result)
        return result

    def _get_context(
        self,
        index: int,
        predictions: List[StatePrediction],
        timestamps: List[float],
        direction: str,
        window_sec: float = 5.0,
    ):
        """Get states and confidences from a temporal window."""
        if not predictions or not timestamps:
            return [], []

        current_ts = timestamps[index] if index < len(timestamps) else 0.0
        states = []
        confs = []

        if direction == "previous":
            for i in range(index - 1, -1, -1):
                if current_ts - timestamps[i] > window_sec:
                    break
                if predictions[i].state != ActivityState.UNKNOWN:
                    states.append(predictions[i].state)
                    confs.append(predictions[i].confidence)
        else:  # next
            for i in range(index + 1, len(predictions)):
                if timestamps[i] - current_ts > window_sec:
                    break
                if predictions[i].state != ActivityState.UNKNOWN:
                    states.append(predictions[i].state)
                    confs.append(predictions[i].confidence)

        return states, confs

    def _majority_state(
        self,
        states: List[ActivityState],
        confidences: List[float],
    ) -> ActivityState:
        """Confidence-weighted majority vote."""
        weights: dict[ActivityState, float] = {}
        for s, c in zip(states, confidences):
            weights[s] = weights.get(s, 0.0) + c
        if not weights:
            return ActivityState.UNKNOWN
        return max(weights, key=lambda s: weights[s])

    def reset(self) -> None:
        self._results.clear()
