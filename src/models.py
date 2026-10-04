"""Domain models for the elderly activity monitoring system.

Defines the core enumerations, data structures, and schemas used
throughout the temporal reasoning pipeline.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core enumerations
# ---------------------------------------------------------------------------

class ActivityState(str, enum.Enum):
    """Primary activity states for the elderly person."""
    LYING_IN_BED = "LYING_IN_BED"
    SITTING_ON_BED = "SITTING_ON_BED"
    SITTING_OUTSIDE_BED = "SITTING_OUTSIDE_BED"
    STANDING = "STANDING"
    WALKING = "WALKING"
    OUT_OF_BED = "OUT_OF_BED"
    UNKNOWN = "UNKNOWN"


class BedStatus(str, enum.Enum):
    """Whether the person is currently in/on bed or not."""
    IN_BED = "IN_BED"
    ON_BED_EDGE = "ON_BED_EDGE"
    OUT_OF_BED = "OUT_OF_BED"
    UNKNOWN = "UNKNOWN"


class AlertLevel(str, enum.Enum):
    """Safety decision for each observation/event."""
    NORMAL = "NORMAL"
    MONITOR = "MONITOR"
    ALERT = "ALERT"


class BedEventType(str, enum.Enum):
    """Types of bed-related events."""
    BED_EXIT = "BED_EXIT"
    BED_RETURN = "BED_RETURN"


class ContextDirection(str, enum.Enum):
    """Direction of temporal context requested by agentic reasoner."""
    PREVIOUS = "PREVIOUS"
    NEXT = "NEXT"
    BOTH = "BOTH"
    NONE = "NONE"


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

@dataclass
class TemporalObservation:
    """Normalized per-detection observation from the perception pipeline.

    All fields use Optional where the existing pipeline may not provide
    a value. The system must never crash because any field is None.
    """
    # Identity / timing
    frame_index: int
    timestamp_sec: float
    track_id: Optional[int] = None
    detection_index: int = 0

    # Detection confidence
    person_confidence: float = 0.0

    # Bounding box (pixel coordinates)
    bbox: Optional[tuple] = None           # (x1, y1, x2, y2)
    bbox_center: Optional[tuple] = None    # (cx, cy)
    bbox_width: float = 0.0
    bbox_height: float = 0.0
    bbox_area_ratio: float = 0.0           # area / frame_area

    # Pose keypoints (COCO-17 format, pixel coordinates)
    pose_keypoints: Optional[List[List[float]]] = None
    pose_confidences: Optional[List[float]] = None
    pose_quality: str = "unknown"

    # Named body part coordinates (pixel)
    shoulder_left: Optional[tuple] = None
    shoulder_right: Optional[tuple] = None
    shoulder_mid: Optional[tuple] = None
    hip_left: Optional[tuple] = None
    hip_right: Optional[tuple] = None
    hip_mid: Optional[tuple] = None
    knee_left: Optional[tuple] = None
    knee_right: Optional[tuple] = None
    ankle_left: Optional[tuple] = None
    ankle_right: Optional[tuple] = None

    # Body geometry
    torso_angle_deg: Optional[float] = None
    body_aspect_ratio: Optional[float] = None
    body_orientation: Optional[str] = None  # "vertical", "horizontal", "diagonal"

    # Motion (relative to previous observation for same track)
    motion_displacement: float = 0.0
    motion_speed: float = 0.0              # pixels per second (normalized)

    # Bed relation
    bed_region: Optional[tuple] = None     # (x1, y1, x2, y2) pixel
    bbox_bed_overlap: float = 0.0
    keypoints_inside_bed: int = 0
    distance_to_bed: Optional[float] = None
    near_bed: bool = False
    on_bed: bool = False
    bed_relation: str = "not_configured"

    # Quality
    visibility_quality: str = "unknown"    # "good", "partial", "poor", "missing"

    # Raw evidence
    raw_evidence: Optional[Dict[str, Any]] = None

    # Frame dimensions (for normalisation)
    frame_width: int = 0
    frame_height: int = 0


# ---------------------------------------------------------------------------
# State prediction
# ---------------------------------------------------------------------------

@dataclass
class StatePrediction:
    """Result of raw state classification for a single observation."""
    state: ActivityState
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    bed_status: BedStatus = BedStatus.UNKNOWN


# ---------------------------------------------------------------------------
# State transition
# ---------------------------------------------------------------------------

@dataclass
class StateTransition:
    """Confirmed state transition event."""
    previous_state: ActivityState
    new_state: ActivityState
    start_timestamp: float
    confirmed_timestamp: float
    confidence: float
    reason: str = ""
    evidence: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bed event
# ---------------------------------------------------------------------------

@dataclass
class BedEvent:
    """A confirmed bed exit or return event."""
    event_type: BedEventType
    start_time: float
    confirmed_time: float
    previous_state: ActivityState
    current_state: ActivityState
    confidence: float
    decision: AlertLevel = AlertLevel.NORMAL
    evidence: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agentic reasoning
# ---------------------------------------------------------------------------

@dataclass
class ContextRequest:
    """Request for additional temporal context."""
    direction: ContextDirection
    reason: str
    observation_index: int = 0
    window_sec: float = 5.0


@dataclass
class AgentReasoningResult:
    """Result of agentic temporal reasoning."""
    observation_index: int
    timestamp_sec: float
    uncertainty_reason: str = ""
    requested_context: ContextDirection = ContextDirection.NONE
    evidence_gathered: List[str] = field(default_factory=list)
    conclusion: Optional[ActivityState] = None
    original_state: Optional[ActivityState] = None
    revised_state: Optional[ActivityState] = None
    confidence: float = 0.0
    resolution: str = ""  # "confirmed", "revised", "unresolved"


# ---------------------------------------------------------------------------
# Alert event
# ---------------------------------------------------------------------------

@dataclass
class AlertEvent:
    """A safety alert or monitoring event."""
    timestamp: float
    level: AlertLevel
    rule: str
    reason: str
    confidence: float
    state: Optional[ActivityState] = None
    trigger_event: Optional[str] = None


# ---------------------------------------------------------------------------
# Timeline interval
# ---------------------------------------------------------------------------

@dataclass
class TimelineInterval:
    """A compressed period of stable activity."""
    start_sec: float
    end_sec: float
    duration_sec: float
    state: ActivityState
    mean_confidence: float
    observation_count: int = 0


# ---------------------------------------------------------------------------
# Activity summary
# ---------------------------------------------------------------------------

@dataclass
class ActivitySummary:
    """Aggregated duration and event statistics."""
    observation_duration_sec: float = 0.0
    activity_durations: Dict[str, float] = field(default_factory=dict)
    activity_durations_human: Dict[str, str] = field(default_factory=dict)
    bed_exit_count: int = 0
    bed_return_count: int = 0
    total_in_bed_sec: float = 0.0
    total_out_of_bed_sec: float = 0.0
    longest_out_of_bed_period_sec: float = 0.0
    final_state: Optional[str] = None


# ---------------------------------------------------------------------------
# Evaluation structures
# ---------------------------------------------------------------------------

@dataclass
class ClassMetrics:
    """Per-class precision/recall/F1."""
    label: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0


@dataclass
class EventMetrics:
    """Event-level metrics for bed exits/returns."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


@dataclass
class FailureCase:
    """A documented failure case."""
    timestamp_sec: float
    window_sec: float = 5.0
    expected: str = ""
    predicted: str = ""
    reason: str = ""
    likely_cause: str = ""
    suggested_improvement: str = ""
