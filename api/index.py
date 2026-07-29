import os
import json
import math
import re
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import groq
from groq import Groq

# Initialize local environment attributes if executing outside isolated production nodes
load_dotenv()

app = Flask(__name__, template_folder='../templates')

# Initialize Groq client securely using environment properties
api_key = os.environ.get("GROQ_API_KEY")
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
STRICT_SCHEMA_GROQ_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
requested_groq_model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
GROQ_MODEL = (
    requested_groq_model
    if requested_groq_model in STRICT_SCHEMA_GROQ_MODELS
    else DEFAULT_GROQ_MODEL
)
GROQ_MODEL_OVERRIDE_IGNORED = requested_groq_model != GROQ_MODEL
GROQ_REQUEST_TIMEOUT_SECONDS = 10.0
groq_client = (
    Groq(
        api_key=api_key,
        timeout=GROQ_REQUEST_TIMEOUT_SECONDS,
        max_retries=1,
    )
    if api_key
    else None
)

# --------------------------------------------------------------------------------------
# POST-PROCESSING / SAFETY LAYER
#
# The Groq model is instructed (via the system prompt) to stay within these numeric
# ranges, but LLM output can still occasionally drift, hallucinate, or omit fields.
# The constants and sanitize_kinematic_matrix() below are a defense-in-depth layer that
# clips or discards anything out of bounds before a response ever leaves the backend, so
# a bad number from the model can never reach (and destabilize) the client renderer.
#
# These mirror the client-side LIP (Linear Inverted Pendulum) locomotion constants that
# already exist in index.html, so both layers agree on what "physically sane" means.
# --------------------------------------------------------------------------------------
VELOCITY_MIN, VELOCITY_MAX = 0.0, 3.0
STEP_FREQUENCY_MIN, STEP_FREQUENCY_MAX = 0.4, 3.0
CAPTURE_X_MIN, CAPTURE_X_MAX = -0.3, 0.3
CAPTURE_Z_MIN, CAPTURE_Z_MAX = -0.1, 0.35
LATERAL_SHIFT_MIN, LATERAL_SHIFT_MAX = 0.0, 2.0
ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX = 0.0, 360.0
TORQUE_MIN, TORQUE_MAX = -150.0, 150.0
STABILITY_MIN, STABILITY_MAX = 0.0, 100.0
COM_Y_MIN, COM_Y_MAX = -0.6, 0.3
COM_XZ_MIN, COM_XZ_MAX = -0.3, 0.3
PELVIC_PITCH_MIN, PELVIC_PITCH_MAX = -0.35, 0.35
PELVIC_ROLL_MIN, PELVIC_ROLL_MAX = -0.3, 0.3

VALID_MOVEMENT_STATES = {"idle", "walk", "run", "crouch", "jump", "climb", "sidestep", "brace", "slide"}
VALID_OBSTACLE_ACTIONS = {
    "none", "jump", "climb", "crouch", "sidestep_left", "sidestep_right", "push_through", "brace", "slide"
}
VALID_ROTATION_TURNS = {"clockwise", "counterclockwise", "none"}
VALID_FALL_DIRECTIONS = {"forward", "backward", "left", "right", "none"}
VALID_TERRAINS = {"flat", "ice", "rubble", "mud", "volcanic", "snow", "collapsing", "mountains"}
VALID_OBSTACLE_TYPES = {"boulder", "crevice", "ice_patch", "debris"}
OBSTACLE_CLEAR_ACTIONS = {
    "boulder": {"jump", "climb"},
    "crevice": {"jump"},
    "ice_patch": {"crouch", "sidestep_left", "sidestep_right", "slide"},
    "debris": {"climb", "sidestep_left", "sidestep_right", "push_through", "slide"},
}
ACTION_MOVEMENT_STATES = {
    "jump": "jump",
    "climb": "climb",
    "crouch": "crouch",
    "sidestep_left": "sidestep",
    "sidestep_right": "sidestep",
    "push_through": "brace",
    "slide": "slide",
}

KINEMATIC_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "movement_state": {"type": "string", "enum": sorted(VALID_MOVEMENT_STATES)},
        "velocity": {"type": "number", "minimum": VELOCITY_MIN, "maximum": VELOCITY_MAX},
        "center_of_mass_shift": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "minimum": COM_XZ_MIN, "maximum": COM_XZ_MAX},
                "y": {"type": "number", "minimum": COM_Y_MIN, "maximum": COM_Y_MAX},
                "z": {"type": "number", "minimum": COM_XZ_MIN, "maximum": COM_XZ_MAX},
            },
            "required": ["x", "y", "z"],
            "additionalProperties": False,
        },
        "step_frequency": {
            "type": "number",
            "minimum": STEP_FREQUENCY_MIN,
            "maximum": STEP_FREQUENCY_MAX,
        },
        "target_pelvic_quaternion": {
            "type": "object",
            "properties": {
                "w": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                "x": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                "y": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                "z": {"type": "number", "minimum": -1.0, "maximum": 1.0},
            },
            "required": ["w", "x", "y", "z"],
            "additionalProperties": False,
        },
        "calculated_capture_point": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "minimum": CAPTURE_X_MIN, "maximum": CAPTURE_X_MAX},
                "z": {"type": "number", "minimum": CAPTURE_Z_MIN, "maximum": CAPTURE_Z_MAX},
            },
            "required": ["x", "z"],
            "additionalProperties": False,
        },
        "torque_compensation": {
            "type": "object",
            "properties": {
                "ankle": {"type": "number", "minimum": TORQUE_MIN, "maximum": TORQUE_MAX},
                "knee": {"type": "number", "minimum": TORQUE_MIN, "maximum": TORQUE_MAX},
                "waist": {"type": "number", "minimum": TORQUE_MIN, "maximum": TORQUE_MAX},
            },
            "required": ["ankle", "knee", "waist"],
            "additionalProperties": False,
        },
        "stability_projection": {
            "type": "number",
            "minimum": STABILITY_MIN,
            "maximum": STABILITY_MAX,
        },
        "obstacle_action": {"type": "string", "enum": sorted(VALID_OBSTACLE_ACTIONS)},
        "lateral_shift": {
            "type": "number",
            "minimum": LATERAL_SHIFT_MIN,
            "maximum": LATERAL_SHIFT_MAX,
        },
        "rotation": {
            "type": "object",
            "properties": {
                "turn": {"type": "string", "enum": sorted(VALID_ROTATION_TURNS)},
                "angle_degrees": {
                    "type": "number",
                    "minimum": ROTATION_ANGLE_MIN,
                    "maximum": ROTATION_ANGLE_MAX,
                },
            },
            "required": ["turn", "angle_degrees"],
            "additionalProperties": False,
        },
        "trigger_fall": {"type": "boolean"},
        "fall_direction": {"type": "string", "enum": sorted(VALID_FALL_DIRECTIONS)},
        "biomechanical_rationale": {"type": "string"},
    },
    "required": [
        "movement_state",
        "velocity",
        "center_of_mass_shift",
        "step_frequency",
        "target_pelvic_quaternion",
        "calculated_capture_point",
        "torque_compensation",
        "stability_projection",
        "obstacle_action",
        "lateral_shift",
        "rotation",
        "trigger_fall",
        "fall_direction",
        "biomechanical_rationale",
    ],
    "additionalProperties": False,
}

KINEMATIC_SYSTEM_PROMPT = """
You are TerraWalk's stateless high-level kinematic controller. Convert only the current
operator command and telemetry snapshot into the enforced response schema. Return no
conversation or markdown.

Continuity:
- Treat every request independently. Never carry a hazard or stopped state from an earlier call.
- After clearing a hazard, keep positive forward velocity unless the command explicitly says stop,
  idle, hold, or brace.
- If wording is ambiguous, choose a calm walk near 0.8 m/s with an upright pelvis; when a hazard is
  active and the command does not address it, brace without clearing it.

Hazards:
- boulder: jump or climb.
- crevice: jump.
- ice_patch: crouch, sidestep_left, sidestep_right, or slide.
- debris: climb, sidestep_left, sidestep_right, push_through, or slide.
- brace never clears a hazard. With no active hazard, obstacle_action must be none.
- "walk/go around the obstacle" means a 45-degree walking detour (clockwise unless left is stated),
  not a 180-degree reversal. For that request use movement_state walk, positive velocity,
  obstacle_action none, and the requested 45-degree rotation. The browser completes the bypass and
  returns to the original obstacle route after passing. "turn/spin around" means 180 degrees.
- An explicit request to retreat, back away, move away from hazards, or reverse direction must use
  movement_state walk, positive velocity, obstacle_action none, and a 180-degree rotation.
- Jump and slide may also be standalone movement states when there is no hazard.

Fall commands:
- trigger_fall is true only for an explicit request to fall, collapse, topple, faint, lie down, or
  play dead. Choose the stated direction or forward by default. Then use brace, zero velocity,
  obstacle_action none, and no rotation. Otherwise trigger_fall is false and fall_direction is none.

Pelvic SIP:
- The quaternion is command posture only; never encode terrain incline or arbitrary yaw. The browser
  measures terrain slope separately. Idle and steady walking are upright. Running, climbing, sliding,
  and pushing may use a modest negative pitch. Roll is allowed only for a sidestep.
- Choose pitch p in [-0.35, 0.35] radians and roll r in [-0.3, 0.3], then output
  w=cos(p/2)cos(r/2), x=sin(p/2)cos(r/2), y=sin(p/2)sin(r/2), z=cos(p/2)sin(r/2).
- Project the capture point in the robot frame: x is lateral [-0.3, 0.3] m and z is forward
  [-0.1, 0.35] m. Faster motion projects farther forward; idle/brace stays near zero.

Use physically calm values inside the schema bounds. lateral_shift is 1-2 m only for sidesteps and
zero otherwise. rotation angle is zero when turn is none. Keep the rationale concise.
""".strip()


def _safe_float(value, default=0.0):
    """Coerces a value to a finite float. Falls back to `default` for anything
    non-numeric, missing, NaN, or +/-Infinity (guards against LLM hallucinations)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def _canonicalize_pelvic_quaternion(quat, movement_state, obstacle_action):
    """Normalizes the model quaternion, removes unsupported yaw/twist, clamps the
    command-driven SIP pitch/roll, and rebuilds the exact canonical quaternion the
    browser expects. Terrain compensation is intentionally excluded because the
    client measures the real local slope independently every frame."""
    if not isinstance(quat, dict):
        quat = {}

    qw = _clip(_safe_float(quat.get("w"), 1.0), -1.0, 1.0)
    qx = _clip(_safe_float(quat.get("x"), 0.0), -1.0, 1.0)
    qy = _clip(_safe_float(quat.get("y"), 0.0), -1.0, 1.0)
    qz = _clip(_safe_float(quat.get("z"), 0.0), -1.0, 1.0)

    magnitude = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if magnitude < 1e-6:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    else:
        qw, qx, qy, qz = qw / magnitude, qx / magnitude, qy / magnitude, qz / magnitude

    # Remove rotation around the vertical Y axis with a swing/twist decomposition.
    # This prevents a pure (or near-180-degree) hallucinated yaw from being
    # misinterpreted as a large pitch by the Euler extraction below.
    twist_magnitude = math.sqrt(qw * qw + qy * qy)
    if twist_magnitude >= 1e-6:
        twist_w = qw / twist_magnitude
        twist_y = qy / twist_magnitude
        swing_w = qw * twist_w + qy * twist_y
        swing_x = qx * twist_w + qz * twist_y
        swing_y = qy * twist_w - qw * twist_y
        swing_z = qz * twist_w - qx * twist_y
        qw, qx, qy, qz = swing_w, swing_x, swing_y, swing_z
        swing_magnitude = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if swing_magnitude >= 1e-6:
            qw, qx, qy, qz = (
                qw / swing_magnitude,
                qx / swing_magnitude,
                qy / swing_magnitude,
                qz / swing_magnitude,
            )

    # The supported posture is Rz(roll) * Rx(pitch). Recover only those axes,
    # clamp them by movement state, then rebuild an exact canonical quaternion.
    pitch = math.atan2(2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy))
    roll = math.atan2(2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qy * qy + qz * qz))
    pitch = _clip(pitch, PELVIC_PITCH_MIN, PELVIC_PITCH_MAX)
    roll = _clip(roll, PELVIC_ROLL_MIN, PELVIC_ROLL_MAX)

    state_pitch_ranges = {
        "idle": (0.0, 0.0),
        "walk": (0.0, 0.0),
        "run": (-0.24, 0.04),
        "crouch": (-0.08, 0.15),
        "jump": (-0.22, 0.08),
        "climb": (-0.30, 0.02),
        "sidestep": (0.0, 0.0),
        "brace": (-0.04, 0.18),
        "slide": (-0.28, 0.02),
    }
    pitch_lo, pitch_hi = state_pitch_ranges.get(movement_state, (-0.04, 0.04))

    if obstacle_action in {"climb", "push_through"}:
        pitch_lo, pitch_hi = -0.30, 0.02
    elif obstacle_action == "brace":
        pitch_lo, pitch_hi = -0.04, 0.18

    pitch = _clip(pitch, pitch_lo, pitch_hi)
    if abs(pitch) < 0.01:
        pitch = 0.0

    if movement_state == "sidestep" or obstacle_action in {"sidestep_left", "sidestep_right"}:
        roll = _clip(roll, -0.24, 0.24)
        if abs(roll) < 0.01:
            roll = 0.0
    else:
        # Random lateral pelvis tilt on flat ground came from allowing arbitrary
        # model roll in states where no sideways maneuver exists.
        roll = 0.0

    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    return {
        "w": cp * cr,
        "x": sp * cr,
        "y": sp * sr,
        "z": cp * sr,
    }


def sanitize_kinematic_matrix(matrix, obstacle_type=None):
    """Post-processes the raw JSON object returned by the Kinematic Brain: clips every
    numeric field into its physically valid range, drops/repairs malformed nested
    objects, canonicalizes the pelvic quaternion, and falls back to safe defaults for
    invalid or missing enum values. This runs before every response reaches the client."""
    if not isinstance(matrix, dict):
        matrix = {}

    sanitized = {}

    movement_state = matrix.get("movement_state")
    movement_state = movement_state if movement_state in VALID_MOVEMENT_STATES else "idle"

    obstacle_action = matrix.get("obstacle_action")
    obstacle_action = obstacle_action if obstacle_action in VALID_OBSTACLE_ACTIONS else "none"
    if obstacle_type not in VALID_OBSTACLE_TYPES:
        obstacle_type = None
        obstacle_action = "none"

    trigger_fall = matrix.get("trigger_fall") is True
    valid_clear_action = bool(
        obstacle_type and obstacle_action in OBSTACLE_CLEAR_ACTIONS.get(obstacle_type, set())
    )
    if trigger_fall:
        movement_state = "brace"
        obstacle_action = "none"
        valid_clear_action = False
    elif valid_clear_action:
        movement_state = ACTION_MOVEMENT_STATES[obstacle_action]
    elif obstacle_action == "brace":
        movement_state = "brace"

    sanitized["movement_state"] = movement_state
    sanitized["obstacle_action"] = obstacle_action

    velocity = _clip(_safe_float(matrix.get("velocity"), 0.0), VELOCITY_MIN, VELOCITY_MAX)
    step_frequency = _clip(
        _safe_float(matrix.get("step_frequency"), STEP_FREQUENCY_MIN),
        STEP_FREQUENCY_MIN,
        STEP_FREQUENCY_MAX,
    )
    if trigger_fall or (movement_state in {"idle", "brace"} and obstacle_action != "push_through"):
        velocity = 0.0
    elif valid_clear_action:
        if obstacle_action == "crouch":
            minimum_velocity = 0.35
        elif obstacle_action.startswith("sidestep"):
            minimum_velocity = 0.6
        else:
            minimum_velocity = 1.2
        velocity = max(velocity, minimum_velocity)
        step_frequency = max(step_frequency, 1.4)

    sanitized["velocity"] = velocity
    sanitized["step_frequency"] = step_frequency

    com_shift = matrix.get("center_of_mass_shift")
    if not isinstance(com_shift, dict):
        com_shift = {}
    sanitized["center_of_mass_shift"] = {
        "x": _clip(_safe_float(com_shift.get("x"), 0.0), COM_XZ_MIN, COM_XZ_MAX),
        "y": _clip(_safe_float(com_shift.get("y"), 0.0), COM_Y_MIN, COM_Y_MAX),
        "z": _clip(_safe_float(com_shift.get("z"), 0.0), COM_XZ_MIN, COM_XZ_MAX),
    }

    sanitized["target_pelvic_quaternion"] = _canonicalize_pelvic_quaternion(
        matrix.get("target_pelvic_quaternion"), movement_state, obstacle_action
    )

    capture = matrix.get("calculated_capture_point")
    if not isinstance(capture, dict):
        capture = {}
    sanitized["calculated_capture_point"] = {
        "x": _clip(_safe_float(capture.get("x"), 0.0), CAPTURE_X_MIN, CAPTURE_X_MAX),
        "z": _clip(_safe_float(capture.get("z"), 0.0), CAPTURE_Z_MIN, CAPTURE_Z_MAX),
    }

    torque = matrix.get("torque_compensation")
    if not isinstance(torque, dict):
        torque = {}
    sanitized["torque_compensation"] = {
        "ankle": _clip(_safe_float(torque.get("ankle"), 0.0), TORQUE_MIN, TORQUE_MAX),
        "knee": _clip(_safe_float(torque.get("knee"), 0.0), TORQUE_MIN, TORQUE_MAX),
        "waist": _clip(_safe_float(torque.get("waist"), 0.0), TORQUE_MIN, TORQUE_MAX),
    }

    sanitized["stability_projection"] = _clip(
        _safe_float(matrix.get("stability_projection"), 90.0), STABILITY_MIN, STABILITY_MAX
    )

    if obstacle_action in {"sidestep_left", "sidestep_right"}:
        lateral_shift = _clip(
            _safe_float(matrix.get("lateral_shift"), 1.4), 1.0, LATERAL_SHIFT_MAX
        )
    else:
        lateral_shift = 0.0
    sanitized["lateral_shift"] = lateral_shift

    rotation = matrix.get("rotation")
    if not isinstance(rotation, dict):
        rotation = {}
    turn = rotation.get("turn")
    turn = turn if turn in VALID_ROTATION_TURNS else "none"
    angle_degrees = _clip(_safe_float(rotation.get("angle_degrees"), 0.0), ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX)
    if turn == "none":
        angle_degrees = 0.0
    sanitized["rotation"] = {"turn": turn, "angle_degrees": angle_degrees}

    sanitized["trigger_fall"] = trigger_fall

    fall_direction = matrix.get("fall_direction")
    if not trigger_fall:
        fall_direction = "none"
    elif fall_direction not in VALID_FALL_DIRECTIONS or fall_direction == "none":
        fall_direction = "forward"
    sanitized["fall_direction"] = fall_direction

    rationale = matrix.get("biomechanical_rationale")
    sanitized["biomechanical_rationale"] = (
        rationale if isinstance(rationale, str) and rationale.strip() else "Kinematic adjustment computed."
    )

    if isinstance(matrix.get("joint_angles"), dict):
        sanitized["joint_angles"] = matrix["joint_angles"]

    return sanitized


def _contains_any_phrase(text, phrases):
    return any(re.search(rf"\b{re.escape(phrase)}\b", text) for phrase in phrases)


def _is_walk_around_detour(text):
    if not isinstance(text, str):
        return False
    normalized = text.lower().strip()
    if re.search(
        r"\b(?:do\s+not|don['’]?t|never)\s+"
        r"(?:(?:please|try\s+to)\s+)?"
        r"(?:walk|go|move|route|navigate|travel|detour)\b"
        r"[^.!?\n]{0,36}\baround\b",
        normalized,
    ):
        return False
    if _contains_any_phrase(normalized, ("turn around", "spin around", "reverse direction")):
        return False
    return bool(
        re.search(
            r"\b(?:walk|go|move|route|navigate|travel|detour)\b[^.!?\n]{0,36}\baround\b",
            normalized,
        )
    )


def _parse_numeric_turn(text):
    if not isinstance(text, str):
        return None
    normalized = text.lower().strip()
    if re.search(r"\b(?:do\s+not|don['’]?t|never)\s+(?:turn|rotate)\b", normalized):
        return None

    direction_first = re.search(
        r"\b(?:turn|rotate)\s+"
        r"(clockwise|counter[\s-]?clockwise|left|right)\s+"
        r"(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:degrees?|°)?\b",
        normalized,
    )
    angle_first = re.search(
        r"\b(?:turn|rotate)\s+(?:by\s+)?"
        r"(\d+(?:\.\d+)?)\s*(?:degrees?|°)?\s+"
        r"(clockwise|counter[\s-]?clockwise|left|right)\b",
        normalized,
    )
    if direction_first:
        raw_direction, raw_angle = direction_first.groups()
    elif angle_first:
        raw_angle, raw_direction = angle_first.groups()
    else:
        return None

    normalized_direction = raw_direction.replace(" ", "").replace("-", "")
    turn = (
        "counterclockwise"
        if normalized_direction in {"counterclockwise", "left"}
        else "clockwise"
    )
    return turn, _clip(_safe_float(raw_angle), ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX)


def _is_explicit_away_command(text):
    if not isinstance(text, str):
        return False
    normalized = text.lower().strip()
    if re.search(
        r"\b(?:do\s+not|don['’]?t|never)\s+"
        r"(?:(?:please|try\s+to)\s+)?"
        r"(?:(?:walk|go|move|head|run|back)\s+away|"
        r"(?:walk|go|move)\s+backwards?|"
        r"retreat|flee|escape|withdraw|"
        r"avoid(?:\s+(?:the\s+)?(?:obstacle|obstacles|hazard|hazards))?|"
        r"steer\s+clear(?:\s+of\s+(?:the\s+)?(?:obstacle|obstacles|hazard|hazards))?|"
        r"(?:keep|stay)\s+away|"
        r"(?:leave|exit)\s+(?:the\s+)?(?:obstacle|hazard|course|route|area)|"
        r"(?:turn|spin)\s+around|reverse(?:\s+direction)?)\b",
        normalized,
    ):
        return False
    numeric_turn = _parse_numeric_turn(normalized)
    if numeric_turn and 150.0 <= numeric_turn[1] <= 210.0:
        return True
    return bool(
        re.search(
            r"\b(?:retreat|flee|escape|withdraw|reverse(?:\s+direction)?|"
            r"(?:walk|go|move|head|run|back)\s+away|"
            r"(?:keep|stay)\s+away(?:\s+from\s+(?:the\s+)?(?:obstacle|obstacles|hazard|hazards|them))?|"
            r"avoid\s+(?:the\s+)?(?:obstacle|obstacles|hazard|hazards)|"
            r"steer\s+clear\s+of\s+(?:the\s+)?(?:obstacle|obstacles|hazard|hazards)|"
            r"(?:leave|exit)\s+(?:the\s+)?(?:obstacle|hazard|course|route|area)|"
            r"(?:walk|go|move)\s+backwards?|"
            r"(?:turn|spin)\s+around)\b",
            normalized,
        )
    )


def _apply_command_semantic_overrides(matrix, command, obstacle_type):
    """Applies narrow, deterministic intent rules after either controller path.

    The browser owns world-space detour waypoints, but it needs a consistent walking
    response from both Groq and the local fallback. This override prevents an explicit
    walk-around request from being converted into a jump, climb, or slide merely
    because a particular hazard type is active.
    """
    explicit_detour = bool(obstacle_type and _is_walk_around_detour(command))
    explicit_away = bool(obstacle_type and _is_explicit_away_command(command))
    if not explicit_detour and not explicit_away:
        return matrix

    text = command.lower()
    numeric_turn = _parse_numeric_turn(text) if explicit_away and not explicit_detour else None
    if numeric_turn:
        turn, turn_angle = numeric_turn
    else:
        turn = (
            "counterclockwise"
            if explicit_detour and "left" in text and "right" not in text
            else "clockwise"
        )
        turn_angle = 45.0 if explicit_detour else 180.0
    detour_matrix = dict(matrix) if isinstance(matrix, dict) else {}
    detour_matrix.update(
        {
            "movement_state": "walk",
            "velocity": max(_safe_float(detour_matrix.get("velocity"), 0.8), 0.8),
            "step_frequency": max(
                _safe_float(detour_matrix.get("step_frequency"), 1.5),
                1.4,
            ),
            "target_pelvic_quaternion": _posture_quaternion(),
            "obstacle_action": "none",
            "lateral_shift": 0.0,
            "rotation": {"turn": turn, "angle_degrees": turn_angle},
            "trigger_fall": False,
            "fall_direction": "none",
            "biomechanical_rationale": (
                "A controlled 45-degree walking bypass will clear the hazard and rejoin the obstacle route."
                if explicit_detour
                else "The robot will turn away and retreat without attempting to clear the active hazard."
            ),
        }
    )
    com_shift = detour_matrix.get("center_of_mass_shift")
    if not isinstance(com_shift, dict):
        com_shift = {}
    detour_matrix["center_of_mass_shift"] = {
        "x": _safe_float(com_shift.get("x"), 0.0),
        "y": 0.0,
        "z": _safe_float(com_shift.get("z"), 0.08),
    }
    return sanitize_kinematic_matrix(detour_matrix, obstacle_type)


def _posture_quaternion(pitch=0.0, roll=0.0):
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    return {
        "w": cp * cr,
        "x": sp * cr,
        "y": sp * sr,
        "z": cp * sr,
    }


def build_deterministic_kinematic_matrix(command, terrain, friction, angle, obstacle):
    """Provides a safe, stateless traversal controller when Groq is temporarily
    unavailable or a deployment is still running an incompatible provider
    configuration. It intentionally covers the same command/action vocabulary as
    the model prompt so a provider failure never leaves the robot frozen."""
    text = command.lower().strip()
    obstacle_type = obstacle.get("type") if isinstance(obstacle, dict) else None
    explicit_stop = _contains_any_phrase(
        text,
        ("stop", "halt", "hold", "wait", "freeze", "idle", "stand still"),
    )
    explicit_fall = (
        _contains_any_phrase(
            text,
            ("fall", "collapse", "topple", "faint", "lie down", "play dead"),
        )
        and "do not fall" not in text
        and "don't fall" not in text
    )

    turn = "none"
    turn_angle = 0.0
    explicit_detour = bool(obstacle_type and _is_walk_around_detour(text))
    numeric_turn = _parse_numeric_turn(text)
    turn_is_negated = bool(
        re.search(
            r"\b(?:do\s+not|don['’]?t|never)\s+"
            r"(?:(?:please|try\s+to)\s+)?"
            r"(?:turn|rotate|go|move|walk|head)\s+"
            r"(?:to\s+the\s+)?(?:left|right)\b",
            text,
        )
    )
    if _is_explicit_away_command(text):
        turn = "clockwise"
        turn_angle = 180.0
        if numeric_turn:
            turn, turn_angle = numeric_turn
    elif numeric_turn:
        turn, turn_angle = numeric_turn
    elif not turn_is_negated and ("turn left" in text or "go left" in text):
        turn = "counterclockwise"
        turn_angle = 45.0
    elif not turn_is_negated and ("turn right" in text or "go right" in text or explicit_detour):
        turn = "clockwise"
        turn_angle = 45.0
    if explicit_detour and "left" in text and "right" not in text:
        turn = "counterclockwise"

    fall_direction = "none"
    if explicit_fall:
        if "back" in text:
            fall_direction = "backward"
        elif "left" in text:
            fall_direction = "left"
        elif "right" in text:
            fall_direction = "right"
        else:
            fall_direction = "forward"

    obstacle_action = "none"
    movement_state = "walk"
    velocity = 0.8
    step_frequency = 1.5
    lateral_shift = 0.0
    pitch = 0.0
    roll = 0.0

    if explicit_fall:
        movement_state = "brace"
        velocity = 0.0
        step_frequency = STEP_FREQUENCY_MIN
        turn = "none"
        turn_angle = 0.0
    elif explicit_stop:
        movement_state = "brace" if obstacle_type else "idle"
        obstacle_action = "brace" if obstacle_type else "none"
        velocity = 0.0
        step_frequency = STEP_FREQUENCY_MIN
    elif explicit_detour:
        movement_state = "walk"
        obstacle_action = "none"
        velocity = 0.8
        step_frequency = 1.5
    elif obstacle_type == "boulder":
        obstacle_action = "climb" if _contains_any_phrase(text, ("climb", "mount", "grip", "ledge")) else "jump"
    elif obstacle_type == "crevice":
        obstacle_action = "jump"
    elif obstacle_type == "ice_patch":
        if "left" in text:
            obstacle_action = "sidestep_left"
        elif "right" in text:
            obstacle_action = "sidestep_right"
        elif "crouch" in text:
            obstacle_action = "crouch"
        else:
            obstacle_action = "slide"
    elif obstacle_type == "debris":
        if "left" in text:
            obstacle_action = "sidestep_left"
        elif "right" in text or turn != "none":
            obstacle_action = "sidestep_right"
        elif _contains_any_phrase(text, ("push", "through", "forward")):
            obstacle_action = "push_through"
        elif "slide" in text:
            obstacle_action = "slide"
        else:
            obstacle_action = "climb"
    elif "run" in text or "sprint" in text:
        movement_state = "run"
        velocity = 1.8
        step_frequency = 2.4
        pitch = -0.16
    elif "jump" in text:
        movement_state = "jump"
        velocity = 1.4
        step_frequency = 2.0
        pitch = -0.12
    elif "slide" in text:
        movement_state = "slide"
        velocity = 1.3
        step_frequency = 1.8
        pitch = -0.2
    elif "crouch" in text:
        movement_state = "crouch"
        velocity = 0.35
        step_frequency = 1.0
        pitch = 0.08
    elif "sidestep" in text or "step left" in text or "step right" in text:
        movement_state = "sidestep"
        obstacle_action = "none"
        velocity = 0.7
        step_frequency = 1.5
        lateral_shift = 1.4
        roll = -0.16 if "left" in text else 0.16

    if obstacle_action in ACTION_MOVEMENT_STATES:
        movement_state = ACTION_MOVEMENT_STATES[obstacle_action]
        if obstacle_action == "jump":
            velocity, step_frequency, pitch = 1.5, 2.1, -0.12
        elif obstacle_action == "climb":
            velocity, step_frequency, pitch = 1.2, 2.0, -0.18
        elif obstacle_action == "slide":
            velocity, step_frequency, pitch = 1.3, 1.8, -0.2
        elif obstacle_action == "crouch":
            velocity, step_frequency, pitch = 0.4, 1.0, 0.08
        elif obstacle_action == "push_through":
            velocity, step_frequency, pitch = 1.2, 1.8, -0.2
        elif obstacle_action.startswith("sidestep"):
            velocity, step_frequency = 0.7, 1.5
            lateral_shift = 1.4
            roll = -0.16 if obstacle_action == "sidestep_left" else 0.16

    friction_factor = _clip(friction, 0.05, 1.0)
    stability = _clip(
        96.0 - abs(angle) * 0.7 - (1.0 - friction_factor) * 18.0,
        STABILITY_MIN,
        STABILITY_MAX,
    )
    capture_z = 0.0 if velocity == 0 else _clip(velocity * 0.11, CAPTURE_Z_MIN, CAPTURE_Z_MAX)
    capture_x = -0.16 if obstacle_action == "sidestep_left" else 0.16 if obstacle_action == "sidestep_right" else 0.0

    matrix = {
        "movement_state": movement_state,
        "velocity": velocity,
        "center_of_mass_shift": {
            "x": capture_x * 0.5,
            "y": -0.18 if movement_state in {"crouch", "slide"} else 0.06 if movement_state == "climb" else 0.0,
            "z": min(capture_z, 0.2),
        },
        "step_frequency": step_frequency,
        "target_pelvic_quaternion": _posture_quaternion(pitch, roll),
        "calculated_capture_point": {"x": capture_x, "z": capture_z},
        "torque_compensation": {
            "ankle": _clip(angle * -1.2, TORQUE_MIN, TORQUE_MAX),
            "knee": 22.0 if movement_state in {"jump", "climb", "crouch", "slide"} else 8.0,
            "waist": _clip(pitch * -90.0, TORQUE_MIN, TORQUE_MAX),
        },
        "stability_projection": stability,
        "obstacle_action": obstacle_action,
        "lateral_shift": lateral_shift,
        "rotation": {"turn": turn, "angle_degrees": turn_angle},
        "trigger_fall": explicit_fall,
        "fall_direction": fall_direction,
        "biomechanical_rationale": (
            "Provider fallback selected a safe, deterministic maneuver for the current telemetry snapshot."
        ),
    }
    return _apply_command_semantic_overrides(
        sanitize_kinematic_matrix(matrix, obstacle_type),
        command,
        obstacle_type,
    )


def _provider_failure_metadata(error):
    error_name = type(error).__name__
    provider_status = getattr(error, "status_code", None)

    if provider_status == 429 or error_name == "RateLimitError":
        return 429, "provider_rate_limited", "Kinematic Brain rate limit reached. Wait briefly and retry the command."
    if provider_status in {401, 403} or error_name in {"AuthenticationError", "PermissionDeniedError"}:
        return 502, "provider_authentication_failed", "Groq rejected the configured API credentials."
    if provider_status == 404 or error_name == "NotFoundError":
        return 502, "provider_model_unavailable", "The configured Groq model is unavailable."
    if provider_status in {400, 422} or error_name in {"BadRequestError", "UnprocessableEntityError", "TypeError"}:
        return 502, "provider_request_rejected", "Groq rejected the model request configuration."
    if error_name in {"APITimeoutError", "TimeoutException", "ReadTimeout"}:
        return 504, "provider_timeout", "Kinematic Brain inference timed out. Please retry the command."
    if error_name in {"APIConnectionError", "ConnectError", "ConnectionError"}:
        return 502, "provider_connection_failed", "Could not connect to the Kinematic Brain provider."
    if provider_status and provider_status >= 500:
        return 502, "provider_unavailable", "Groq is temporarily unavailable."
    if isinstance(error, (json.JSONDecodeError, ValueError)):
        return 502, "invalid_provider_response", "Kinematic Brain returned an invalid response."
    return 502, "provider_error", "Kinematic Brain request failed."


def _provider_error_response(error, request_id=None, fallback_message="Kinematic Brain request failed."):
    """Converts provider/network failures into stable JSON errors without leaking credentials or internals."""
    status_code, error_code, message = _provider_failure_metadata(error)
    if error_code == "provider_error":
        message = fallback_message

    app.logger.exception("Groq request failed [%s]", error_code)
    payload = {"error": message, "error_code": error_code}
    if request_id is not None:
        payload["request_id"] = request_id
    return jsonify(payload), status_code


@app.route('/')
def index():
    """Renders the master telemetry UI dashboard."""
    return render_template(
        'index.html',
        supabase_url=os.environ.get("SUPABASE_URL", ""),
        supabase_anon_key=os.environ.get("SUPABASE_ANON_KEY", "")
    )

@app.route('/api/health')
def health_check():
    """Verifies backend operational status and configuration parsing state."""
    has_api_key = bool(os.environ.get("GROQ_API_KEY"))
    has_supabase = bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_ANON_KEY"))
    return jsonify({
        "status": "online",
        "engine": "TerraWalk AI Kinematic System",
        "model": GROQ_MODEL,
        "groq_configured": has_api_key,
        "groq_auth_established": has_api_key,
        "groq_sdk_version": getattr(groq, "__version__", "unknown"),
        "strict_schema_compatible": GROQ_MODEL in STRICT_SCHEMA_GROQ_MODELS,
        "model_override_ignored": GROQ_MODEL_OVERRIDE_IGNORED,
        "deterministic_fallback_available": True,
        "supabase_configured": has_supabase
    })

@app.route('/api/traverse', methods=['POST'])
def traverse():
    """Maps traversal prompts and environmental variables to the Groq High-Level Kinematic Brain."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Traversal request body must be a JSON object."}), 400

    raw_request_id = data.get("request_id")
    request_id = str(raw_request_id)[:80] if raw_request_id is not None else None

    raw_command = data.get("command", "Maintain standard balance stance")
    command = raw_command.strip()[:1000] if isinstance(raw_command, str) else ""
    if not command:
        return jsonify({"error": "Traversal command must be a non-empty string.", "request_id": request_id}), 400

    raw_terrain = data.get("terrain", "flat")
    terrain = raw_terrain if raw_terrain in VALID_TERRAINS else "flat"
    friction = _clip(_safe_float(data.get("friction"), 0.5), 0.01, 1.0)
    angle = _clip(_safe_float(data.get("angle"), 0.0), -45.0, 45.0)

    obstacle = data.get("obstacle")
    if isinstance(obstacle, dict) and obstacle.get("type") in VALID_OBSTACLE_TYPES:
        obstacle = {
            "type": obstacle.get("type"),
            "distance": _clip(_safe_float(obstacle.get("distance"), 0.0), 0.0, 50.0),
        }
    else:
        obstacle = None

    user_prompt = json.dumps(
        {
            "command": command,
            "terrain": terrain,
            "friction": friction,
            "ground_angle_degrees": angle,
            "active_hazard": obstacle,
        },
        separators=(",", ":"),
    )

    controller_source = "groq"
    provider_warning = None
    if groq_client:
        try:
            # Strict schema decoding keeps every response structurally valid before the
            # defense-in-depth numerical sanitizer applies the simulation-specific rules.
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": KINEMATIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=GROQ_MODEL,
                temperature=0.1,
                max_completion_tokens=700,
                reasoning_effort="low",
                include_reasoning=False,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "terrawalk_kinematic_matrix",
                        "strict": True,
                        "schema": KINEMATIC_RESPONSE_SCHEMA,
                    },
                },
            )

            response_content = chat_completion.choices[0].message.content
            if not isinstance(response_content, str) or not response_content.strip():
                raise ValueError("Kinematic Brain returned an empty response.")

            kinematic_matrix = json.loads(response_content)
            kinematic_matrix = sanitize_kinematic_matrix(
                kinematic_matrix,
                obstacle.get("type") if obstacle else None,
            )
        except Exception as error:
            _, error_code, error_message = _provider_failure_metadata(error)
            app.logger.exception(
                "Groq traversal failed [%s]; deterministic controller engaged",
                error_code,
            )
            controller_source = "deterministic_fallback"
            provider_warning = {
                "code": error_code,
                "message": error_message,
            }
            kinematic_matrix = build_deterministic_kinematic_matrix(
                command,
                terrain,
                friction,
                angle,
                obstacle,
            )
    else:
        controller_source = "deterministic_fallback"
        provider_warning = {
            "code": "provider_not_configured",
            "message": "Groq is not configured; the safe local traversal controller was used.",
        }
        kinematic_matrix = build_deterministic_kinematic_matrix(
            command,
            terrain,
            friction,
            angle,
            obstacle,
        )

    kinematic_matrix = _apply_command_semantic_overrides(
        kinematic_matrix,
        command,
        obstacle.get("type") if obstacle else None,
    )

    kinematic_matrix["schema_version"] = "3.2"
    kinematic_matrix["controller_source"] = controller_source
    if provider_warning:
        kinematic_matrix["provider_warning"] = provider_warning
    if request_id is not None:
        kinematic_matrix["request_id"] = request_id

    response = jsonify(kinematic_matrix)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["X-TerraWalk-Controller"] = controller_source
    return response

# --------------------------------------------------------------------------------------
# ROBOT SYSTEMS ASSISTANT (Chatbot)
#
# Conversational Q&A endpoint that powers the chatbot on the "Robot Systems" info page.
# It is intentionally independent of the /api/traverse Kinematic Brain pipeline above --
# it does not affect movement, balance, or any simulation state, it only answers operator
# questions about the robot's physics/control systems and the application itself.
# --------------------------------------------------------------------------------------
ROBOT_ASSISTANT_SYSTEM_PROMPT = (
    "You are the TerraWalk AI Systems Assistant, an onboard documentation assistant embedded in the "
    "TerraWalk AI humanoid robotics simulation platform. You answer operator questions about the robot's "
    "physics, control systems, and the application itself clearly and concisely (roughly 2-5 sentences "
    "unless the operator explicitly asks for a deeper breakdown).\n\n"
    "KNOWLEDGE BASE -- TerraWalk AI Kinematic System:\n"
    "- Spherical Inverted Pendulum (SIP): the torso/pelvis balance model used instead of independent joint "
    "angles. Balance is expressed as a single unit quaternion (target_pelvic_quaternion) built from a pitch "
    "angle p (hard range -0.35 to 0.35 radians) and a roll angle r (hard range -0.3 to 0.3 radians):\n"
    "    w = cos(p/2) * cos(r/2)\n"
    "    x = sin(p/2) * cos(r/2)\n"
    "    y = sin(p/2) * sin(r/2)\n"
    "    z = cos(p/2) * sin(r/2)\n"
    "  Negative pitch leans the torso forward (running, climbing, accelerating, pushing through). Positive "
    "pitch leans it backward (braking, bracing, standing up out of a crouch). Roll is driven by sidesteps. "
    "Quaternions are used specifically because they avoid gimbal lock, unlike stacked Euler joint angles.\n"
    "- Zero Moment Point (ZMP) / Capture Point: the response's bounded target ground point "
    "(calculated_capture_point, x/z in meters relative to the midpoint of the feet). The simulator "
    "interpolates the pulsing red marker toward the latest accepted target. The fixed green ring marks "
    "the generated terrain origin; it is not a live capture/support boundary. z ranges -0.1 to 0.35m "
    "and x ranges -0.3 to 0.3m.\n"
    "- Linear Inverted Pendulum (LIP) locomotion model: bounds walking/running speed using physical leg "
    "parameters instead of trusting the LLM's raw velocity number directly. Pendulum height is approx. "
    "0.92m, natural angular frequency omega = sqrt(gravity / height), max step length is approx. 0.85m, "
    "step frequency ranges 0.4-3.0 Hz, comfortable walking ceiling is 1.6 m/s, and the hard run ceiling is "
    "max step length multiplied by max step frequency.\n"
    "- Torque compensation telemetry: the response contains bounded ankle, knee, and waist estimates in "
    "Newton-meters, each ranging -150 to 150. The current console displays ankle and knee values; these "
    "numbers are not directly applied as joint-physics actuators in the renderer.\n"
    "- Stability projection: the response contains a sanitized 0-100% estimate, but the current console "
    "does not consume that field. Its Motion Stability Estimate is a separate client-side heuristic based "
    "on movement speed, stumble state, and incapacitation state.\n"
    "- Terrain adaptation: the client continuously measures the actual ground slope under the robot's feet "
    "and blends a deterministic compensating lean into the pelvic quaternion every frame. Below 10 degrees "
    "of tilt the robot walks normally; between 10 and 32 degrees it progressively slows down and leans into "
    "the grade. Beyond 32 degrees it first checks for a nearby climbable ledge or boulder/debris grip; when "
    "one exists it runs the climb animation. Known downhill faces on authored rubble, volcanic, and mountain "
    "features use a smooth terrain-assisted descent slide instead of collapsing. Unsupported steep terrain "
    "still triggers a fall and comes to rest at an 84 degree lie angle, recoverable with a reboot. The "
    "Mountain Range biome is built primarily from climbable miniature mountains. Generated worlds use a "
    "safe spawn platform and cap their initial global incline at plus/minus 18 degrees.\n"
    "- Hazards: a boulder or crevice can be cleared with 'jump' or 'climb'; an ice patch with 'crouch', a "
    "sidestep, or a 'slide'; debris with 'climb', a sidestep, or a 'slide'. Jump and slide each play their "
    "own dedicated animation, and both can also be issued as standalone commands (e.g. just 'jump' or "
    "'slide') even with no hazard present. Operators can also command a turn/rotation to detour around a "
    "hazard instead of tackling it head-on -- phrasing like 'walk around it' or 'go around the obstacle' is "
    "treated as a gentle detour that passes the hazard and rejoins the obstacle route, distinct from 'turn "
    "around', which reverses direction 180 degrees. A return control can place the robot in front of the "
    "most recently cleared hazard without resetting progress, and gentle safety guidance steers ordinary "
    "forward traversal toward the next uncleared hazard unless the operator explicitly directs it away.\n"
    "- Map boundary: the rendered terrain has a real edge. Crossing it produces a gravity-driven void "
    "descent and a distinct SYSTEM VOIDED failure instead of an invisible boundary or a tilt-overload "
    "message. Reboot returns the robot to the safe spawn while retaining cleared-hazard progress.\n"
    "- Application stack: a Flask backend (deployed on Vercel) calls Groq's GPT-OSS 20B model (the "
    "'Kinematic Brain') to translate natural-language traversal commands into a balance/gait JSON schema "
    "every time the operator issues a command. Three.js renders the humanoid and procedural terrain client-"
    "side at interactive framerates. Supabase provides optional operator authentication.\n\n"
    "STYLE RULES:\n"
    "- Stay strictly on the topic of this robot, its physics/control systems, and this application.\n"
    "- If asked something unrelated (general chit-chat, unrelated coding help, unrelated trivia, etc.), "
    "briefly and politely redirect the operator back to robot/app topics.\n"
    "- Be precise with numbers and units when they're relevant, but keep answers conversational rather than "
    "a dense wall of bullet points, unless the operator explicitly asks for a structured breakdown.\n"
    "- Never claim the robot is a physical, real-world machine -- it is a browser-based kinematic "
    "simulation."
)


@app.route('/api/robot-chat', methods=['POST'])
def robot_chat():
    """Conversational Q&A endpoint for the Robot Systems info page. Answers operator questions about the
    robot's physics/control systems and the application itself using the Groq LLM grounded in the fixed
    knowledge base above. Does not read or write any simulation/traversal state."""
    if not groq_client:
        return jsonify({
            "error": "Groq client uninitialized. Please confirm your GROQ_API_KEY environment configuration mapping."
        }), 500

    try:
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        raw_history = data.get("history")
        raw_history = raw_history if isinstance(raw_history, list) else []

        if not user_message:
            return jsonify({"error": "Empty message."}), 400

        # Cap history to the last few turns and coerce to the plain {role, content} shape the API
        # expects, silently dropping anything malformed rather than erroring the whole request out.
        trimmed_history = []
        for turn in raw_history[-8:]:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                trimmed_history.append({"role": role, "content": content.strip()[:2000]})

        messages = [{"role": "system", "content": ROBOT_ASSISTANT_SYSTEM_PROMPT}]
        messages.extend(trimmed_history)
        messages.append({"role": "user", "content": user_message[:2000]})

        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
            temperature=0.4,
            max_tokens=500,
            reasoning_effort="low",
        )

        reply = chat_completion.choices[0].message.content
        return jsonify({"reply": reply})

    except Exception as error:
        return _provider_error_response(
            error,
            fallback_message="Failed to reach the TerraWalk AI Systems Assistant.",
        )


# Expose app cluster instance to Vercel global runtime context
if __name__ == '__main__':
    app.run(debug=True)
