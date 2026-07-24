import os
import json
import math
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from groq import Groq

# Initialize local environment attributes if executing outside isolated production nodes
load_dotenv()

app = Flask(__name__, template_folder='../templates')

# Initialize Groq client securely using environment properties
api_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None

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

VALID_MOVEMENT_STATES = {"idle", "walk", "run", "crouch", "jump", "climb", "sidestep", "brace"}
VALID_OBSTACLE_ACTIONS = {
    "none", "jump", "climb", "crouch", "sidestep_left", "sidestep_right", "push_through", "brace"
}
VALID_ROTATION_TURNS = {"clockwise", "counterclockwise", "none"}
VALID_FALL_DIRECTIONS = {"forward", "backward", "left", "right", "none"}


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


def sanitize_kinematic_matrix(matrix):
    """Post-processes the raw JSON object returned by the Kinematic Brain: clips every
    numeric field into its physically valid range, drops/repairs malformed nested
    objects, and falls back to safe defaults for invalid or missing enum values. This
    runs on every response before it is ever returned to the client."""
    if not isinstance(matrix, dict):
        matrix = {}

    sanitized = {}

    movement_state = matrix.get("movement_state")
    sanitized["movement_state"] = movement_state if movement_state in VALID_MOVEMENT_STATES else "idle"

    sanitized["velocity"] = _clip(_safe_float(matrix.get("velocity"), 0.0), VELOCITY_MIN, VELOCITY_MAX)

    com_shift = matrix.get("center_of_mass_shift")
    if not isinstance(com_shift, dict):
        com_shift = {}
    sanitized["center_of_mass_shift"] = {
        "x": _clip(_safe_float(com_shift.get("x"), 0.0), COM_XZ_MIN, COM_XZ_MAX),
        "y": _clip(_safe_float(com_shift.get("y"), 0.0), COM_Y_MIN, COM_Y_MAX),
        "z": _clip(_safe_float(com_shift.get("z"), 0.0), COM_XZ_MIN, COM_XZ_MAX),
    }

    sanitized["step_frequency"] = _clip(
        _safe_float(matrix.get("step_frequency"), STEP_FREQUENCY_MIN), STEP_FREQUENCY_MIN, STEP_FREQUENCY_MAX
    )

    # Target pelvic quaternion: clip each component individually first (guards against a
    # wildly out-of-range hallucinated number), then re-normalize so the client always
    # receives a mathematically valid unit quaternion, never a raw/malformed one.
    quat = matrix.get("target_pelvic_quaternion")
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
    sanitized["target_pelvic_quaternion"] = {"w": qw, "x": qx, "y": qy, "z": qz}

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

    obstacle_action = matrix.get("obstacle_action")
    sanitized["obstacle_action"] = obstacle_action if obstacle_action in VALID_OBSTACLE_ACTIONS else "none"

    sanitized["lateral_shift"] = _clip(
        _safe_float(matrix.get("lateral_shift"), 0.0), LATERAL_SHIFT_MIN, LATERAL_SHIFT_MAX
    )

    rotation = matrix.get("rotation")
    if not isinstance(rotation, dict):
        rotation = {}
    turn = rotation.get("turn")
    turn = turn if turn in VALID_ROTATION_TURNS else "none"
    angle_degrees = _clip(_safe_float(rotation.get("angle_degrees"), 0.0), ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX)
    if turn == "none":
        angle_degrees = 0.0
    sanitized["rotation"] = {"turn": turn, "angle_degrees": angle_degrees}

    # Operator-commanded fall (independent of hazard/terrain state).
    trigger_fall = matrix.get("trigger_fall") is True
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

    # joint_angles is not part of the schema requested from the model below, but the
    # client still defensively reads it if present. Pass it through untouched rather
    # than dropping it, since it falls outside the sanitized numeric contract above.
    if isinstance(matrix.get("joint_angles"), dict):
        sanitized["joint_angles"] = matrix["joint_angles"]

    return sanitized


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
        "groq_auth_established": has_api_key,
        "supabase_configured": has_supabase
    })

@app.route('/api/traverse', methods=['POST'])
def traverse():
    """Maps traversal prompts and environmental variables to the Groq High-Level Kinematic Brain."""
    if not groq_client:
        return jsonify({
            "error": "Groq client uninitialized. Please confirm your GROQ_API_KEY environment configuration mapping."
        }), 500

    try:
        data = request.get_json() or {}
        command = data.get("command", "Maintain standard balance stance")
        terrain = data.get("terrain", "flat")
        friction = data.get("friction", 0.5)
        angle = data.get("angle", 0)
        obstacle = data.get("obstacle")  # {"type": "boulder", "distance": 2.3} or None

        # Structure strict system instructions forcing explicit JSON output schemas with movement controls
        system_prompt = (
            "You are the High-Level Kinematic Brain of a sophisticated humanoid robot operating in extreme conditions.\n"
            "Analyze the environment telemetry, any active hazard directly ahead, and the operator's command, then compute\n"
            "the balance adjustment matrices. You must return exclusively a valid JSON object matching the exact\n"
            "specification schema detailed below. Do not output markdown formatting blocks, prefixes, or conversational\n"
            "notes. Output raw, clean JSON text.\n\n"
            "HAZARD RESPONSE RULES:\n"
            "- If an active hazard is present, set obstacle_action to the action the operator's command actually requests.\n"
            "- Valid obstacle_action values and what they mean:\n"
            "  'jump'          -> leap over the hazard (clears boulder or crevice)\n"
            "  'climb'         -> climb over/through the hazard (clears boulder or debris)\n"
            "  'crouch'        -> lower center of mass and creep across (clears ice_patch)\n"
            "  'sidestep_left' -> shift laterally left around the hazard (clears ice_patch or debris)\n"
            "  'sidestep_right'-> shift laterally right around the hazard (clears ice_patch or debris)\n"
            "  'push_through'  -> brace and force through (use only if command explicitly says push/force/tackle through)\n"
            "  'brace'         -> stop and hold stance defensively, does not clear the hazard\n"
            "  'none'          -> command does not address the hazard at all\n"
            "- If there is no active hazard, obstacle_action must always be 'none'.\n"
            "- lateral_shift is only meaningful for sidestep actions: meters to shift sideways, typically between 1.0\n"
            "  and 2.0.\n\n"
            "ROTATION / DETOUR RULES:\n"
            "- If the command explicitly asks the robot to turn or rotate (e.g. 'turn clockwise', 'rotate left 45\n"
            "  degrees', 'turn around', 'spin right'), set rotation.turn to 'clockwise' or 'counterclockwise' and\n"
            "  rotation.angle_degrees to the requested amount. Default to 90 degrees if no amount is given, or 180\n"
            "  degrees for 'turn around'/'spin around'. 'clockwise'/'right' turns map to clockwise; 'counterclockwise'/\n"
            "  'left' turns map to counterclockwise.\n"
            "- Turning is a valid way for the operator to detour and route around a hazard instead of tackling it\n"
            "  head-on. If the command does not request turning, set rotation.turn to 'none' and angle_degrees to 0.\n\n"
            "FALL COMMAND RULES:\n"
            "- If the operator's command explicitly asks the robot to fall down, collapse, drop, topple over, faint,\n"
            "  or lie down/play dead on its own initiative (e.g. 'fall down', 'let yourself fall', 'collapse to the\n"
            "  ground', 'topple over', 'play dead') -- and this is NOT a reaction to the hazard telemetry above --\n"
            "  set trigger_fall to true. Otherwise trigger_fall must always be false.\n"
            "- When trigger_fall is true: set fall_direction to 'forward', 'backward', 'left', or 'right' if the\n"
            "  command names a direction (e.g. 'fall forward' -> forward, 'fall backwards' -> backward, 'collapse to\n"
            "  your left' -> left); default fall_direction to 'forward' if no direction is given. Also set\n"
            "  movement_state to 'brace', velocity to 0, step_frequency to 0.4, obstacle_action to 'none', and\n"
            "  rotation.turn to 'none' -- this is a deliberate operator-triggered shutdown, independent of any hazard\n"
            "  or terrain state.\n"
            "- When trigger_fall is false, fall_direction must always be 'none'.\n\n"
            "SPHERICAL INVERTED PENDULUM (SIP) BALANCE RULES:\n"
            "- The torso/pelvis is no longer balanced with separate joint angles. It is modeled as a 3D spherical\n"
            "  inverted pendulum: you must output a single balance orientation as a unit quaternion\n"
            "  (target_pelvic_quaternion) plus a 2D ground point (calculated_capture_point) representing the target\n"
            "  Zero-Moment Point (ZMP) the pendulum must project onto to stay stable.\n"
            "- Step 1: choose a pitch angle p, in radians, HARD CEILING -0.35 to 0.35 -- never exceed this range even\n"
            "  for extreme commands (forward/back lean). Negative p = leaning forward (running, climbing,\n"
            "  accelerating, pushing through). Positive p = leaning backward (braking, bracing, standing up out of a\n"
            "  crouch). Near 0 for idle or a steady flat-ground walk.\n"
            "- Step 2: choose a roll angle r, in radians, HARD CEILING -0.3 to 0.3 -- never exceed this range even for\n"
            "  extreme commands (side-to-side lean). Driven mainly by the Ground Incline/Tilt telemetry (lean into the\n"
            "  slope to compensate) and by sidestep_left/sidestep_right actions (lean into the direction of the\n"
            "  sidestep). Near 0 otherwise.\n"
            "- Step 3: combine p and r into the unit quaternion with:\n"
            "    w = cos(p/2) * cos(r/2)\n"
            "    x = sin(p/2) * cos(r/2)\n"
            "    y = sin(p/2) * sin(r/2)\n"
            "    z = cos(p/2) * sin(r/2)\n"
            "  These four values already form a normalized quaternion (w^2+x^2+y^2+z^2 = 1) when p and r are in\n"
            "  radians - output them directly as target_pelvic_quaternion.\n"
            "- calculated_capture_point.x and .z are the target ZMP in meters, in the robot's own local ground frame\n"
            "  relative to the midpoint of its feet:\n"
            "    z: positive projects the capture point forward (ahead, the direction the robot is facing), negative\n"
            "       projects it backward. Scale roughly with velocity/step_frequency - faster movement projects the\n"
            "       point further forward (0.0 to 0.35); idle or braced stays near 0. HARD CEILING -0.1 to 0.35.\n"
            "    x: positive shifts the capture point toward the robot's right, negative toward its left. Use larger\n"
            "       magnitudes (0.1 to 0.3) during sidestep_left/sidestep_right or a strong lateral incline, near 0\n"
            "       otherwise. HARD CEILING -0.3 to 0.3.\n\n"
            "HARD NUMERIC CONSTRAINTS (never exceed these, regardless of how extreme the command is -- if your own\n"
            "internal estimate for any of these falls outside its range, clamp it to the nearest bound before writing\n"
            "the final JSON; a server-side safety layer will also silently clip anything you miss, so staying inside\n"
            "these ranges yourself simply produces the most physically sane result):\n"
            "- velocity: 0.0 to 3.0 (m/s). 0 for idle/brace/held crouch, roughly 0.3-1.6 for a walk, roughly 1.6-3.0\n"
            "  for a run.\n"
            "- step_frequency: 0.4 to 3.0 (Hz).\n"
            "- SIP pitch angle p: -0.35 to 0.35 radians.\n"
            "- SIP roll angle r: -0.3 to 0.3 radians.\n"
            "- calculated_capture_point.x: -0.3 to 0.3 meters. calculated_capture_point.z: -0.1 to 0.35 meters.\n"
            "- lateral_shift: 1.0 to 2.0 meters (0 when not sidestepping).\n"
            "- rotation.angle_degrees: 0 to 360.\n"
            "- torque_compensation values (ankle/knee/waist): -150 to 150 Nm.\n"
            "- stability_projection: 0 to 100.\n\n"
            "JSON SCHEMA EXPECTATION:\n"
            "{\n"
            '  "movement_state": "idle" | "walk" | "run" | "crouch" | "jump" | "climb" | "sidestep" | "brace",\n'
            '  "velocity": float,\n'
            '  "center_of_mass_shift": {"x": float, "y": float, "z": float},\n'
            '  "step_frequency": float,\n'
            '  "target_pelvic_quaternion": {"w": float, "x": float, "y": float, "z": float},\n'
            '  "calculated_capture_point": {"x": float, "z": float},\n'
            '  "torque_compensation": {"ankle": float, "knee": float, "waist": float},\n'
            '  "stability_projection": float,\n'
            '  "obstacle_action": "none" | "jump" | "climb" | "crouch" | "sidestep_left" | "sidestep_right" | "push_through" | "brace",\n'
            '  "lateral_shift": float,\n'
            '  "rotation": {"turn": "clockwise" | "counterclockwise" | "none", "angle_degrees": float},\n'
            '  "trigger_fall": boolean,\n'
            '  "fall_direction": "forward" | "backward" | "left" | "right" | "none",\n'
            '  "biomechanical_rationale": "string"\n'
            "}"
        )

        if obstacle and obstacle.get("type"):   
            hazard_block = (
                f"- Active Hazard: {obstacle.get('type')}\n"
                f"- Distance Ahead: {obstacle.get('distance', 0):.1f} meters\n"
                f"- The robot has HALTED in front of this hazard and is awaiting a tackling instruction.\n"
            )
        else:
            hazard_block = "- Active Hazard: none (clear path ahead)\n"

        user_prompt = (
            f"ENVIRONMENT TELEMETRY MATRIX:\n"
            f"- Terrain Profile: {terrain}\n"
            f"- Surface Friction Index: {friction}\n"
            f"- Ground Incline/Tilt: {angle} degrees\n\n"
            f"HAZARD STATUS:\n"
            f"{hazard_block}\n"
            f"OPERATIONAL INTERACTION CRITERIA:\n"
            f"- Traversal Intent: '{command}'"
        )

        # Call ultra-fast Groq LLM inference architecture using Llama 3.1 parsing protocols
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        response_content = chat_completion.choices[0].message.content
        kinematic_matrix = json.loads(response_content)
        kinematic_matrix = sanitize_kinematic_matrix(kinematic_matrix)
        return jsonify(kinematic_matrix)

    except Exception as e:
        return jsonify({
            "error": "Failed to parse mechanical intelligence matrix.",
            "details": str(e)
        }), 500

# Expose app cluster instance to Vercel global runtime context
if __name__ == '__main__':
    app.run(debug=True)