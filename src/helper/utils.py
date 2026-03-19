import pandas as pd
from flask import request, abort
import json
import ast


# ============================================================================
# FLASK UTILITIES
# ============================================================================


def get_request_body():
    if not request.is_json:
        abort(400, "Request must be JSON")

    request_body = request.get_json()

    if "request" not in request_body:
        abort(400, "Missing 'request' field in request body")

    req_data = request_body.get("request", {})
    if "message" not in req_data:
        abort(400, "Missing required field 'message' in request['request']")

    return request_body


# ============================================================================
# SERIALIZATION UTILITIES
# تحويل البيانات إلى صيغة JSON قابلة للإرسال
# ============================================================================


def make_serializable(results, keys_to_remove=None):
    # القيم الافتراضية للمفاتيح المحذوفة
    if keys_to_remove is None:
        keys_to_remove = ["enriched_df"]

    serialized = {}

    for item_id, item_data in results.items():
        item_data = item_data.copy()

        # حذف المفاتيح غير المرغوب فيها
        for key in keys_to_remove:
            if key in item_data:
                del item_data[key]

        # تحويل كائنات pandas إلى قواميس عادية
        for key, value in item_data.items():
            if isinstance(value, (pd.DataFrame, pd.Series)):
                item_data[key] = value.to_dict()

        serialized[str(item_id)] = item_data

    return serialized


def parse_json_response(raw: str, context: str = "") -> dict | None:
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as json_err:
        try:
            # Fallback for trailing commas and single quotes
            fixed_text = text.replace("null", "None").replace("true", "True").replace("false", "False")
            parsed_dict = ast.literal_eval(fixed_text)
            
            if isinstance(parsed_dict, dict):
                print(f"   ℹ️  [{context}] JSON auto-repaired using fallback parser.")
                return parsed_dict
            else:
                return None
        except Exception as fallback_err:
            print(f"   ⚠️  JSON parse error [{context}]: {json_err} | Fallback failed: {fallback_err}")
            return None
