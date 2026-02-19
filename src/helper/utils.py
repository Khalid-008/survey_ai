import pandas as pd
from flask import request, abort


# ============================================================================
# FLASK UTILITIES
# ============================================================================

def get_request_body():
    if not request.is_json:
        abort(400, "Request must be JSON")

    request_body = request.get_json()

    if 'request' not in request_body:
        abort(400, "Missing 'request' field in request body")

    req_data = request_body.get('request', {})
    if 'message' not in req_data:
        abort(400, "Missing required field 'message' in request['request']")

    return request_body


# ============================================================================
# SERIALIZATION UTILITIES
# تحويل البيانات إلى صيغة JSON قابلة للإرسال
# ============================================================================

def make_serializable(results, keys_to_remove=None):
    """
    تحويل أي قاموس يحتوي على بيانات إلى صيغة JSON قابلة للإرسال.

    results       : القاموس الرئيسي { id: { key: value, ... } }
    keys_to_remove: قائمة بأسماء المفاتيح التي تريد حذفها من كل عنصر
                    مثال: ['enriched_df', 'raw_data']
                    إذا تركتها فارغة ستُحذف: ['enriched_df'] بشكل افتراضي
    """
    # القيم الافتراضية للمفاتيح المحذوفة
    if keys_to_remove is None:
        keys_to_remove = ['enriched_df']

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