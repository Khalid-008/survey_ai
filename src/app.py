import os
import json
import nest_asyncio
from flask import Flask, abort, jsonify
from flask_cors import CORS
from collections import OrderedDict
from helper.utils import get_request_body
from agents.graphs.workflow import create_survey_insight_workflow

nest_asyncio.apply()
app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins (change to specific domain in production)
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.route('/survey_insight', methods=['POST'])
def survey_insight():
    request_body = get_request_body()
    survey_id    = request_body['request']['survey_id']
    user_message = request_body['request']['message']
    session_id   = request_body['request']['session_id']
    date_from    = request_body['request'].get('date_from')  # اختياري
    date_to      = request_body['request'].get('date_to')    # اختياري

    # Get workflow response (now returns dict with synthesis and charts)
    response = create_survey_insight_workflow(
        survey_id, user_message, session_id,
        date_from=date_from, date_to=date_to
    )
    
    # Return JSON response with both synthesis and chart configurations
    return jsonify(response)


@app.route('/test_charts', methods=['GET'])
def test_charts():
    """
    يقرأ chart_agent_snapshot.json ويشغّل generate_charts_agent مباشرةً.
    يُستخدم لاختبار النود بدون تشغيل الـ workflow الكامل.
    """
    import json as _json
    from langchain_core.messages import HumanMessage
    from agents.graphs.nodes import generate_charts_agent

    snapshot_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tests", "chart_agent_snapshot.json"
    )

    if not os.path.exists(snapshot_path):
        return jsonify({
            "error": "snapshot not found",
            "message": "شغّل النظام الكامل مرة أولى لتوليد chart_agent_snapshot.json"
        }), 404

    with open(snapshot_path, "r", encoding="utf-8") as f:
        snapshot = _json.load(f)

    mock_state = {
        "survey_id":                  snapshot["survey_id"],
        "messages":                   [HumanMessage(content="test")],
        "survey_data":                [],
        "text_questions_result":      snapshot.get("text_questions_result", {}),
        "selection_questions_result": snapshot.get("selection_questions_result", {}),
        "chart_configs":              [],
    }

    result   = generate_charts_agent(mock_state)
    update   = result.update if hasattr(result, "update") else {}
    charts   = update.get("chart_configs", [])

    return jsonify({
        "survey_id": snapshot["survey_id"],
        "charts":    charts
    })


@app.route('/test_full', methods=['GET'])
def test_full():
    """
    يشغّل synthesis_agent + generate_charts_agent من الـ snapshot.
    يُرجع نفس هيكل /survey_insight عشان يُعرض على لوحة التحليل.
    """
    from langchain_core.messages import HumanMessage, AIMessage
    from agents.graphs.nodes import synthesis_agent, generate_charts_agent

    snapshot_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tests", "chart_agent_snapshot.json"
    )

    if not os.path.exists(snapshot_path):
        return jsonify({
            "error": "snapshot not found",
            "message": "شغّل النظام الكامل مرة أولى لتوليد chart_agent_snapshot.json"
        }), 404

    with open(snapshot_path, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    # ── بناء الـ state ─────────────────────────────────────────────────────
    state = {
        "survey_id":                  snapshot["survey_id"],
        "messages":                   [HumanMessage(content="test")],
        "survey_data":                [],
        "text_questions_result":      snapshot.get("text_questions_result", {}),
        "selection_questions_result": snapshot.get("selection_questions_result", {}),
        "chart_configs":              [],
    }

    # ── تشغيل synthesis_agent ──────────────────────────────────────────────
    synth_result   = synthesis_agent(state)
    synth_update   = synth_result.update if hasattr(synth_result, "update") else {}
    synth_messages = synth_update.get("messages", [])

    synthesis_text = ""
    for msg in reversed(synth_messages):
        if isinstance(msg, AIMessage) and msg.name == "synthesis_agent":
            synthesis_text = msg.content
            break

    # أضف رسالة synthesis إلى الـ state حتى يراها generate_charts_agent
    state["messages"] = state["messages"] + synth_messages

    # ── تشغيل generate_charts_agent ────────────────────────────────────────
    charts_result = generate_charts_agent(state)
    charts_update = charts_result.update if hasattr(charts_result, "update") else {}
    chart_configs = charts_update.get("chart_configs", [])

    # ── تحليل synthesis JSON (نفس منطق workflow.py) ────────────────────────
    try:
        clean = synthesis_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
            clean = clean.rsplit("```", 1)[0].strip()

        parsed = json.loads(clean)
        parsed.setdefault("visualizations", [])
        parsed.setdefault("executive_summary", "")
        parsed.setdefault("detailed_analysis", "")
        parsed.setdefault("key_metrics", [])
        parsed.setdefault("recommendations", [])

        return jsonify({**parsed, "charts": chart_configs})

    except Exception:
        return jsonify({
            "synthesis": synthesis_text,
            "charts":    chart_configs
        })


if __name__ == '__main__':
    print("Starting Survey AI Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5566)
