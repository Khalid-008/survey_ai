import os
import json
import nest_asyncio
from flask import Flask, abort, jsonify
from flask_cors import CORS
from collections import OrderedDict
from helper.utils import get_request_body, extract_json
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
    survey_id = request_body['request']['survey_id']
    user_message  = request_body['request']['message']
    session_id  = request_body['request']['session_id']

    # Get workflow response
    response = create_survey_insight_workflow(survey_id, user_message, session_id)
    return response


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="5566")
