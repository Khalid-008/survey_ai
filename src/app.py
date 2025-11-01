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
    try:
        survey_id = request_body['request']['survey_id']
        user_message  = request_body['request']['message']
        session_id  = request_body['request']['session_id']

        # Get workflow response
        response = create_survey_insight_workflow(survey_id, user_message, session_id)

        # Extract main content JSON
        try:
            content = extract_json(response["content"])
            print(f"✓ Successfully parsed main content JSON")
        except Exception as e:
            print(f"✗ Failed to parse main content as JSON: {e}")
            print(f"Raw content: {response['content'][:500]}")
            raise ValueError(f"Synthesis agent did not return valid JSON. Error: {str(e)}")

        # Validate required fields
        required_fields = ["executive_summary", "detailed_analysis", "key_metrics", "recommendations"]
        missing_fields = [field for field in required_fields if field not in content]
        if missing_fields:
            raise ValueError(f"Missing required fields in synthesis response: {missing_fields}")

        print("✓ All required fields present")
        print("=" * 80)

        # Parse charts from JSON strings to actual objects
        parsed_charts = []
        chart_type_stats = {}
        invalid_charts_count = 0
        
        def validate_chart(chart):
            """Validate that a chart has valid data structure"""
            if not isinstance(chart, dict):
                return False, "Chart is not a dictionary"
            
            chart_type = chart.get('chart_type')
            if not chart_type:
                return False, "Missing 'chart_type' field (required)"
            
            # Validate chart_type is a recognized type
            valid_types = ['bar', 'line', 'doughnut', 'pie', 'scatter', 'bubble', 'radar', 'wordcloud']
            if chart_type not in valid_types:
                return False, f"Invalid chart_type '{chart_type}' (must be one of: {', '.join(valid_types)})"
            
            # Skip data validation for wordcloud charts
            if chart_type == 'wordcloud':
                return True, None
            
            data = chart.get('data', {})
            if not isinstance(data, dict):
                return False, "Missing or invalid 'data' field (must be an object)"
            
            # Check if datasets exists and is a non-empty array
            datasets = data.get('datasets', [])
            if not isinstance(datasets, list) or len(datasets) == 0:
                return False, "Missing or empty 'datasets' array"
            
            # Check if each dataset has valid data
            for ds_idx, dataset in enumerate(datasets):
                if not isinstance(dataset, dict):
                    return False, f"Dataset {ds_idx} is not a dictionary"
                
                ds_data = dataset.get('data', [])
                if not isinstance(ds_data, list) or len(ds_data) == 0:
                    return False, f"Dataset {ds_idx} has empty or missing 'data' array"
            
            return True, None
        
        for idx, chart_str in enumerate(response["charts"]):
            try:
                print(f"\n🔍 PARSING CHART SET {idx+1}:")
                print(f"Raw string length: {len(chart_str)}")
                
                # Parse the JSON string to get the actual chart objects
                chart_data = json.loads(chart_str)
                
                # If chart_data is an array, extend parsed_charts; otherwise append
                if isinstance(chart_data, list):
                    print(f"Parsed {len(chart_data)} charts from this set")
                    for chart in chart_data:
                        chart_type = chart.get('chart_type', 'UNKNOWN')
                        
                        # Validate chart before adding
                        is_valid, error_msg = validate_chart(chart)
                        if is_valid:
                            chart_type_stats[chart_type] = chart_type_stats.get(chart_type, 0) + 1
                            print(f"  ✅ Chart type: {chart_type}, title: {chart.get('title')}")
                            parsed_charts.append(chart)
                        else:
                            invalid_charts_count += 1
                            title = chart.get('title', 'Untitled')
                            print(f"  ❌ Invalid chart skipped: {chart_type}, title: {title}")
                            if error_msg:
                                print(f"     Reason: {error_msg}")
                else:
                    chart_type = chart_data.get('chart_type', 'UNKNOWN')
                    
                    # Validate chart before adding
                    is_valid, error_msg = validate_chart(chart_data)
                    if is_valid:
                        chart_type_stats[chart_type] = chart_type_stats.get(chart_type, 0) + 1
                        print(f"  ✅ Chart type: {chart_type}, title: {chart_data.get('title')}")
                        parsed_charts.append(chart_data)
                    else:
                        invalid_charts_count += 1
                        title = chart_data.get('title', 'Untitled')
                        print(f"  ❌ Invalid chart skipped: {chart_type}, title: {title}")
                        if error_msg:
                            print(f"     Reason: {error_msg}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse chart JSON: {e}")
                print(f"Chart string: {chart_str[:200]}")
                continue
        
        print(f"\n📊 FINAL CHART TYPE STATISTICS:")
        for chart_type, count in sorted(chart_type_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {chart_type}: {count} charts")
        print(f"Total charts parsed: {len(parsed_charts)}")
        if invalid_charts_count > 0:
            print(f"⚠️ Invalid charts filtered out: {invalid_charts_count}\n")
        else:
            print()

        return jsonify(OrderedDict([
            ("executive_summary", content["executive_summary"]),
            ("detailed_analysis", content["detailed_analysis"]),
            ("key_metrics", content["key_metrics"]),
            ("recommendations", content["recommendations"]),
            ("charts", parsed_charts)
        ])), 200

    except Exception as e:
        print(f"ERROR in survey_insight endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return abort(500, f"Server error: {str(e)}")



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="5566")
