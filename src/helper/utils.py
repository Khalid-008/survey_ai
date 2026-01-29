from flask import request, abort

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