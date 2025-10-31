from flask import Flask, request

app = Flask(__name__)

@app.route('/api/incoming', methods=['GET'])
def incoming_call():
    uuid = request.args.get('uuid')
    caller = request.args.get('caller')
    print(f"Incoming call - UUID: {uuid}, Caller: {caller}")
    return f"Received call info for UUID {uuid} from caller {caller}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
