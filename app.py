from flask import Flask, render_template, request, jsonify, Response
import techgpt
import traceback
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    print("chat route called")
    try:
        message = request.form['message']
        model_app = request.form['modelApp']
        response = techgpt.run_model(message, model_app)
        # print("this is flask print",response)
        print("result",response)
        return jsonify({'response': response})
        # return jsonify(response)   -> this works for privateGPT_archived_agent_working1.py
        
        # # Convert the answer to a JSON string
        # answer_str = json.dumps(response, indent=4)
        
        # # Return the answer string with the correct content type
        # return Response(answer_str, content_type='text/plain; charset=utf-8')
    except Exception as e:
        print("Exception occurred:", traceback.format_exc())
        return Response(str(e), status=500, content_type='text/plain; charset=utf-8')

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host = '0.0.0.0', port=8050)