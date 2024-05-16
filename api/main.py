from flask import Flask, request, jsonify
from flask_cors import CORS
from api.test import minmodel_deter, exp_min_time, exp_min_cost

app = Flask(__name__)
CORS(app)

@app.route('/submit_form', methods=['POST', 'OPTIONS'])
def submit_form():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    elif request.method == 'POST':
        return _handle_post_request()

def _build_cors_preflight_response():
    response = jsonify({'status': 'Preflight request handled'})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return response

def _handle_post_request():
    # Get JSON data from the request body
    data = request.json

    # Check if data is not None and contains the required keys
    required_keys = [
        'd1', 'd2', 'd3', 'd4', 
        'lt_vl_mean', 'lt_vl_std', 'ut_vl_mean', 'ut_vl_std', 
        'lt_l_mean', 'lt_l_std', 'ut_l_mean', 'ut_l_std', 
        'speed_lt_vl_mean', 'speed_lt_vl_std', 'speed_ut_vl_mean', 'speed_ut_vl_std', 
        'speed_lt_l_mean', 'speed_lt_l_std', 'speed_ut_l_mean', 'speed_ut_l_std', 
        'objective', 'dropdown1', 'dropdown2', 'dropdown3', 'dropdown4','num_vl','num_l', 'checkbox1', 'checkbox2', 'checkbox3', 'checkbox4'
    ]

    if data is not None and all(key in data for key in required_keys):
        # Convert string values to integers/floats
        d1 = int(data['d1'])
        d2 = int(data['d2'])
        d3 = int(data['d3'])
        d4 = int(data['d4'])
        lt_vl_mean = float(data['lt_vl_mean'])
        lt_vl_std = float(data['lt_vl_std'])
        ut_vl_mean = float(data['ut_vl_mean'])
        ut_vl_std = float(data['ut_vl_std'])
        lt_l_mean = float(data['lt_l_mean'])
        lt_l_std = float(data['lt_l_std'])
        ut_l_mean = float(data['ut_l_mean'])
        ut_l_std = float(data['ut_l_std'])
        speed_lt_vl_mean = float(data['speed_lt_vl_mean'])
        speed_lt_vl_std = float(data['speed_lt_vl_std'])
        speed_ut_vl_mean = float(data['speed_ut_vl_mean'])
        speed_ut_vl_std = float(data['speed_ut_vl_std'])
        speed_lt_l_mean = float(data['speed_lt_l_mean'])
        speed_lt_l_std = float(data['speed_lt_l_std'])
        speed_ut_l_mean = float(data['speed_ut_l_mean'])
        speed_ut_l_std = float(data['speed_ut_l_std'])

        num_vl = int(data['num_vl'])
        num_l = int(data['num_l'])

        print(num_vl,num_l)

        # Get the objective
        objective = data['objective']

        # Get dropdown values
        dropdown1 = data['dropdown1']
        dropdown2 = data['dropdown2']
        dropdown3 = data['dropdown3']
        dropdown4 = data['dropdown4']

        # Get checkbox values
        checkbox1 = data['checkbox1']
        checkbox2 = data['checkbox2']
        checkbox3 = data['checkbox3']
        checkbox4 = data['checkbox4']


        # Call the appropriate function based on the objective
        if objective == 'deterministic_min_makespan':
            result = minmodel_deter(
                d1, d2, d3, d4, 
                lt_vl_mean, lt_vl_std, ut_vl_mean, ut_vl_std, 
                lt_l_mean, lt_l_std, ut_l_mean, ut_l_std, 
                speed_lt_vl_mean, speed_lt_vl_std, speed_ut_vl_mean, speed_ut_vl_std, 
                speed_lt_l_mean, speed_lt_l_std, speed_ut_l_mean, speed_ut_l_std,
                dropdown1, dropdown2, dropdown3, dropdown4,num_vl,num_l,checkbox1,checkbox2,checkbox3,checkbox4
            )
        elif objective == 'expected_min_time':
            result = exp_min_time(
                d1, d2, d3, d4, 
                lt_vl_mean, lt_vl_std, ut_vl_mean, ut_vl_std, 
                lt_l_mean, lt_l_std, ut_l_mean, ut_l_std, 
                speed_lt_vl_mean, speed_lt_vl_std, speed_ut_vl_mean, speed_ut_vl_std, 
                speed_lt_l_mean, speed_lt_l_std, speed_ut_l_mean, speed_ut_l_std,
                dropdown1, dropdown2, dropdown3, dropdown4,num_vl,num_l,checkbox1,checkbox2,checkbox3,checkbox4
            )
        elif objective == 'expected_min_cost':
            result = exp_min_cost(
                d1, d2, d3, d4, 
                lt_vl_mean, lt_vl_std, ut_vl_mean, ut_vl_std, 
                lt_l_mean, lt_l_std, ut_l_mean, ut_l_std, 
                speed_lt_vl_mean, speed_lt_vl_std, speed_ut_vl_mean, speed_ut_vl_std, 
                speed_lt_l_mean, speed_lt_l_std, speed_ut_l_mean, speed_ut_l_std,
                dropdown1, dropdown2, dropdown3, dropdown4,num_vl,num_l,checkbox1,checkbox2,checkbox3,checkbox4
            )
        else:
            return jsonify({'error': 'Invalid objective'}), 400

        data['result'] = result

        # Process the data as needed
        response = jsonify(data)  # Send data back as JSON for confirmation
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    else:
        response = jsonify({'error': 'Invalid data format or missing keys'})
        response.headers.add("Access-Control-Allow-Origin", "*")

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
