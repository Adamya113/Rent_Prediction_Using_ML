from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            size_sq_ft =int(request.form['size_sq_ft'])
            bedrooms =int(request.form['bedrooms'])
            latitude =float(request.form['latitude'])
            longitude =float(request.form['longitude'])
            closest_metro_station_km =float(request.form['closest_metro_station_km'])
       
            data = [size_sq_ft,bedrooms,latitude,longitude,closest_metro_station_km]
            data = np.array(data).reshape(1, 5)
            data = pd.DataFrame(data)
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict[0][0]))

        except Exception as e:
            print('The Exception message is: ',e)
            return f'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)