from flask import Flask, make_response, request
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle



app = Flask(__name__)

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def form():
    return """
        <html>
            <body>
            <h1>Customer Churn Prediction - CSV Format</h1>
                <h2>Let's TRY to Predict..</h2>
                </br>
                </br>
                <p> Insert your CSV file and then download the Result
                <form action="/transform" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" class="btn btn-block"/>
                    </br>
                    </br>
                    <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
                </form>
            </body>
        </html>
    """
@app.route('/transform', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    #print("file contents: ", file_contents)
    #print(type(file_contents))
    print(csv_input)
    for row in csv_input:
        print(row)

    stream.seek(0)
    result = transform(stream.read())

    df = pd.read_csv(StringIO(result))
    

    # load the model from disk
    loaded_model = pickle.load(open("rf_model_csv.sav", 'rb'))
    df['prediction'] = loaded_model.predict(df.tail(1))
    df['probablity'] = loaded_model.predict_proba(df.tail(1))

    

    response = make_response(df.to_csv())
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response

if __name__ == "__main__":
    app.run(debug=True,port=8080)