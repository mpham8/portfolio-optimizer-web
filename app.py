from flask import Flask, render_template, request, redirect, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.utils import secure_filename
import os.path
import shutil


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)


app.config["UPLOAD_FOLDER"] = "uploads/"

@app.route('/upload', methods = ['GET', 'POST'])
def display_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)

        f.save(app.config['UPLOAD_FOLDER'] + filename)

        file = open(app.config['UPLOAD_FOLDER'] + filename,"r")
        content = file.read()   
        
        shutil.copy(os.path.join("uploads/", "stocks.csv"), os.getcwd())

        
    return render_template('index.html', content=content) 


@app.route('/', methods=["GET", "POST"])
def index():
  return render_template("index.html")


@app.route('/download')
def download():
    path = 'results.csv'
    return send_file(path, as_attachment=True)


@app.route('/calculate', methods=["GET", "POST"])
def calc():
  if request.method == 'POST':
    
    simulation_month = request.form['simulation_month']
    years_data = int(request.form['years_data'])
    min_position_size = float(request.form['min_position_size'])
    max_position_size = float(request.form['max_position_size'])
    pricing_model = request.form['pricing_model']
    risk_model = request.form['risk_model']
    objective_function = request.form['objective_function']
    target_return = int(request.form['target_return'])

    print(simulation_month)
    print(years_data)
    print(min_position_size)
    print(max_position_size)
    print(pricing_model)
    print(risk_model)
    print(objective_function)
    print(target_return)


    
    # render_template('loading.html')
    import optimizer as op
    expected_return, expected_volatility = op.main(simulation_month, years_data, min_position_size, max_position_size, pricing_model, objective_function, target_return, risk_model)
    # render_template("index.html")
    return redirect('/download')



  return redirect('/')


if __name__ == '__main__':
  app.run(host="localhost", port=8000, debug=True)