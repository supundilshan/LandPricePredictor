from flask import Flask, redirect, url_for, request , render_template
import numpy as np
import sklearn
import pickle
import sqlite3 as sql
import threading

from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor
app = Flask(__name__)

def Train_GBM():
    print("Train")
    con = sql.connect('DataBase/LandData.db')
    c = con.cursor()  # cursor

    c.execute("SELECT * FROM DataTable;")
    all_result = c.fetchall()

    Train_dataset = np.array(all_result)
    X_train = Train_dataset[:, 1:8]
    Y_train = Train_dataset[:, 0]

    # define the model
    # model = GradientBoostingRegressor(n_estimators = 50, learning_rate = 0.1, subsample = 1, max_depth = 6)
    model = GradientBoostingRegressor()
    # define the evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model
    n_scores = cross_val_score(model, X_train, Y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    # report performance
    print('MSE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    # fit the model on the whole dataset
    model.fit(X_train, Y_train)

    # save the model to disk
    filename = 'GBM_Model/finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

@app.route('/Saved/<massage>')
def Saved(massage):
   return massage

@app.route('/')
def Index():
   return render_template('Get_LandData.html')
   # return 'Distance to road %f' % Predict

@app.route('/Index/<Inputs_str>')
def Index2(Inputs_str):
    Inputs = np.fromstring(Inputs_str, dtype=int, sep=',')
    return render_template('Get_LandData.html' , road= Inputs[0], city = Inputs[1], school= Inputs[2], hospital= Inputs[3], bus= Inputs[4], train= Inputs[5], market= Inputs[6], area= Inputs[7])
    # return 'Distance to road %f' % Predict

@app.route('/LandPrice/<Inputs_str>')
def Calculate_Land_Price(Inputs_str):
    # Inputs = Inputs_str.astype(np.float)
    Inputs=np.fromstring(Inputs_str, dtype=int, sep=',')
    Data = np.delete(Inputs, 7)
    Data = Data.reshape(1, -1)
    # load the model from disk
    try:
        loaded_model = pickle.load(open('GBM_Model/finalized_model.sav', 'rb'))
    except:
        loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    Predicted_Landprice = loaded_model.predict(Data)
    # return render_template('LandData.html', land_price=Predicted_Landprice)
    Land_Price = round(Predicted_Landprice[0]*10**5)*Inputs[7]
    return render_template('LandData_Confirm.html' , land_price = Predicted_Landprice[0], rounded_land_price= Land_Price, road= Inputs[0], city = Inputs[1], school= Inputs[2], hospital= Inputs[3], bus= Inputs[4], train= Inputs[5], market= Inputs[6],area= Inputs[7])

@app.route('/Get_Distance_Inputs',methods = ['POST', 'GET'])
def Get_Distance_Inputs():
    if request.method == 'POST':
       Road = request.form['road']
       City = request.form['city']
       School = request.form['school']
       Hospital = request.form['hospital']
       Bus = request.form['bus']
       Train = request.form['train']
       Market = request.form['market']
       Area = request.form['area']
    else:
       Road = request.args.get('road')
       City = request.args.get('city')
       School = request.args.get('school')
       Hospital = request.args.get('hospital')
       Bus = request.args.get('bus')
       Train = request.args.get('train')
       Market = request.args.get('market')
       Area = request.args.get('market')

    Distances = np.array([str(Road),",", str(City),",", str(School),",",str(Hospital),",", str(Bus), ",",str(Train),",", str(Market),",", str(Area)])
    Distances_str = ''.join(Distances)
    # return Dist
    return redirect(url_for('Calculate_Land_Price', Inputs_str=Distances_str))

@app.route('/Save_To_DB',methods = ['POST', 'GET'])
def Save_To_DB():
    if request.method == 'POST':
       Road = request.form['road']
       City = request.form['city']
       School = request.form['school']
       Hospital = request.form['hospital']
       Bus = request.form['bus']
       Train = request.form['train']
       Market = request.form['market']
       User_Responce = request.form['submit_button']
       Prediction = request.form['Prediction']
       Area = request.form['area']
    else:
       Road = request.args.get('road')
       City = request.args.get('city')
       School = request.args.get('school')
       Hospital = request.args.get('hospital')
       Bus = request.args.get('bus')
       Train = request.args.get('train')
       Market = request.args.get('market')
       User_Responce = request.args.get('submit_button')
       Prediction = request.args.get('Prediction')
       Area = request.args.get('market')

    # connect to qa_database.sq (database will be created, if not exist)
    con = sql.connect('DataBase/LandData.db')
    con.execute('CREATE TABLE IF NOT EXISTS DataTable (LandPrice REAL NOT NULL,Road REAL NOT NULL, City REAL NOT NULL, School REAL NOT NULL, Hospital REAL NOT NULL, Bus REAL NOT NULL, Train REAL NOT NULL, Market REAL NOT NULL)')
    con.close

    if User_Responce == 'YES':
        try:
           con = sql.connect('DataBase/LandData.db')
           c = con.cursor()  # cursor
           # insert data
           c.execute("INSERT INTO DataTable (LandPrice,Road, City, School, Hospital, Bus, Train, Market) VALUES (?,?,?,?,?,?,?,?)",(Prediction,Road, City, School, Hospital, Bus, Train, Market))
           con.commit()  # apply changes

           Distances = np.array([str(Road), ",", str(City), ",", str(School), ",", str(Hospital), ",", str(Bus), ",", str(Train), ",",str(Market), ",",str(Area)])
           Distances_str = ''.join(Distances)
           # return Dist
           Call_GBM_Train = threading.Thread(target=Train_GBM)
           if Call_GBM_Train.is_alive() is False:
               Call_GBM_Train = threading.Thread(target=Train_GBM)
               Call_GBM_Train.start()
           # Train_GBM()
           return redirect(url_for('Index2', Inputs_str=Distances_str))
        except con.Error as err:  # if error
            # then display the error in 'database_error.html' page
            return redirect(url_for('Saved', massage=err))
        finally:
           con.close()  # close the connection
    else:
        Distances = np.array([str(Road), ",", str(City), ",", str(School), ",", str(Hospital), ",", str(Bus), ",", str(Train), ",",str(Market), ",",str(Area)])
        Distances_str = ''.join(Distances)
        # return Dist
        return redirect(url_for('Index2', Inputs_str=Distances_str))

if __name__ == '__main__':
   app.run(debug=True,
         host='0.0.0.0',
         port=9000,
         threaded=True)