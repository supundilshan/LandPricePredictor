from flask import Flask, redirect, url_for, request , render_template
import numpy as np
import sqlite3 as sql
import sklearn
import pickle
app = Flask(__name__)

@app.route('/Index')
def Index1():
   return render_template('LandData.html' , land_price = "")
   # return 'Distance to road %f' % Predict

@app.route('/')
def Index():
   return render_template('LandData_Confirm.html' , land_price = "")
   # return 'Distance to road %f' % Predict

@app.route('/Saved/<massage>')
def Saved(massage):
   return massage

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
    else:
       Road = request.args.get('road')
       City = request.args.get('city')
       School = request.args.get('school')
       Hospital = request.args.get('hospital')
       Bus = request.args.get('bus')
       Train = request.args.get('train')
       Market = request.args.get('market')
       User_Responce = request.args.get('submit_button')

    # connect to qa_database.sq (database will be created, if not exist)
    con = sql.connect('DataBase/LandData.db')
    con.execute('CREATE TABLE IF NOT EXISTS DataTable (LandPrice REAL NOT NULL,Road REAL NOT NULL, City REAL NOT NULL, School REAL NOT NULL, Hospital REAL NOT NULL, Bus REAL NOT NULL, Train REAL NOT NULL, Market REAL NOT NULL)')
    con.close

    if User_Responce == 'YES':
        try:
           con = sql.connect('DataBase/LandData.db')
           c = con.cursor()  # cursor
           # insert data
           # c.execute("INSERT INTO DataTable (LandPrice,Road, City, School, Hospital, Bus, Train, Market) VALUES (?,?,?,?,?,?,?,?)",(111,Road, City, School, Hospital, Bus, Train, Market))
           # con.commit()  # apply changes
           return redirect(url_for('Saved', massage=User_Responce))
        except con.Error as err:  # if error
            # then display the error in 'database_error.html' page
            return redirect(url_for('Saved', massage=err))
        finally:
           con.close()  # close the connection

        return redirect(url_for('Calculate_Land_Price', Inputs_str=Distances_str))
    else:
        return redirect(url_for('Index1'))

if __name__ == '__main__':
   app.run(debug=True,
         host='0.0.0.0',
         port=9000,
         threaded=True)