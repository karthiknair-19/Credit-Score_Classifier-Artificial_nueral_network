from flask import Flask,render_template,url_for,redirect,request
import numpy
import tensorflow as tf
import joblib

scaler=joblib.load('standardscaler.pkl')

model = tf.keras.models.load_model("model")

app=Flask(__name__)

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about_us')
def about():
    return render_template('about.html')      

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        Outstanding_Debt=request.form['Outstanding Debt']
        Annual_income=request.form['Annual Income']     
        Monthly_In_hand=request.form['Monthly In-hand']
        Interest_Rate=request.form['Interest Rate']
        Credit_Mix=request.form['Credit Mix']
        
        if Credit_Mix == 'Bad':
            Credit_Mix=0
        elif Credit_Mix == 'Good':
            Credit_Mix=1
        else:
            Credit_Mix=2
        x_test=[int(Annual_income)/80,int(Monthly_In_hand)/80,int(Interest_Rate),int(Credit_Mix)/80,Outstanding_Debt]         
        
        x=numpy.array(x_test,ndmin=2)
        
        x=scaler.transform(x) 
        
        y_pred=model.predict(x)
        
        ans=numpy.round(y_pred[0][0])
        
        if ans==0:
            
            return render_template('bcredit.html')
        
        else:
            
            return render_template('gcredit.html')

if __name__ == '__main__':
    app.run(debug=True,port=5000)