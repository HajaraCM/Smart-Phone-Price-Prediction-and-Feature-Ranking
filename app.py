from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("rank_model.pkl", "rb"))
scaler=pickle.load(open("scaling_features.pkl","rb"))



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods = ["GET", "POST"])

def predict():
    
        battery_power=request.form['battery']
        blue=request.form['blue']
        if (blue=='Yes'):
            blue=1
        else: blue=0
        clock_speed=request.form['clock']
        dual_sim=request.form['sim']
        if (dual_sim=='Yes'):
            dual_sim=1
        else: dual_sim=0
        fc=request.form['frontcamera']
        four_g=request.form['4G']
        if (four_g=='Yes'):
            four_g=1
        else: four_g=0
        int_memory=request.form['memory']
        m_dep=request.form['depth']
        mobile_wt=request.form['weight']
        n_cores=request.form['cores']
        pc=request.form['camera']
        px_height=request.form['height']
        px_width=request.form['width']
        ram=request.form['ram']
        sc_h=request.form['srheight']
        sc_w=request.form['srwidth']
        talk_time=request.form['time']
        three_g=request.form['3G']
        if (three_g=='Yes'):
            three_g=1
        else: three_g=0
        touch_screen=request.form['touchscreen']
        if (touch_screen=='Yes'):
            touch_screen=1
        else: touch_screen=0
        wifi=request.form['wifi']
        if (wifi=='Yes'):
            wifi=1
        else: wifi=0

        df={' battery_power':[ battery_power],'blue':[blue],'clock_spped':[clock_speed],'dual_sim':[dual_sim],'fc':[fc],
              'four_g':[four_g],'int_memory':[int_memory],'m_dep':[m_dep],'mobile_wt':[mobile_wt],'n_cores':[n_cores],'pc':[pc],
              'px_height':[px_height],'px_width':[px_width],'ram':[ram],'sc_h':[sc_h],'sc_w':[sc_w],'talk_time':[talk_time],
              'three_g':[three_g],'touch_screen':[touch_screen],'wifi':[wifi]}
        data=pd.DataFrame(data=df)
        
        print(data)
        input_data= scaler.transform(data)

        prediction=model.predict(input_data)
        pred=prediction[0]
        output='Error'
        if pred==0:
            output='Your expected price range for this smartphone is: Low Cost'
        elif pred==1:
            output='Your expected price range for this smartphone is: Medium Cost'
        elif pred==2:
            output='Your expected price range for this smartphone is: High Cost'
        else :
            output='Your expected price range for this smartphone is:Very High Cost'
        coefficients = model.coef_

        avg_importance =np.mean(np.abs(coefficients),axis=0)
        log_feature_importance = pd.DataFrame({'Feature': data.columns, 'Importance': avg_importance})
        log_feature_importance = log_feature_importance.sort_values('Importance', ascending=False)
        log_feature_importance['Rank']=log_feature_importance['Importance'].rank(ascending=False)
        
        return render_template('after.html',results=output, tables=[log_feature_importance.to_html(classes='data',header="true", index=False)])
                                                                    #titles=log_feature_importance.columns.values)
    
   
if __name__== "__main__":
    app.run(debug=True)


        
        
        
        