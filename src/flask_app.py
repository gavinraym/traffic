from flask import Flask, request, render_template, Response
import pandas as pd
import sys
import numpy as np
import jinja2
import json
import os
import ast
from PIL import Image
import pickle
from model import Model
import PIL

app = Flask(__name__)
class_map = pd.read_csv('meta/meta_map.csv', index_col=0)

@app.route('/', methods= ['GET', 'POST'])
def home_page():
    
    html = f'''
        <html><center>
            <head>
                <img src="meta/0.png"  />
                <h1>Image Classification</h1>
            </head>
        '''

    with open('models/index.txt', 'r') as f:
        info_list = f.readlines()

    for line in info_list:
        info = ast.literal_eval(line)

        html = html + f'''
        <body><center>
            <h4>[{info['id']}] {info['name']} - {info['timestamp']}</h4>
            <h6>Score: {info['Accuracy score']} ({info['True Predictions']}/{info['Total Predictions']})
            <form action="/report" method="POST" >
                <input type="submit" name="{info['id']}" value="Generate Report" />
            </form>
        </center></body>
        '''

    return f'{html}\n</html>'


@app.route('/report', methods=['GET','POST'])
def report_page():
    info = None
    model = None

    with open('models/index.txt', 'r') as f:
        info_list = f.readlines() 
    for line in info_list:
        info = ast.literal_eval(line)
    
        try:
            if 'Generate Report' == request.form[str(info['id'])]:
                
                with open(info['pickle path'], 'rb') as fb:
                    model = pickle.load(fb)
                break
        except:
            pass

    

    html = f'''
    <html>
        <head>
            <h1>Model name: {model.name}</h1>
            <h2>Created on: {model.timestamp}
            <h2>Accuracy Score: {info['Accuracy score']} ({info['True Predictions']} / {info['Total Predictions']})
        </head>
        <body>
            <img src="{{model.graph}}"">
        </body>
    '''
    # preds data
    preds = model.preds.dropna()

    # Each individual class data is displayed
    for sign in range(43):
        sign_preds = preds[preds.actual_class == sign]
        group_preds = sign_preds.groupby('predicted_class').count()

        html = html+f'''
        <body><br><br>-------------------------------------------------<br>
            <h3>Target Class #{sign} - {class_map.iloc[sign][0]}</h3>
            <img src="meta/{sign}.png" >
            <h3>Accuracy score = {int((group_preds.loc[sign][0] / len(sign_preds))*100)}%</h3>
            <h3>Predicted as (class, number):</h3>
            <h5>          
            '''
        for c in group_preds.index:
            html = html + f' ({c} : {group_preds.loc[c].actual_class})<br>'
        html = html+f'</h5><br><br>'
    return html


app.run(host='0.0.0.0', port=8080, debug=True)
    