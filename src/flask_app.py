from flask import Flask, request, render_template, Response
import pandas as pd
import sys
import numpy as np
import jinja2
import json
import os
import ast
from PIL import Image

app = Flask(__name__)
class_map = pd.read_csv('data/meta/meta_map.csv', index_col=0)


@app.route('/', methods= ['GET', 'POST'])
def home_page():
    
    html = f'''
        <html><center>
            <head>
                <img src={{url_for('static', filename='0.png') }} >
                <h1>Image Classification</h1>
            </head>
        '''

    with open('reports/index.txt', 'r') as f:
        models = f.readlines()
    for model in models:
        model = ast.literal_eval(model)
        html = html + f'''
        <body><center>
            <h4>[{model['id']}] {model['name']} - {model['timestamp']}</h4>
            <h6>Score: {model['Accuracy score']} ({model['True Predictions']}/{model['Total Predictions']})
            <form action="/report" method="POST" >
                <input type="submit" name="{model['id']}" value="Generate Report" />
            </form>
        </center></body>
        '''

    return f'{html}\n</html>'


@app.route('/report', methods=['GET','POST'])
def report_page():
    model = None
    with open('reports/index.txt', 'r') as f:
        model_list = f.readlines() 
    for line in model_list:
        line = ast.literal_eval(line)
        try:
            if 'Generate Report' == request.form[line['id']]:
                model = line
                break
        except:
            pass

    html = f'''
    <html>
        <head>
            <h1>Model name: {model['name']}</h1>
            <h2>Created on: {model['timestamp']}
            <h2>Accuracy Score: {model['Accuracy score']} ({model['True Predictions']} / {model['Total Predictions']})
        </head>
    '''

    # This data is used for class specific reporting
    preds = pd.read_csv(model['path'], index_col=0)

    # Each individual class data is displayed
    for sign in range(43):
        sign_preds = preds[preds.actual_class == sign]
        group_preds = sign_preds.groupby('predicted_class').count()

        html = html+f'''
        <body><br><br>-------------------------------------------------<br>
            <h3>Target Class #{sign} - {class_map.iloc[sign][0]}</h3>
            <img src="data/meta/{sign}.png" >
            <h3>Accuracy score = {int((group_preds.loc[sign][0] / len(sign_preds))*100)}%</h3>
            <h3>Predicted as (class, number):</h3>
            <h5>          
            '''
        for c in group_preds.index:
            html = html + f' ({c} : {group_preds.loc[c].actual_class})<br>'
        html = html+f'</h5><br><br>'
    return html


app.run(host='0.0.0.0', port=8080, debug=True)
    