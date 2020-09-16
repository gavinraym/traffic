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
import matplotlib.pyplot as plt

app = Flask(__name__)

with open('app/static/meta/meta_map.txt', 'r') as f:
    meta_map = ast.literal_eval(f.readline())
    
@app.route("/")
def main():

    info_list = list()
    with open('models/index.txt', 'r') as f:
        index_list = f.readlines()
    for line in index_list:
        info_list.append(ast.literal_eval(line))
    return render_template('home.html', info_list=info_list)


@app.route('/report', methods=['GET','POST'])
def report_page():
    summary = None
    model = None

    with open('models/index.txt', 'r') as f:
        info_list = f.readlines() 
    for line in info_list:
        summary = ast.literal_eval(line)
    
        try:
            if 'Generate Report' == request.form[str(summary['id'])]:
                break
        except:
            pass

    with open(summary['pickle path'], 'rb') as fb:
        model = pickle.load(fb)
    model.graph.savefig('app/static/images/graph.png')
    model.heat_map.savefig('app/static/images/heat_map.png')
    model.ROC.savefig('app/static/images/roc.png')

    wrong = model.preds[model.preds.actual_class != model.preds.predicted_class]
    



    return render_template('report.html', summary=summary, wrong_list=wrong.image_location.sample(25))

    
app.run(host='0.0.0.0', port=8080, debug=True)