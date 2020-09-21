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
import PIL
import matplotlib.pyplot as plt

app = Flask(__name__)

class Model():
    pass
    
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
    with open('app/static/meta_images/meta_map.txt', 'r') as f:
        meta_map = ast.literal_eval(f.readline())

    with open('models/index.txt', 'r') as f:
        info_list = f.readlines() 
    for line in info_list:
        summary = ast.literal_eval(line)
        # breakpoint()
        try:
            if 'Generate Report' == request.form[str(summary['id'])]:
                break
        except:
            pass

    with open(summary['pickle path'], 'rb') as fb:
        model = pickle.load(fb)

    wrong = model.preds[model.preds.actual_class != model.preds.predicted_class]
    w_samples = wrong.image_location.sample(min(len(wrong),75))
 
    right = model.preds[model.preds.actual_class == model.preds.predicted_class]
    r_samples = right.image_location.sample(min(len(right),75))

    return render_template(
        'report.html',
        summary=summary,
        fit_time = model.fit_time,
        description = model.description,
        filters = model.filters,
        architecture = model.architecture,
        model_summary = model.summary,
        wrong_list=w_samples,
        right_list=r_samples,
        meta_map=meta_map,
        most_tp = list(model.class_data.most_tp),
        least_tp = list(model.class_data.least_tp),
        acc = list(model.class_data.acc),
        class_preds = list(model.class_data.class_pred)
        )

@app.route('/final', methods=['GET','POST'])
def final():
    final_eval = list()
    with open('app/static/final_report/final_eval.txt', 'r') as f:
        for line in f.readlines():
            final_eval.append(line.split(','))
    return render_template('final.html', final_eval=final_eval)

@app.route('/about', methods=['GET','POST'])
def about():
    return render_template('about.html')

app.run(host='0.0.0.0', port=8080, debug=True)