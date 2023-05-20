# -*- coding: utf-8 -*-
"""
Script for performing vacancy parking spot detection and running a Flask server
 that displays the amount of detected vacant parking spots on a map.

Author: Bc. Marek Sicha
"""

import torch
import detectron2
import numpy as np
import os
import json
import cv2
import hashlib

from flask import Flask, render_template, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

app = Flask(__name__)


def get_image_from_camera(url):
    """
    Get the image frame from the camera URL.

    Args:
        url: URL of the camera stream.

    Returns:
        Resized image frame if successful, otherwise None.
    """
    try:
        cap = cv2.VideoCapture(url)
        ret, frame = cap.read()
        cap.release()
        return cv2.resize(frame, (1088, 640))
    except Exception:
        return None


def run_predictions():
    """
    Perform object detection predictions on camera images
    and update the cameras .json file.
    """
    with open('cameras.json') as f:
        data = json.load(f)
    previous_hash = hashlib.md5(json.dumps(data).
                                encode('utf-8')).hexdigest()
    predictor = DefaultPredictor(cfg)
    for camera in data['cameras']:
        frame = get_image_from_camera(camera['url'])
        if frame is not None:
            outputs = predictor(frame)
            camera['spots'] = len(outputs["instances"])
        else:
            camera['spots'] = 0
    with open('cameras.json', 'r') as f:
        c_data = json.load(f)
    current_hash = hashlib.md5(json.dumps(c_data).
                               encode('utf-8')).hexdigest()
    if current_hash == previous_hash:
        with open('cameras.json', 'w') as f:
            json.dump(data, f, indent=4)
    else:
        c_data['cameras'] = [{**cam, 'spots': camera['spots']}
                             if cam['uniqueID'] == camera['uniqueID']
                             else cam for cam in c_data['cameras']]
        with open('cameras.json', 'w') as f:
            json.dump(c_data, f, indent=4)
    del predictor
    torch.cuda.empty_cache()


@app.route('/')
def index():
    """
    Route decorator for the root URL ("/").

    Returns:
        The rendered HTML template for the index page.
    """
    return render_template('index.html')


@app.route('/get_markers')
def get_markers():
    """
    Route decorator for the "/get_markers" URL.

    Returns:
        The JSON response contains the data from the "cameras.json" file.
    """
    with open('cameras.json') as f:
        data = json.load(f)
    return jsonify(data)


if __name__ == '__main__':
    cfg = get_cfg()
    cfg.merge_from_file('config.yaml')
    cfg.MODEL.WEIGHTS = os.path.join("model_0001399.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(run_predictions, 'interval', minutes=2,
                  misfire_grace_time=60, max_instances=1)
    sched.start()
    app.run()
