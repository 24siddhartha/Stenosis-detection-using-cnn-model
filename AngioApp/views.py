import matplotlib
matplotlib.use('Agg')
from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pickle
import pymysql
import os
from django.core.files.storage import FileSystemStorage
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

global uname
global filename, yolo_model
labels = ['Stenosis']
CONFIDENCE_THRESHOLD = 0.3
GREEN = (0, 255, 0)

yolo_model = YOLO("model/best.pt")

def getSegmented(image, x, y, w, h):
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segmentation_mask = image[y:h, x:w]
    segmented_image = np.zeros_like(image1)
    for j in range(y, h):
        for i in range(x, w):
            segmented_image[j,i] = image1[j,i]
    return segmented_image      

def getDetection(frame):
    global yolo_model, labels
    status = 0
    detections = yolo_model(frame)[0]
    area = 0
    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]
        cls_id = int(data[5])
        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue
        # if the confidence is greater than the minimum confidence,
        # draw the bounding box on the frame
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        if status == 0:
            segmentation_mask = getSegmented(frame.copy(), xmin, ymin, xmax, ymax)
            status = 1
        width = xmax - xmin
        height = ymax - ymin
        area = width * height
        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 3)
        cv2.putText(frame, labels[cls_id]+" "+str(round(confidence,2))+"%", (xmin, ymin-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.putText(frame, "Blockade Area Size = "+str(area), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)        
    if status == 0:
        cv2.putText(frame, "No Blockade Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        segmentation_mask = None
    return frame, status, segmentation_mask


def LoadDataset(request):
    if request.method == 'GET':
        global labels
        output = "Angiography Dataset Loaded<br/>"
        output += "Total images Found in Dataset = <font size=3 color=blue>8135 from 100 subjects</font><br/>"
        output += "Different Labels Found in Dataset = <font size=3 color=blue>"+str(labels)+"</font><br/>"
        context= {'data': output}
        return render(request, 'UserScreen.html', context)

def TrainModel(request):
    if request.method == 'GET':
        cnn_train_detection = cv2.imread("model/result.png")
        plt.figure(figsize=(12,7))
        plt.imshow(cnn_train_detection)
        plt.title("CNN Blood Vessel Detection Accuracy Graph")
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        #plt.close()
        context= {'data':"CNN Blood Vessel Detection Accuracy Graph", 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def PredictAction(request):
    if request.method == 'POST':
        global scaler, labels, rf_cls
        #facenet_model = load_model('model/facenet_keras.h5')
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists('AngioApp/static/'+fname):
            os.remove('AngioApp/static/'+fname)
        with open('AngioApp/static/'+fname, "wb") as file:
            file.write(myfile)
        file.close()
        img = cv2.imread('AngioApp/static/'+fname)
        img, status, segmentation_mask = getDetection(img)
        if status == 0:
            plt.imshow(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))#display original and predicted segmented image
            axis[0].set_title("Detection & Classification")
            axis[1].set_title("Segmentation")
            axis[0].imshow(img)
            axis[1].imshow(segmentation_mask,cmap="gray")
        plt.subplots_adjust(wspace=5, hspace=5)    
        plt.tight_layout()    
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data': "Detection & Classification Result", 'img': img_b64}
        return render(request, 'UserScreen.html', context)  


def Predict(request):
    if request.method == 'GET':
       return render(request, 'Predict.html', {})

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})    

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})   

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'angio',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username, password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'UserLogin.html', context)        
    
def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        contact = request.POST.get('contact', False)
        email = request.POST.get('email', False)
        address = request.POST.get('address', False)
        status = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'angio',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    status = "Username already exists"
                    break
        if status == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'angio',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = "Signup process completed"
        context= {'data': status}
        return render(request, 'Register.html', context)

