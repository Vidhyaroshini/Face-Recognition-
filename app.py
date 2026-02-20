from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import sqlite3
import cv2
import os
from flask import Flask,request,render_template,redirect,session,url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import cv2
import numpy as np
import time
import serial
from flask import Flask, redirect, url_for, request, render_template,session,flash,redirect, url_for, session,flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = '1a2b3c4d5e'


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)


DATABASE = "database.db"



#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points
    

def affender():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('login.html') 

    cap = cv2.VideoCapture(0)
    ret = True
    idc=""
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            if (idc!=identified_person):
                #add_affender(identified_person)
                idc=identified_person
                print(idc)
                #return render_template('result.html',totalreg1=totalreg(),datetoday3=datetoday2)
            
            cv2.putText(frame,f'{identified_person}',(x + 6, y - 6),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2)
            # img, name, , , 1, (255, 255, 255), 2

        # Display the resulting frame
        cv2.imshow('Face Check', frame)
        cv2.putText(frame,'hello',(30,30),cv2.FONT_HERSHEY_COMPLEX,2,(255, 255, 255))
        
    # Wait for the user to press 'q' to quit
        if cv2. waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return idc

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/",methods=["GET", "POST"])
def index():
    if request.method=='POST':
        un=request.form['username']
        pwd=request.form['password']
        if un=="admin" and pwd=="admin":
            return redirect(url_for("dashboard"))
            
    return render_template("login.html")

@app.route("/result")
def result():
    idc=affender()
    return redirect(url_for("dashboard"))
    
@app.route("/dashboard")
def dashboard():
    conn = get_db()
    cursor = conn.execute("SELECT COUNT(*) as total FROM criminals")
    total_criminals = cursor.fetchone()["total"]
    conn.close()
    return render_template("dashboard.html", total_criminals=total_criminals)

@app.route("/criminals")
def criminal_list():
    conn = get_db()
    criminals = conn.execute("SELECT * FROM criminals").fetchall()
    conn.close()
    return render_template("criminal_list.html", criminals=criminals)

@app.route("/add", methods=["GET", "POST"])
def add_criminal():
    if request.method == "POST":
        name = request.form["name"]
        name1=name
        phone = request.form["phone"]
        email = request.form["email"]
        address = request.form["adrs"]
        gender = request.form["gender"]
        age = request.form["age"]
        crime = request.form["crime"]
        userimagefolder = 'static/faces/'+name+'_'+str(phone)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        cap = cv2.VideoCapture(0)
        i,j = 0,0
        while 1:
            _,frame = cap.read()
            faces = extract_faces(frame)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                if j%10==0:
                    name = name1+'_'+str(i)+'.jpg'
                    print(name)
                    cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                    
                    i+=1
                j+=1
            if j==500:
                break
            cv2.imshow('Adding new User',frame)
            if cv2.waitKey(1)==27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        train_model()
        conn = get_db()
        conn.execute("INSERT INTO criminals (name,phone,email,address,gender, age, crime) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                     (name1,phone,email,address,gender, age, crime))
        conn.commit()
        conn.close()
        return redirect(url_for("criminal_list"))
    return render_template("add_criminal.html")

@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit_criminal(id):
    conn = get_db()
    criminal = conn.execute("SELECT * FROM criminals WHERE id=?", (id,)).fetchone()
    if request.method == "POST":
        name = request.form["name"]
        name1=name
        phone = request.form["phone"]
        email = request.form["email"]
        address = request.form["adrs"]
        gender = request.form["gender"]
        age = request.form["age"]
        crime = request.form["crime"]
        conn.execute("UPDATE criminals SET name=?,phone=?,email=?,address=?,gender=?, age=?, crime=? WHERE id=?", 
                     (name,phone,email,address,gender, age, crime, id))
        conn.commit()
        conn.close()
        return redirect(url_for("criminal_list"))
    conn.close()
    return render_template("edit_criminal.html", criminal=criminal)

@app.route("/delete/<int:id>")
def delete_criminal(id):
    conn = get_db()
    conn.execute("DELETE FROM criminals WHERE id=?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for("criminal_list"))

if __name__ == "__main__":
    # Initialize database if not exists
    conn = sqlite3.connect(DATABASE)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS criminals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone INTEGER,
        email TEXT NOT NULL,
        address TEXT NOT NULL,
        gender TEXT NOT NULL,
        age INTEGER,
        crime TEXT NOT NULL
    )
    """)
    conn.close()
    app.run(debug=True)
