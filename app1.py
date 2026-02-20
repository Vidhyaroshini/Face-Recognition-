from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import cv2
import os
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = '1a2b3c4d5e'

DATABASE = "database.db"

# threshold for unknown face detection
UNKNOWN_THRESHOLD = 4000

# Load face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# -------------------------------------------------------------------
# FACE EXTRACTION
# -------------------------------------------------------------------
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


# -------------------------------------------------------------------
# FACE IDENTIFICATION (UPDATED WITH UNKNOWN DETECTION)
# -------------------------------------------------------------------
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')

    # Predict class
    pred = model.predict(facearray)[0]

    # Calculate nearest distance
    distances, indices = model.kneighbors(facearray, n_neighbors=1)
    dist = distances[0][0]

    return pred, dist


# -------------------------------------------------------------------
# FACE RECOGNITION LIVE CAMERA (UPDATED)
# -------------------------------------------------------------------
def affender():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('login.html')

    cap = cv2.VideoCapture(0)
    idc = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            pred, dist = identify_face(face.reshape(1, -1))
            print(dist)

            if dist > UNKNOWN_THRESHOLD:
                person_name = "UNKNOWN"
            else:
                person_name = pred

            if idc != person_name:
                idc = person_name
                print("Detected:", idc)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, person_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

        cv2.imshow('Face Check', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return idc


# -------------------------------------------------------------------
# TRAINING MODEL
# -------------------------------------------------------------------
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')

    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized = cv2.resize(img, (50, 50))
            faces.append(resized.ravel())
            labels.append(user)

    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# -------------------------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


# -------------------------------------------------------------------
# ROUTES (UNMODIFIED)
# -------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        un = request.form['username']
        pwd = request.form['password']
        if un == "admin" and pwd == "admin":
            return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/result")
def result():
    idc = affender()
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
        phone = request.form["phone"]
        email = request.form["email"]
        address = request.form["adrs"]
        gender = request.form["gender"]
        age = request.form["age"]
        crime = request.form["crime"]

        folder = f'static/faces/{name}_{phone}'
        if not os.path.isdir(folder):
            os.makedirs(folder)

        cap = cv2.VideoCapture(0)
        i, j = 0, 0

        while True:
            _, frame = cap.read()
            faces = extract_faces(frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images: {i}/50', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

                if j % 10 == 0:
                    imgname = f'{name}_{i}.jpg'
                    cv2.imwrite(f'{folder}/{imgname}', frame[y:y + h, x:x + w])
                    i += 1
                j += 1

            if i >= 50 or j >= 500:
                break

            cv2.imshow('Adding User', frame)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        train_model()

        conn = get_db()
        conn.execute("""INSERT INTO criminals (name,phone,email,address,gender, age, crime)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                     (name, phone, email, address, gender, age, crime))
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
        phone = request.form["phone"]
        email = request.form["email"]
        address = request.form["adrs"]
        gender = request.form["gender"]
        age = request.form["age"]
        crime = request.form["crime"]

        conn.execute("""UPDATE criminals SET name=?,phone=?,email=?,address=?,gender=?,age=?,crime=? WHERE id=?""",
                     (name, phone, email, address, gender, age, crime, id))
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


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
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
