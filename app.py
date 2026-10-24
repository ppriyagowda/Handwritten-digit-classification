from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, session
)
import numpy as np
from PIL import Image, ImageOps
import base64
import io
import sqlite3
import os
import json
from datetime import datetime
from functools import wraps
import tensorflow as tf


# ==========================================
# 1. FLASK CONFIGURATION
# ==========================================

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "change-this-secret-key-before-deployment"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_model.h5")
DATABASE_PATH = os.path.join(BASE_DIR, "predictions.db")

# Replace with your model's actual evaluated test accuracy.
MODEL_ACCURACY = 99.21


# ==========================================
# 2. LOAD CNN MODEL
# ==========================================

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("CNN model loaded successfully!")

except Exception as error:
    model = None
    print("Model loading error:", error)


# ==========================================
# 3. DATABASE CONNECTION
# ==========================================

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ==========================================
# 4. CREATE / UPDATE DATABASE
# ==========================================

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            digit INTEGER NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL,
            username TEXT DEFAULT 'priya',
            top3 TEXT,
            feedback INTEGER DEFAULT NULL,
            actual_digit INTEGER DEFAULT NULL
        )
    """)

    # Support databases created by your previous app.py.
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [row["name"] for row in cursor.fetchall()]

    if "username" not in columns:
        cursor.execute(
            "ALTER TABLE predictions ADD COLUMN username TEXT DEFAULT 'priya'"
        )

    if "top3" not in columns:
        cursor.execute(
            "ALTER TABLE predictions ADD COLUMN top3 TEXT"
        )

    if "feedback" not in columns:
        cursor.execute(
            "ALTER TABLE predictions ADD COLUMN feedback INTEGER DEFAULT NULL"
        )

    if "actual_digit" not in columns:
        cursor.execute(
            "ALTER TABLE predictions ADD COLUMN actual_digit INTEGER DEFAULT NULL"
        )

    # Keep existing records visible to the demo account.
    cursor.execute("""
        UPDATE predictions
        SET username = 'priya'
        WHERE username IS NULL OR username = ''
    """)

    conn.commit()
    conn.close()

    print("Database initialized successfully!")


init_db()


# ==========================================
# 5. LOGIN PROTECTION
# ==========================================

def login_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            if request.path.startswith("/predict") or \
               request.path.startswith("/history") or \
               request.path.startswith("/analytics") or \
               request.path.startswith("/feedback"):
                return jsonify({"error": "Please log in first."}), 401

            return redirect(url_for("login_page"))

        return function(*args, **kwargs)

    return wrapper


# ==========================================
# 6. LOGIN PAGE
# ==========================================

@app.route("/")
def login_page():
    if "user" in session:
        return redirect(url_for("home"))

    return render_template("login.html")


# ==========================================
# 7. LOGIN
# ==========================================

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    # Demo credentials
    if username == "priya" and password == "0818":
        session["user"] = username
        return redirect(url_for("home"))

    return """
        <h3>Invalid username or password.</h3>
        <a href="/">Try again</a>
    """, 401


# ==========================================
# 8. HOME DASHBOARD
# ==========================================

@app.route("/home")
@login_required
def home():
    return render_template(
        "index.html",
        accuracy=MODEL_ACCURACY,
        username=session["user"]
    )


# ==========================================
# 9. PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():

    try:
        if model is None:
            return jsonify({
                "error": "CNN model could not be loaded. Check mnist_model.h5."
            }), 500

        data = request.get_json(silent=True)

        if not data or not data.get("image"):
            return jsonify({"error": "Please draw a digit first."}), 400

        image_data = data["image"]

        if not isinstance(image_data, str) or "," not in image_data:
            return jsonify({"error": "Invalid image data."}), 400

        # Decode browser canvas image
        encoded_image = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(encoded_image, validate=True)

        image = Image.open(io.BytesIO(image_bytes)).convert("L")

        # Canvas uses white drawing on black background.
        # MNIST normally uses white digit on black background after
        # inversion of a black digit on white background.
        image = ImageOps.invert(image)
        image = image.resize((28, 28))

        image_array = np.array(image, dtype=np.float32) / 255.0

        # Support common model input shapes.
        input_shape = model.input_shape

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if len(input_shape) == 4:
            image_array = image_array.reshape(1, 28, 28, 1)
        elif len(input_shape) == 3:
            image_array = image_array.reshape(1, 28, 28)
        elif len(input_shape) == 2:
            image_array = image_array.reshape(1, 784)
        else:
            return jsonify({
                "error": "Unsupported CNN model input shape."
            }), 500

        probabilities = np.asarray(
            model.predict(image_array, verbose=0)[0]
        )

        if probabilities.size < 3:
            return jsonify({
                "error": "The model must return at least three digit classes."
            }), 500

        # Top three predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]

        digit = int(top_indices[0])
        confidence = float(probabilities[digit]) * 100

        top3 = [
            {
                "digit": int(index),
                "confidence": round(float(probabilities[index]) * 100, 2)
            }
            for index in top_indices
        ]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save prediction to SQLite
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions
                (digit, confidence, timestamp, username, top3)
            VALUES (?, ?, ?, ?, ?)
        """, (
            digit,
            confidence,
            timestamp,
            session["user"],
            json.dumps(top3)
        ))

        prediction_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "id": prediction_id,
            "digit": digit,
            "confidence": round(confidence, 2),
            "top3": top3,
            "timestamp": timestamp
        })

    except Exception as error:
        print("Prediction error:", repr(error))

        return jsonify({
            "error": "Prediction failed. Check the PyCharm terminal."
        }), 500


# ==========================================
# 10. PREDICTION HISTORY
# ==========================================

@app.route("/history")
@login_required
def history():

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, digit, confidence, timestamp
        FROM predictions
        WHERE username = ?
        ORDER BY id DESC
        LIMIT 10
    """, (session["user"],))

    rows = cursor.fetchall()
    conn.close()

    return jsonify([
        [
            row["id"],
            row["digit"],
            round(row["confidence"], 2),
            row["timestamp"]
        ]
        for row in rows
    ])


# ==========================================
# 11. ANALYTICS DASHBOARD
# ==========================================

@app.route("/analytics")
@login_required
def analytics():

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) AS total,
               AVG(confidence) AS average_confidence
        FROM predictions
        WHERE username = ?
    """, (session["user"],))

    summary = cursor.fetchone()

    cursor.execute("""
        SELECT digit, COUNT(*) AS count
        FROM predictions
        WHERE username = ?
        GROUP BY digit
    """, (session["user"],))

    digit_counts = [0] * 10

    for row in cursor.fetchall():
        digit_counts[row["digit"]] = row["count"]

    cursor.execute("""
        SELECT COUNT(*) AS evaluated,
               SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) AS correct
        FROM predictions
        WHERE username = ? AND feedback IS NOT NULL
    """, (session["user"],))

    feedback_stats = cursor.fetchone()
    conn.close()

    evaluated = feedback_stats["evaluated"] or 0
    correct = feedback_stats["correct"] or 0

    feedback_accuracy = (
        round(correct / evaluated * 100, 2)
        if evaluated else None
    )

    return jsonify({
        "total_predictions": summary["total"] or 0,
        "digit_counts": digit_counts,
        "average_confidence": round(
            summary["average_confidence"], 2
        ) if summary["average_confidence"] is not None else 0,
        "model_accuracy": MODEL_ACCURACY,
        "feedback_evaluated": evaluated,
        "feedback_correct": correct,
        "feedback_accuracy": feedback_accuracy
    })


# ==========================================
# 12. USER FEEDBACK
# ==========================================

@app.route("/feedback", methods=["POST"])
@login_required
def feedback():

    data = request.get_json(silent=True) or {}

    try:
        prediction_id = int(data["prediction_id"])
        actual_digit = int(data["actual_digit"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide a valid prediction_id and actual_digit."
        }), 400

    if not 0 <= actual_digit <= 9:
        return jsonify({
            "error": "Actual digit must be between 0 and 9."
        }), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT digit
        FROM predictions
        WHERE id = ? AND username = ?
    """, (prediction_id, session["user"]))

    row = cursor.fetchone()

    if row is None:
        conn.close()
        return jsonify({"error": "Prediction not found."}), 404

    correct = int(row["digit"] == actual_digit)

    cursor.execute("""
        UPDATE predictions
        SET feedback = ?, actual_digit = ?
        WHERE id = ? AND username = ?
    """, (
        correct,
        actual_digit,
        prediction_id,
        session["user"]
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "predicted_digit": row["digit"],
        "actual_digit": actual_digit,
        "correct": bool(correct),
        "message": "Feedback saved successfully."
    })


# ==========================================
# 13. LOGOUT
# ==========================================

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

# ==========================================
# 14. RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)