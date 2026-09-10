"""Веб-интерфейс системы распознавания лиц — запуск: python3 web_server.py"""
from __future__ import annotations

import base64
import os
import time
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from bank_face_system import FaceBank, FaceDetector, FeatureExtractor, MatchResult

app = Flask(__name__)
fb = FaceBank()
detector = FaceDetector()
DB_PATH = os.path.join(os.path.dirname(__file__), "facebank_db.pkl")

if os.path.exists(DB_PATH):
    fb.load(DB_PATH)


@app.route("/")
def index():
    return render_template("index.html", count=fb.customer_count)


@app.route("/api/status")
def status():
    return jsonify({"customers": fb.customer_count})


@app.route("/api/identify", methods=["POST"])
def identify():
    data = request.get_json()
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"found": False, "verdict": "Нет изображения"})

    img_bytes = base64.b64decode(img_b64.split(",")[-1] if "," in img_b64 else img_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    match = fb.identify(frame)
    face_rect = detector.detect_largest(frame)

    return jsonify({
        "found": match.found,
        "verdict": match.verdict,
        "confidence": round(match.confidence * 100, 1),
        "face_detected": face_rect is not None,
        "customer": match.customer.full_name if match.customer else None,
        "customer_id": match.customer.customer_id if match.customer else None,
    })


@app.route("/api/enroll", methods=["POST"])
def enroll():
    data = request.get_json()
    img_b64 = data.get("image", "")
    cid = data.get("id", "").strip()
    fio = data.get("name", "").strip()

    if not cid or not fio:
        return jsonify({"ok": False, "error": "Введите ID и ФИО"})

    if not img_b64:
        return jsonify({"ok": False, "error": "Нет изображения"})

    img_bytes = base64.b64decode(img_b64.split(",")[-1] if "," in img_b64 else img_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    photo = os.path.join(os.path.dirname(__file__), f"enroll_{cid}_{int(time.time())}.jpg")
    cv2.imwrite(photo, frame)

    customer = fb.enroll(photo, cid, fio)
    fb.save(DB_PATH)

    if customer:
        return jsonify({"ok": True, "name": fio, "id": cid})
    return jsonify({"ok": False, "error": "Лицо не найдено на фото"})


@app.route("/api/customers")
def customers():
    return jsonify({"count": fb.customer_count, "list": fb.list_customers()})


@app.route("/api/save")
def save():
    fb.save(DB_PATH)
    return jsonify({"ok": True})


if __name__ == "__main__":
    print(f"\n  Откройте в браузере: http://localhost:8080\n")
    app.run(host="0.0.0.0", port=8080, debug=False)
