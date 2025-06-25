import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, jsonify, render_template_string
from threading import Thread
from picamera2 import Picamera2
import time

# === Model Setup ===
model_path = "/home/pi/Downloads/best.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [1, 3, 864, 1536]

class_names = ["Stress"]  # Only one class

# === Flask Setup ===
app = Flask(__name__)
status_color = "yellow"
detected_classes = []

# === Post-processing ===
def postprocess(output, img_shape, conf_threshold=0.2):  # Lower threshold for more detections
    predictions = output[0].squeeze().T  # shape becomes (8400, 6)
    boxes = []
    image_height, image_width = img_shape

    print("📦 Predictions shape after squeeze+T:", predictions.shape)

    for i, pred in enumerate(predictions):
        x_center, y_center, width, height, objectness, class_score = pred
        confidence = float(objectness * class_score)

        if confidence > conf_threshold:
            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)

            boxes.append((x1, y1, x2, y2, confidence, 0))
            print(f"✅ Box {i}: {(x1, y1, x2, y2)} conf: {confidence:.2f}")

    if not boxes:
        print("❌ No detections above confidence threshold.")
    
    return boxes

# === Drawing Boxes ===
def draw_boxes(frame, boxes):
    global detected_classes, status_color
    detected_classes = []

    if len(boxes) == 0:
        status_color = "yellow"
    else:
        stress_detected = False
        for (x1, y1, x2, y2, conf, class_id) in boxes:
            label = f"{class_names[class_id]}: {conf:.2f}"
            color = (0, 0, 255)  # Red
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detected_classes.append(class_names[class_id])
            if class_names[class_id] == "Stress":
                stress_detected = True

        status_color = "red" if stress_detected else "green"

    return frame

# === Camera Setup ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (1536, 864), "format": "RGB888"}))
picam2.start()
time.sleep(2)

frame_buffer = None

def camera_loop():
    global frame_buffer
    while True:
        frame = picam2.capture_array()
        rgb = cv2.resize(frame, (input_shape[3], input_shape[2]))
        img_input = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32) / 255.0

        output = session.run(None, {input_name: img_input})
        boxes = postprocess(output, frame.shape[:2])
        preview = draw_boxes(frame.copy(), boxes)
        _, jpeg = cv2.imencode('.jpg', preview)
        frame_buffer = jpeg.tobytes()

# === Flask Routes ===
@app.route('/')
def index():
    return render_template_string('''
    <html>
      <head><title>Tomato Stress Detector</title></head>
      <body style="background-color: {{ color }}; font-family: sans-serif;">
        <h1>🍅 Tomato Leaf Stress Detection</h1>
        <img src="{{ url_for('video_feed') }}" width="90%">
        <p>Detected Classes: {{ detected }}</p>
      </body>
    </html>
    ''', color=status_color, detected=detected_classes)

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            if frame_buffer:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_buffer + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return jsonify({'status': status_color, 'classes': detected_classes})

# === Start Everything ===
if __name__ == "__main__":
    Thread(target=camera_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
