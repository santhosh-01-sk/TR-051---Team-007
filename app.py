from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import torch
import numpy as np
from PIL import Image
import io
import base64
from pipeline import DisasterAssessmentPipeline

app = Flask(__name__, static_folder='.', template_folder='.')

# Initialize pipeline once to avoid loading overhead
print("Initializing Siamese U-Net Pipeline...")
pipeline = DisasterAssessmentPipeline(num_disaster_types=10)

def array_to_base64_img(arr, cmap=None):
    """Converts a numpy mask array into a base64 string for HTML rendering"""
    # Normalize or apply colormap logic
    if cmap == 'damage':
        # Create a colored mask for damage classes (0: none, 1: minor, 2: major, 3: destroyed)
        colored = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        colored[arr == 1] = [255, 255, 0]   # Yellow (Minor)
        colored[arr == 2] = [255, 165, 0]   # Orange (Major)
        colored[arr == 3] = [255, 0, 0]     # Red (Destroyed)
        img = Image.fromarray(colored)
    else:
        # Binary segmentation (buildings)
        img = Image.fromarray((arr * 255).astype(np.uint8))
        
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def home():
    return send_from_directory('.', 'disaster.html')

@app.route('/ingest', methods=['POST'])
def ingest():
    if 'pre_image' not in request.files or 'post_image' not in request.files:
        return jsonify({"error": "Missing image files"}), 400

    pre_file = request.files['pre_image']
    post_file = request.files['post_image']
    disaster_type = int(request.form.get('disaster_type', 0))

    try:
        # Load images
        pre_img = Image.open(pre_file).convert('RGB')
        post_img = Image.open(post_file).convert('RGB')

        # Run model
        seg_mask, damage_mask = pipeline.predict(pre_img, post_img, disaster_type)

        # Convert to Base64 for the frontend
        seg_b64 = array_to_base64_img(seg_mask, cmap='seg')
        damage_b64 = array_to_base64_img(damage_mask, cmap='damage')

        return jsonify({
            "status": "success",
            "seg_base64": seg_b64,
            "damage_base64": damage_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Serving DisasterAI application on http://127.0.0.1:8000")
    app.run(host='127.0.0.1', port=8000, debug=False)
