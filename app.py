from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image, ExifTags
from sklearn.cluster import KMeans
import tensorflow as tf
from statistics import mode
from collections import defaultdict
from ingredients_db import ingredients_db
from product_db import product_db
from utils import detect_face, analyze_skin_color, determine_undertone, predict_skin_concerns, generate_skincare_routine

app = Flask(__name__)
CORS(app, resources={r"/analyze": {"origins": "https://skinova-nine.vercel.app"}})
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB max upload

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_image(filepath):
    try:
        img = Image.open(filepath)
        
        # Handle EXIF orientation
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = img._getexif()
            if exif and orientation in exif:
                if exif[orientation] == 3:
                    img = img.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    img = img.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    img = img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass

        # Resize large images
        max_size = 1024
        width, height = img.size
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to OpenCV format
        cv_img = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
        return cv_img
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'frontal' not in request.files or 'left' not in request.files or 'right' not in request.files:
        return jsonify({'error': 'Missing one or more images'}), 400

    files = {
        'frontal': request.files['frontal'],
        'left': request.files['left'],
        'right': request.files['right']
    }

    images = []
    error_messages = []
    
    try:
        for view, file in files.items():
            if not file or file.filename == '':
                error_messages.append(f'No file selected for {view}')
                continue
                
            if not allowed_file(file.filename):
                error_messages.append(f'Invalid file type for {view}')
                continue

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and process image
            image = load_image(filepath)
            os.remove(filepath)  # Clean up immediately after processing
            
            if image is None:
                error_messages.append(f'Could not process image for {view}')
                continue
            
            images.append(image)

        if error_messages:
            return jsonify({'error': '; '.join(error_messages)}), 400

        # Image processing pipeline
        undertones = []
        all_concerns = []
        
        for img in images:
            face = detect_face(img)
            if face is None:
                continue
                
            try:
                skin_color = analyze_skin_color(face)
                undertone = determine_undertone(skin_color)
                undertones.append(undertone)
                
                concerns = predict_skin_concerns(face, model=None)  # Replace with actual model
                all_concerns.extend(concerns)
            except Exception as e:
                print(f"Error processing face: {str(e)}")
                continue

        final_undertone = mode(undertones) if undertones else 'neutral'
        
        # Aggregate concerns
        concern_counts = defaultdict(lambda: defaultdict(int))
        for concern in all_concerns:
            concern_counts[concern["name"]][concern["severity"]] += 1
            
        detected_concerns = [
            {"name": name, "severity": max(severities, key=severities.get)}
            for name, severities in concern_counts.items() if severities
        ]

        routine = generate_skincare_routine(detected_concerns)

        return jsonify({
            'undertone': final_undertone,
            'concerns': detected_concerns,
            'routine': routine
        })

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, host='0.0.0.0', port=5000)