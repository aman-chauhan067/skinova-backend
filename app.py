from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image, ExifTags
from collections import defaultdict
from statistics import mode
import tempfile
import shutil
from gevent.pywsgi import WSGIServer

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/analyze": {
        "origins": [
            "https://skinova-km345f3pq-aman-chauhans-projects-d51024c8.vercel.app/analysis",
            "http://localhost:3000"
        ],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type"]
    }
})

# Configuration constants
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_image(file):
    temp_dir = tempfile.mkdtemp()
    try:
        # Validate and secure filename
        if not file or not allowed_file(file.filename):
            raise ValueError("Invalid file type")

        # Create temp file path
        filename = secure_filename(file.filename)
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)

        # Open and process image
        img = Image.open(filepath)
        
        # Handle EXIF orientation
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = img._getexif()
            if exif and orientation in exif:
                rotation = {
                    3: 180,
                    6: 270,
                    8: 90
                }.get(exif[orientation], 0)
                if rotation:
                    img = img.rotate(rotation, expand=True)
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
        app.logger.error(f"Image processing error: {str(e)}")
        raise
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Validate request
        if not all(key in request.files for key in ['frontal', 'left', 'right']):
            return jsonify({'error': 'Missing required images'}), 400

        # Process images
        results = []
        for view in ['frontal', 'left', 'right']:
            file = request.files[view]
            if not file or file.filename == '':
                return jsonify({'error': f'No file selected for {view}'}), 400

            try:
                cv_img = load_image(file)
                # Add your image processing logic here
                # results.append(process_image(cv_img))
            except Exception as e:
                return jsonify({'error': f'Error processing {view}: {str(e)}'}), 400

        # Generate response
        return jsonify({
            'status': 'success',
            'results': {
                'undertone': 'warm',
                'concerns': [],
                'routine': {}
            }
        })

    except Exception as e:
        app.logger.error(f"Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'version': '1.0.0'})

if __name__ == '__main__':
    if os.environ.get('FLASK_ENV') == 'production':
        # Production server
        http_server = WSGIServer('0.0.0.0', int(os.environ.get('PORT', 5000)), app)
        http_server.serve_forever()
    else:
        # Development server
        app.run(debug=True, host='0.0.0.0', port=5000)