from flask import Flask
import os
import logging
from pathlib import Path
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_app():
    app = Flask(__name__)

    # Configuration
    app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

    # Create upload and debug directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('debug_text', exist_ok=True)

    # Register routes
    from routes import bp
    app.register_blueprint(bp)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
