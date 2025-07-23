from flask import Flask
from flask_cors import CORS  # Added for CORS support
from .config.config import Config
from .utils.database import init_db
from .utils.limiter import init_limiter
from .api.auth import auth_bp
from .api.user import user_bp
from .api.subscription import subscription_bp
from .api.assistant import assistant_bp
from .models.law import assistant as law_assistant

def create_app():
    import logging
    from logging.handlers import RotatingFileHandler
    import os

    app = Flask(__name__)
    app.config.from_object(Config)

    # Production logging setup
    if not app.debug:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(os.path.join(log_dir, 'app.log'), maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Production logging is enabled.')

    # Enable CORS for all routes (you can restrict origins if needed)
    CORS(app)
    # Example for restricting: CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize database within app context
    with app.app_context():
        init_db()
        # Initialize the model at startup
        # law_assistant.get_legal_response("hello")
        app.logger.info("Model initialized.")

    # Initialize rate limiter
    init_limiter(app)

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(user_bp, url_prefix='/api')
    app.register_blueprint(subscription_bp, url_prefix='/api')
    app.register_blueprint(assistant_bp, url_prefix='/api')
    app.logger.info("Backend and model are running.")

    return app