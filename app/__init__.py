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
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Enable CORS for all routes (you can restrict origins if needed)
    CORS(app)
    # Example for restricting: CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Initialize database within app context
    with app.app_context():
        init_db()
        # Initialize the model at startup
        law_assistant.get_legal_response("hello")
        print("Model initialized.")
    
    # Initialize rate limiter
    init_limiter(app)
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(user_bp, url_prefix='/api')
    app.register_blueprint(subscription_bp, url_prefix='/api')
    app.register_blueprint(assistant_bp, url_prefix='/api')
    print("Backend and model are running.")
    
    return app