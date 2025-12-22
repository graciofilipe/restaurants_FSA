import os
from flask import Flask
from .routes import admin_bp
from .public import public_bp
from .auth_routes import auth_bp

def create_app():
    app = Flask(__name__, template_folder='../templates')
    
    # Set secret key for sessions. In production, this should be a secure environment variable.
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key_change_me")
    
    app.register_blueprint(public_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(auth_bp)
    
    @app.context_processor
    def inject_firebase_config():
        return dict(
            firebase_api_key=os.environ.get("FIREBASE_API_KEY", ""),
            firebase_auth_domain=os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
            firebase_project_id=os.environ.get("FIREBASE_PROJECT_ID", ""),
            firebase_storage_bucket=os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
            firebase_messaging_sender_id=os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
            firebase_app_id=os.environ.get("FIREBASE_APP_ID", "")
        )
    
    return app
