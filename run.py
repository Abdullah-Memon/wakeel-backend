from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(
        host=app.config['BACKEND_URL'],
        port=app.config['BACKEND_PORT'],
        debug=app.config['DEBUG'],
        threaded=True
    )


# ---

# <xaiArtifact artifact_id="71318407-694b-46ce-ae42-f35404b996d2" artifact_version_id="86e45533-3687-4f84-b174-9073af771515" title="README.md" contentType="text/markdown">
# # Flask Backend

# ## Overview
# A professional Flask-based backend application built with Flask, following a clean architecture for scalability and maintainability. The project includes authentication, user management, subscription handling, and an assistant service, integrated with a MySQL database.

# ## Prerequisites
# - Python 3.8+
# - MySQL Server
# - pip

# ## Installation

# 1. Clone the repository:
#    ```bash
#    git clone <repository-url>
#    cd project
#    ```

# 2. Create a virtual environment:
#    ```bash
#    python -m venv venv
#    source venv/bin/activate  # On Windows: venv\Scripts\activate
#    ```

# 3. Install dependencies:
#    ```bash
#    pip install -r requirements.txt
#    ```

# 4. Set up environment variables:
#    - Copy the `.env.example` to `.env`
#    - Update the `.env` file with your database credentials and secret key

# 5. Run the application:
#    ```bash
#    python run.py
#    ```

# ## Project Structure
# ```
# project/
# ├── app/
# │   ├── api/              # API route definitions
# │   ├── models/           # Database schemas
# │   ├── services/         # Business logic
# │   ├── utils/            # Utility functions
# │   ├── config/           # Configuration settings
# ├── requirements.txt      # Project dependencies
# ├── .env                 # Environment variables
# ├── run.py               # Application entry point
# ├── README.md            # Project documentation
# ```

# ## API Endpoints
# - **POST /api/login**: User login
# - **POST /api/register**: User registration
# - **POST /api/verify-auth**: Token verification
# - **GET /api/user-details**: Get user details
# - **GET /api/subscription-details**: Get subscription details
# - **POST /api/set-subscription**: Set user subscription
# - **GET /api/assistant**: Assistant response

# ## Security Features
# - JWT-based authentication
# - Rate limiting (100 requests/minute)
# - Password hashing with bcrypt
# - SQL injection prevention
# - Secure database connection handling

# ## Database
# The application uses MySQL with the following tables:
# - users
# - chat_sessions
# - messages
# - subscriptions
# - transactions

# The schema is automatically initialized on application start.

# ## Contributing
# 1. Fork the repository
# 2. Create a feature branch
# 3. Submit a pull request

# ## License
# MIT License