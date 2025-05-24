from backend.app import app

# This is a wrapper file that imports the app from the backend directory
# This allows Gunicorn to find the app when run from the root directory

if __name__ == "__main__":
    app.run(debug=True)
