python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

---- want to update it to use "uv" as a project management tool.

.\venv\Scripts\activate

uvicorn server:app --host 0.0.0.0 --port 8080