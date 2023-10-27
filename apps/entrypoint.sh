#!/bin/bash

python app.py
# gunicorn --timeout 240 --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 app:app
