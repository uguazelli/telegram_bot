#!/bin/bash

# Start both processes in background
uvicorn main:app --host 0.0.0.0 --port 8000 &
python telegram_handler.py &

# Wait for all to finish (keep container running)
wait
