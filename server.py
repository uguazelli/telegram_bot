from flask import Flask, request, jsonify, render_template
import subprocess
import os
from io import StringIO
import threading
from dotenv import dotenv_values

app = Flask(__name__)
BOT_PROCESS = None
LOG_BUFFER = StringIO()
LOG_LOCK = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/is_running', methods=['GET'])
def is_running():
    global BOT_PROCESS
    is_active = BOT_PROCESS is not None and BOT_PROCESS.poll() is None
    return jsonify({'running': is_active})


@app.route('/status', methods=['GET'])
def status():
    config = dotenv_values(".env")
    google_key = config.get("GOOGLE_API_KEY", "")
    telegram_key = config.get("TELEGRAM_API_KEY", "")

    doc_content = ""
    try:
        with open("uploads/company_doc.txt", "r") as f:
            doc_content = f.read()
    except FileNotFoundError:
        pass

    return jsonify({
        "googleKey": google_key,
        "telegramKey": telegram_key,
        "doc": doc_content
    })

@app.route('/start', methods=['POST'])
def start_bot():
    global BOT_PROCESS
    data = request.json

    if not data.get('googleKey') or not data.get('telegramKey') or not data.get('doc'):
        return jsonify({'error': 'Missing fields'}), 400

    # Save .env values
    try:
        with open('.env', 'w') as f:
            f.write(f"GOOGLE_API_KEY={data['googleKey']}\n")
            f.write(f"TELEGRAM_API_KEY={data['telegramKey']}\n")
        print("✅ Saved API keys to .env")
    except Exception as e:
        print(f"❌ Error writing .env: {e}")
        return jsonify({'error': 'Failed to write .env'}), 500

    # Save company_doc.txt
    try:
        with open('uploads/company_doc.txt', 'w') as f:
            f.write(data['doc'])
        print("✅ Saved document to uploads/company_doc.txt")
    except Exception as e:
        print(f"❌ Error writing company_doc.txt: {e}")
        return jsonify({'error': 'Failed to write doc'}), 500

    # Start the bot
    try:
        if BOT_PROCESS is None:
            print("🚀 Starting telegram_handler.py...")
            BOT_PROCESS = subprocess.Popen(
                ['python3', 'telegram_handler.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True  # ✅ makes decoding automatic
            )

            # Optional: print bot logs in the terminal
            import threading
            def log_stream(pipe, name):
                for line in iter(pipe.readline, ''):
                    if line:
                        log_line = f"[{name}] {line.strip()}\n"
                        with LOG_LOCK:
                            LOG_BUFFER.write(log_line)
                        print(log_line, end='')  # still print to terminal

            threading.Thread(target=log_stream, args=(BOT_PROCESS.stdout, "stdout"), daemon=True).start()
            threading.Thread(target=log_stream, args=(BOT_PROCESS.stderr, "stderr"), daemon=True).start()

            return jsonify({'message': 'Bot started!'})
        else:
            print("⚠️ Bot already running.")
            return jsonify({'message': 'Bot is already running'})
    except Exception as e:
        print(f"❌ Failed to start bot: {e}")
        return jsonify({'error': 'Failed to start bot'}), 500


@app.route('/logs', methods=['GET'])
def get_logs():
    with LOG_LOCK:
        logs = LOG_BUFFER.getvalue()
    return logs[-10000:]  # Return only last 10,000 chars for safety


@app.route('/stop', methods=['POST'])
def stop_bot():
    global BOT_PROCESS
    if BOT_PROCESS:
        BOT_PROCESS.terminate()
        BOT_PROCESS = None
        return jsonify({'message': 'Bot stopped!'})
    else:
        return jsonify({'message': 'No bot running'})

if __name__ == '__main__':
    app.run(debug=True)
