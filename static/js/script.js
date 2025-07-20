
function fetchLogs() {
    fetch('/logs')
        .then(res => res.text())
        .then(logs => {
            const viewer = document.getElementById('logViewer');
            viewer.textContent = logs;
            viewer.scrollTop = viewer.scrollHeight;  // auto-scroll to bottom
        })
        .catch(err => console.error('Log fetch error:', err));
}

setInterval(fetchLogs, 3000);  // fetch logs every 3 seconds

async function loadStatus() {
    try {
        const res = await fetch('/status');
        const data = await res.json();

        if (data.googleKey) {
            document.getElementById('googleKey').value = data.googleKey;
        }
        if (data.telegramKey) {
            document.getElementById('telegramKey').value = data.telegramKey;
        }
        if (data.doc) {
            document.getElementById('doc').value = data.doc;
        }

        updateBotButton(); // Check bot state on load
    } catch (err) {
        console.error('Error loading saved values', err);
    }
}

function toggleVisibility(id) {
    const input = document.getElementById(id);
    input.type = input.type === 'password' ? 'text' : 'password';
}

function copyValue(id) {
    const input = document.getElementById(id);
    navigator.clipboard.writeText(input.value);
}

async function updateBotButton() {
    try {
        const res = await fetch('/is_running');
        const data = await res.json();
        const btn = document.getElementById('botToggleButton');
        if (data.running) {
            btn.textContent = 'Stop Bot';
            btn.classList.remove('start');
            btn.classList.add('stop');
        } else {
            btn.textContent = 'Start Bot';
            btn.classList.remove('stop');
            btn.classList.add('start');
        }
    } catch (err) {
        console.error('Failed to check bot status', err);
    }
}

async function toggleBot() {
    const btn = document.getElementById('botToggleButton');
    if (btn.textContent.includes('Stop')) {
        // Stop it
        const res = await fetch('/stop', { method: 'POST' });
        const result = await res.json();
        alert(result.message || 'Bot stopped!');
    } else {
        // Start it
        const googleKey = document.getElementById('googleKey').value.trim();
        const telegramKey = document.getElementById('telegramKey').value.trim();
        const doc = document.getElementById('doc').value.trim();

        if (!googleKey || !telegramKey || !doc) {
            document.getElementById('error').textContent = 'All fields are required before starting the bot.';
            return;
        }
        document.getElementById('error').textContent = '';

        const res = await fetch('/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ googleKey, telegramKey, doc })
        });

        const result = await res.json();
        alert(result.message || 'Bot started!');
    }

    updateBotButton(); // Refresh UI state after action
}

window.onload = loadStatus;
