function readLocalModelBody() {
    return {
        source: document.getElementById('s-local-source').value.trim(),
        filename: document.getElementById('s-local-filename').value.trim(),
        port: parseInt(document.getElementById('s-local-port').value, 10) || 8766,
        n_gpu_layers: parseInt(document.getElementById('s-local-gpu-layers').value, 10),
        n_ctx: parseInt(document.getElementById('s-local-ctx').value, 10) || 16384,
        chat_format: document.getElementById('s-local-chat-format').value.trim(),
    };
}

function setTestResult(text, tone = 'muted') {
    const el = document.getElementById('local-model-test-result');
    if (!el) return;
    el.style.display = text ? 'block' : 'none';
    el.textContent = text;
    el.dataset.tone = tone;
}

export function bindLocalModelControls({ state }) {
    async function updateLocalStatus() {
        if (state.activePage !== 'settings') return;
        try {
            const resp = await fetch('/api/local-model/status', { cache: 'no-store' });
            const d = await resp.json();
            const el = document.getElementById('local-model-status');
            if (!el) return;
            const isReady = d.status === 'ready';
            let text = 'Status: ' + (d.status || 'offline').charAt(0).toUpperCase() + (d.status || 'offline').slice(1);
            if (d.status === 'ready' && d.context_length) text += ` (ctx: ${d.context_length})`;
            if (d.status === 'downloading' && d.download_progress) text += ` ${Math.round(d.download_progress * 100)}%`;
            if (d.error) text += ' - ' + d.error;
            el.textContent = text;
            el.dataset.tone = isReady ? 'ok' : (d.status === 'error' ? 'error' : 'muted');
            document.getElementById('btn-local-stop').disabled = !isReady;
            document.getElementById('btn-local-test').disabled = !isReady;
            ['s-local-main', 's-local-code', 's-local-light', 's-local-fallback'].forEach((id) => {
                const cb = document.getElementById(id);
                const label = cb?.closest('.local-toggle');
                if (!cb || !label) return;
                if (cb.checked && !isReady) {
                    label.title = 'Local server is not running - requests will fail until started';
                    label.dataset.warning = '1';
                } else {
                    label.title = '';
                    delete label.dataset.warning;
                }
            });
        } catch {}
    }

    document.getElementById('btn-local-start').addEventListener('click', async () => {
        const body = readLocalModelBody();
        if (!body.source) {
            alert('Enter a model source (HuggingFace repo ID or local path)');
            return;
        }
        try {
            const resp = await fetch('/api/local-model/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await resp.json();
            if (data.error) alert('Error: ' + data.error);
            else updateLocalStatus();
        } catch (e) {
            alert('Failed: ' + e.message);
        }
    });

    document.getElementById('btn-local-stop').addEventListener('click', async () => {
        try {
            await fetch('/api/local-model/stop', { method: 'POST' });
            updateLocalStatus();
        } catch (e) {
            alert('Failed: ' + e.message);
        }
    });

    document.getElementById('btn-local-test').addEventListener('click', async () => {
        setTestResult('Running tests...', 'muted');
        try {
            const resp = await fetch('/api/local-model/test', { method: 'POST' });
            const r = await resp.json();
            if (r.error) {
                setTestResult('Error: ' + r.error, 'error');
                return;
            }
            const lines = [];
            lines.push((r.chat_ok ? '✓' : '✗') + ' Basic chat' + (r.tokens_per_sec ? ` (${r.tokens_per_sec} tok/s)` : ''));
            lines.push((r.tool_call_ok ? '✓' : '✗') + ' Tool calling');
            if (r.details && !r.success) lines.push(r.details);
            setTestResult(lines.join('\n'), r.success ? 'ok' : 'warn');
            const el = document.getElementById('local-model-test-result');
            if (el) el.style.whiteSpace = 'pre-wrap';
        } catch (e) {
            setTestResult('Test failed: ' + e.message, 'error');
        }
    });

    updateLocalStatus();
    setInterval(updateLocalStatus, 3000);
}
