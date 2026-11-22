# PROHORI | DengueWatch - Streamlit frontend

Dashboard for the DengueWatch FastAPI backend. Pages: Home, Overview, Ward Explorer, Weather Outlook,
Prediction Details, AI Assistant, Model Performance, About & Resources.

All dengue numbers are **model estimates**, not confirmed cases. Weather is city level (one Open-Meteo location per city).

## Requirements
- Python 3.10 or newer (the backend's existing `venv` was created with Python 3.14, which also works for the backend).
- Packages: `frontend/requirements.txt` (streamlit >= 1.52, plotly, pandas, requests).
- Internet access for the backend (Open-Meteo). The frontend only talks to the backend.

## Folder layout
Keep `frontend/` **inside the backend folder**, next to `main.py` and `data/`:

```
<backend folder>/
  main.py  service.py  engine.py  weather.py  chat.py  config.py
  data/wards.csv  data/dengue_history.csv
  frontend/            <- this folder (app.py, api_client.py, styles.css, requirements.txt)
```
The frontend reads `data/wards.csv` (ward vulnerability) and `data/dengue_history.csv` (seasonal chart) directly,
because the API does not serve them. If they are elsewhere, see *Environment variables*.

## 1. Start the backend (terminal 1)
```
cd <backend folder>
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt  # scikit-learn must be exactly 1.8.0 (pinned) to load the saved models
copy .env.example .env           # macOS/Linux: cp .env.example .env
python -m uvicorn main:app --port 8000
```
The chatbot is optional: put your own key in the backend `.env` (never commit it). Without it, predictions and
dashboards still work and the AI Assistant page shows "not enabled".
Check: http://127.0.0.1:8000/health should return `"status": "ok"`.

## 2. Start the frontend (terminal 2)
```
cd <backend folder>\frontend
pip install -r requirements.txt  # use the same venv, or a new one
streamlit run app.py
```
Open http://localhost:8501.

## Environment variables (all optional)
| Variable | Default | Purpose |
|---|---|---|
| `DENGUEWATCH_API_URL` | `http://127.0.0.1:8000` | Backend address |
| `DENGUEWATCH_API_TIMEOUT` | `30` | Seconds to wait for normal API calls |
| `DENGUEWATCH_CHAT_TIMEOUT` | `45` | Seconds to wait for a chat reply |
| `DENGUEWATCH_WARDS_CSV` | `../data/wards.csv` | Full path of `wards.csv` |
| `DENGUEWATCH_HISTORY_CSV` | beside `wards.csv` | Full path of `dengue_history.csv` |

Windows PowerShell: `$env:DENGUEWATCH_API_URL="http://127.0.0.1:8000"` before `streamlit run app.py`.
macOS/Linux: `export DENGUEWATCH_API_URL=http://127.0.0.1:8000`.

## Troubleshooting
- **"Backend offline"**: start the backend (step 1) and confirm the URL/port, or set `DENGUEWATCH_API_URL`.
- **"HTTP 500" on weather or prediction pages**: look at the backend console. A known backend config gap
  (`WEATHER_RETRY_S` missing in `config.py`) turns a failed Open-Meteo request into HTTP 500 once the 3-hour weather
  cache has expired. Restore internet access and retry after the backend fix.
- **"HTTP 503 Live weather unavailable"**: Open-Meteo could not be reached and the backend has no cached copy.
- **Vulnerability values or the seasonal chart say "unavailable"**: the CSV was not found. The page tells you where it
  looked. Move `frontend/` inside the backend folder or set the CSV variables above; the app picks the file up on the
  next page update (or press *Check again*).
- **AI Assistant disabled**: the backend has no chatbot key configured (`/stats` reports `configured: false`).
  **HTTP 429**: more than 10 messages per minute. **HTTP 500 from chat**: the AI provider failed; try again.
- **Icon names appear as text** (for example `smart_toy`): hard-refresh the browser (Ctrl+Shift+R) and report it.
- **Port already in use**: `streamlit run app.py --server.port 8502`.
