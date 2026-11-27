# Prohori — Streamlit frontend (Phase 2)

Minimalist, mobile-friendly dashboard for the Phase 2 FastAPI backend (`main.py`). It only calls the backend API; no models or data are loaded here.

## Run
```bash
# 1) backend (project root)
python -m uvicorn main:app --port 8000
# 2) frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```
Set `DENGUEWATCH_API_URL` if the API is not at `http://127.0.0.1:8000`.

## Pages → backend endpoints
| Page | Endpoint(s) |
|---|---|
| Home | `GET /health`, `GET /predict/all` |
| Forecast | `GET /predict/all`, `GET /cities` |
| Ward | `GET /wards`, `GET /outlook` |
| Allocate | `POST /operations/allocate` |
| Overflow | `POST /operations/overflow` (optional bed-capacity overrides) |
| Simulate | `POST /operations/simulate` (auto ROI or manual crews) |
| Dispatch | `POST /operations/dispatch`, `/dispatch/html`, `/dispatch/export` |
| Weather | `GET /weather`, `POST /weather/offline/{on|off}` |
| Assistant | `POST /chat` |

## Files
- `app.py` — pages and layout (Streamlit `st.navigation`, top bar; collapses to a menu on phones)
- `api_client.py` — thin `requests` client
- `styles.css` — theme, cards, scroll-reveal animations, mobile breakpoints
- `assets/*.svg` — local illustrations and the Prohori logo (`logo.svg`, `wordmark.svg`)
- `.streamlit/config.toml` — colour theme
