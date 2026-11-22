## AI Disclosure and External Services

### AI Assistants Used by Team Strata

Team Strata used the following AI assistants. No other AI assistants were used.

| AI Assistant | Type of Use | How It Was Used |
|---|---|---|
| **Google Gemini** | Runtime and development | **Runtime:** the dashboard's optional AI Assistant uses the Gemini API (`gemini-2.5-flash`) to explain model outputs. It receives only model outputs and weather summaries, and it does not generate predictions or determine risk tiers. **Development:** general development assistance. |
| **ChatGPT (OpenAI)** | Development only | Ideas, code suggestions and debugging. |
| **Claude (Anthropic)** | Development only | Code review, debugging, testing, documentation, and checking README claims against the code. |

All AI-generated suggestions were reviewed and tested by the team. Prediction values and risk tiers are produced only by the trained scikit-learn models in `models/`, never by a chatbot.

### External APIs and Data Sources

| Service | Purpose | Endpoint | API Key Required |
|---|---|---|---|
| Open-Meteo Forecast API | Current-week and forecast weather (up to 16 days) | `api.open-meteo.com/v1/forecast` | No |
| Open-Meteo Historical Archive API | Historical and recent observed weather | `archive-api.open-meteo.com/v1/archive` | No |
| Google Gemini API | Optional chatbot only | via the `google-genai` library | Yes (`GEMINI_API_KEY` in local `.env`) |
| Hackathon dataset (`data/dengue_history.csv`, `data/wards.csv`) | Model training and ward attributes | Provided by the EDU AI Engineering Hackathon 2026 organisers | No |

Weather data by [Open-Meteo.com](https://open-meteo.com/).

**Main libraries:** scikit-learn, pandas, NumPy, SciPy, joblib, FastAPI, Uvicorn, Requests, Streamlit, Plotly and google-genai.

## SDG Alignment (Not a Scored Criterion)

| Goal | Target | How DengueWatch Contributes |
|---|---|---|
| **SDG 3 – Good Health and Well-being** | 3.3 (neglected tropical diseases) and 3.d (early warning and risk reduction) | Gives public health teams a 2–4 week ward-level dengue risk forecast, so vector control and preparation can start before hospitals fill. |
| **SDG 11 – Sustainable Cities and Communities** | Urban resilience | Shows ward vulnerability (waterlogging, population density, sanitation) next to weather drivers, helping identify neighbourhoods that need attention first. |

**Scope:** DengueWatch provides predictive estimates and decision-support information. It does not replace official surveillance or medical assessment, guarantee outbreaks, or allocate public health resources.
