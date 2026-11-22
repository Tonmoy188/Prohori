
"""Grounded chatbot using the configured live Gemini API only; no template/demo fallback."""
import json
import re
import threading
import time
from collections import deque

import config


SYSTEM = (
    "You are the dengue early-warning assistant for public-health officers in Bangladesh. "
    "Answer only from the supplied JSON context. Never invent cases, dates, wards, weather, "
    "or API observations. Clearly call model outputs estimates, not confirmed case counts. "
    "If context is insufficient, say so. Do not give medical treatment advice. Keep answers under 120 words."
)


class ChatConfigurationError(Exception):
    pass


class ChatRateLimited(Exception):
    pass


def _compact_weather(ctx):
    """Trim weather context to what the LLM needs (~70% fewer input tokens)."""
    cur = ctx["current_week_weather"]
    keep = ("date", "rainfall_mm", "avg_temp_c", "relative_humidity_pct")
    return {
        "stale": ctx["stale_cache"],
        "current_week": {
            k: (round(cur[k], 1) if isinstance(cur[k], float) else cur[k])
            for k in keep
        },
        "forecast_7d": [
            {k: d[k] for k in keep}
            for d in ctx["daily_forecast"][:7]
        ],
    }


class ChatService:
    def __init__(self, service):
        self.svc = service
        self.client = None
        self.init_error = None
        self._cache, self._hits, self._lock = {}, {}, threading.Lock()

        if not config.GEMINI_API_KEY:
            self.init_error = "GEMINI_API_KEY is not configured"
        else:
            try:
                from google import genai
                from google.genai import types

                self.client = genai.Client(
                    api_key=config.GEMINI_API_KEY,
                    http_options=types.HttpOptions(
                        api_version="v1beta",
                        timeout=int(config.CHAT_TIMEOUT_S * 1000)
                    ),
                )
            except Exception as exc:
                self.init_error = (
                    f"Gemini client initialization failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    def _resolve(self, message, ward_id, horizon):
        ids = self.svc.ward_ids()

        if ward_id is None:
            normalized = message.lower().replace(" ", "_")
            ward_id = next(
                (item for item in ids if item.lower() in normalized),
                None
            )

        city = None

        if ward_id is None:
            city = next(
                (
                    name for name in self.svc.wards.city.unique()
                    if name.lower() in message.lower()
                ),
                None
            )

        match = re.search(
            r"\b([2-4])\s*-?\s*(?:weeks?|wks?)\b",
            message.lower()
        )

        return ward_id, city, horizon or (
            int(match.group(1)) if match else 3
        )

    def _context(self, ward_id, city, horizon):
        if ward_id:
            p = self.svc.predict_ward(ward_id, horizon)

            return {
                "scope": "ward",
                "prediction": {
                    k: p[k] for k in (
                        "ward_id",
                        "city",
                        "risk_tier",
                        "expected_cases",
                        "probabilities",
                        "horizon_weeks",
                        "origin_week",
                        "forecast_week",
                    )
                },
                "drivers": p["drivers"],
                "recent_weather": p["recent_weather"][-4:],
                "weather": _compact_weather(p["weather_context"]),
                "forecast_coverage": {
                    k: p["forecast_coverage"][k]
                    for k in (
                        "weather_forecast_through",
                        "target_week",
                        "target_week_covered_by_weather_forecast",
                    )
                },
            }

        data = self.svc.predict_all(horizon, city)

        top = sorted(
            data["wards"],
            key=lambda item: -item["expected_cases"]
        )[:5]

        counts = {}

        for item in data["wards"]:
            counts[item["risk_tier"]] = (
                counts.get(item["risk_tier"], 0) + 1
            )

        return {
            "scope": city or "all wards",
            "horizon_weeks": horizon,
            "forecast_week": data["forecast_week"],
            "tier_counts": counts,
            "highest_risk_wards": [
                {
                    k: item[k] for k in (
                        "ward_id",
                        "city",
                        "risk_tier",
                        "expected_cases",
                    )
                }
                for item in top
            ],
            "weather": {
                n: _compact_weather(c)
                for n, c in data.get("weather_contexts", {}).items()
            },
        }

    def ask(self, message, session_id="default", ward_id=None, horizon=None):
        if self.client is None:
            raise ChatConfigurationError(
                self.init_error or "Gemini API is not configured"
            )

        message = re.sub(r"\s+", " ", message).strip()[:300]

        ward_id, city, horizon = self._resolve(
            message, ward_id, horizon
        )

        self._throttle(session_id)

        context = self._context(ward_id, city, horizon)

        key = (
            message.lower(),
            json.dumps(context, sort_keys=True, default=str)
        )

        # Check cache
        with self._lock:
            hit = self._cache.get(key)

        if hit and time.time() - hit[0] < config.CHAT_CACHE_TTL_S:
            return {**hit[1], "cached": True}

        # Gemini API request
        from google.genai import types

        response = self.client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=(
                f"CONTEXT:\n"
                f"{json.dumps(context, separators=(',', ':'))}"
                f"\n\nQUESTION: {message}"
            ),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                max_output_tokens=config.CHAT_MAX_TOKENS,
                temperature=0.2,
            ),
        )

        answer = (response.text or "").strip()

        if not answer:
            raise RuntimeError("Gemini API returned an empty response")

        usage = getattr(response, "usage_metadata", None)

        result = {
            "reply": answer,
            "source": "gemini_api",
            "model": config.GEMINI_MODEL,
            "cached": False,
            "usage": {
                "input_tokens": getattr(
                    usage, "prompt_token_count", None
                ),
                "output_tokens": getattr(
                    usage, "candidates_token_count", None
                ),
            },
        }

        # Save cache
        with self._lock:
            if len(self._cache) > 200:
                self._cache.clear()

            self._cache[key] = (time.time(), result)

        return result

    def _throttle(self, session_id):
        now = time.time()

        with self._lock:
            q = self._hits.setdefault(session_id, deque())

            while q and now - q[0] > 60:
                q.popleft()

            if len(q) >= config.CHAT_RATE_PER_MIN:
                raise ChatRateLimited(
                    f"Limit {config.CHAT_RATE_PER_MIN} "
                    "chat requests/minute per session"
                )

            q.append(now)
