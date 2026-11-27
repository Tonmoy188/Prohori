"""Grounded DengueWatch assistant using Phase 2 forecast data.

OPTIONAL module: never imported by core forecasting/allocation/routing/dispatch code.
"""

import json
import re
import threading
import time
from collections import deque

import config


SYSTEM = (
    "You are DengueWatch, a dengue early-warning assistant for "
    "public-health officers in Bangladesh. Answer only from the supplied "
    "JSON context. Never invent cases, dates, wards, weather, or observations. "
    "Clearly describe predictions as model estimates, not confirmed cases. "
    "If context is insufficient, say so. Do not provide medical treatment "
    "advice. Keep answers under 120 words."
)


class ChatConfigurationError(Exception):
    pass


class ChatRateLimited(Exception):
    pass


class ChatService:
    def __init__(self, service):
        self.svc = service
        self.client = None
        self.init_error = None

        self._cache = {}
        self._hits = {}
        self._lock = threading.Lock()

        if not config.GEMINI_API_KEY:
            self.init_error = "GEMINI_API_KEY is not configured"
            return

        try:
            from google import genai
            from google.genai import types

            self.client = genai.Client(
                api_key=config.GEMINI_API_KEY,
                http_options=types.HttpOptions(
                    api_version="v1beta",
                    timeout=int(config.CHAT_TIMEOUT_S * 1000),
                ),
            )
        except Exception as exc:
            self.init_error = (
                f"Gemini initialization failed: "
                f"{type(exc).__name__}: {exc}"
            )

    def _resolve(self, message, ward_id, horizon):
        wards = self.svc.available_wards()
        ids = {w["ward_id"].lower(): w["ward_id"] for w in wards}
        text = message.lower().replace(" ", "_")

        if ward_id is not None:
            # canonical case so predict_ward's exact lookup works
            ward_id = ids.get(
                str(ward_id).strip().lower().replace(" ", "_"), ward_id
            )
        else:
            # whole-token match, longest id first: "DNCC_Ward_1" must not
            # match inside "DNCC_Ward_12"
            for low in sorted(ids, key=len, reverse=True):
                if re.search(
                    rf"(?<![a-z0-9]){re.escape(low)}(?![a-z0-9])", text
                ):
                    ward_id = ids[low]
                    break

        city = None

        if ward_id is None:
            city = next(
                (
                    name
                    for name in sorted(
                        {w["city"] for w in wards}, key=len, reverse=True
                    )
                    if name.lower() in message.lower()
                ),
                None,
            )

        match = re.search(
            r"\b([2-4])\s*-?\s*(?:weeks?|wks?)\b",
            message.lower(),
        )

        selected_horizon = (
            horizon
            if horizon is not None
            else (int(match.group(1)) if match else 2)
        )

        return ward_id, city, selected_horizon

    def _context(self, ward_id, city, horizon):
        if ward_id:
            prediction = self.svc.predict_ward(
                ward_id,
                horizon,
            )

            return {
                "scope": "ward",
                "prediction": prediction,
                "source": "Phase 2 historical dataset and trained models",
            }

        data = self.svc.predict_all(
            horizon,
            city,
        )

        top_wards = sorted(
            data["wards"],
            key=lambda item: item["expected_cases"],
            reverse=True,
        )[:5]

        tier_counts = {}

        for item in data["wards"]:
            tier = item["risk_tier"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        return {
            "scope": city or "all wards",
            "horizon_weeks": horizon,
            "forecast_week": data["forecast_week"],
            "dataset_through": data["dataset_through"],
            "tier_counts": tier_counts,
            "highest_expected_case_wards": top_wards,
            "source": data["source"],
        }

    def ask(
        self,
        message,
        session_id="default",
        ward_id=None,
        horizon=None,
    ):
        if self.client is None:
            raise ChatConfigurationError(
                self.init_error or "Gemini API is unavailable"
            )

        message = re.sub(
            r"\s+",
            " ",
            str(message),
        ).strip()[:300]

        if not message:
            raise ValueError("Message cannot be empty")

        ward_id, city, horizon = self._resolve(
            message,
            ward_id,
            horizon,
        )

        if horizon not in (2, 3, 4):
            raise ValueError("Horizon must be 2, 3, or 4 weeks")

        self._throttle(session_id)

        context = self._context(
            ward_id,
            city,
            horizon,
        )

        cache_key = (
            message.lower(),
            json.dumps(context, sort_keys=True, default=str),
        )

        with self._lock:
            cached = self._cache.get(cache_key)

        if cached and time.time() - cached[0] < config.CHAT_CACHE_TTL_S:
            return {
                **cached[1],
                "cached": True,
            }

        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=(
                    "CONTEXT:\n"
                    + json.dumps(
                        context,
                        separators=(",", ":"),
                    )
                    + "\n\nQUESTION:\n"
                    + message
                ),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    max_output_tokens=config.CHAT_MAX_TOKENS,
                    temperature=0.2,
                    # Gemini 2.5 "thinking" tokens otherwise consume the
                    # small output budget and yield empty/truncated replies.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            # .text can raise on blocked/empty candidates, so read it here
            answer = (response.text or "").strip()

        except Exception as exc:
            raise RuntimeError(
                f"Gemini request failed: {type(exc).__name__}: {exc}"
            ) from exc

        if not answer:
            raise RuntimeError(
                "Gemini returned an empty response"
            )

        usage = getattr(response, "usage_metadata", None)

        result = {
            "reply": answer,
            "source": "gemini_api",
            "model": config.GEMINI_MODEL,
            "cached": False,
            "usage": {
                "input_tokens": getattr(
                    usage,
                    "prompt_token_count",
                    None,
                ),
                "output_tokens": getattr(
                    usage,
                    "candidates_token_count",
                    None,
                ),
            },
        }

        with self._lock:
            if len(self._cache) >= 200:
                self._cache.clear()

            self._cache[cache_key] = (
                time.time(),
                result,
            )

        return result

    def _throttle(self, session_id):
        now = time.time()

        with self._lock:
            requests = self._hits.setdefault(
                str(session_id),
                deque(),
            )

            while requests and now - requests[0] > 60:
                requests.popleft()

            if len(requests) >= config.CHAT_RATE_PER_MIN:
                raise ChatRateLimited(
                    f"Maximum {config.CHAT_RATE_PER_MIN} "
                    "chat requests per minute"
                )

            requests.append(now)

            if len(self._hits) > 1000:  # drop idle sessions (memory leak)
                for key in [
                    k for k, q in self._hits.items()
                    if not q or now - q[-1] > 60
                ]:
                    del self._hits[key]
