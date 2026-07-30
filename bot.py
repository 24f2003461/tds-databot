import os
import re
import json
import time
import threading
import traceback
import contextlib
import io
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import uvicorn
from openai import OpenAI

# ---------- Config (from environment variables set on Render) ----------

BOT_TOKEN = os.environ["BOT_TOKEN"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "openrouter/free")
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://openrouter.ai/api/v1")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_API_BASE,
    default_headers={
        "HTTP-Referer": BASE_URL,
        "X-Title": "tds-databot",
    },
)

LOG_FILE = "run.jsonl"
LOG_LOCK = threading.Lock()


def log_event(event: dict):
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with LOG_LOCK:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")


# ---------- Web app: health check + public log ----------

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/run.jsonl")
def get_log():
    if not os.path.exists(LOG_FILE):
        return PlainTextResponse("")
    with open(LOG_FILE, "r") as f:
        return PlainTextResponse(f.read())


# ---------- Tool: run_python ----------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code and return whatever it prints to stdout. "
                "Use this to download datasets (requests, pandas, BeautifulSoup), "
                "compute statistics, etc. Always print() the values you need to see."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"],
            },
        },
    }
]


def run_python(code: str) -> str:
    stdout = io.StringIO()
    globs = {}
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, globs)
        out = stdout.getvalue()
    except Exception as e:
        out = stdout.getvalue() + f"\nERROR: {e}\n{traceback.format_exc()}"
    if len(out) > 8000:
        out = out[-8000:]
    return out


# ---------- Per-chat history (multi-turn support) ----------

chat_histories = {}
HISTORY_LIMIT = 20


def get_history(chat_id):
    return chat_histories.setdefault(chat_id, [])


def append_history(chat_id, role, content):
    h = get_history(chat_id)
    h.append({"role": role, "content": content})
    if len(h) > HISTORY_LIMIT:
        del h[0 : len(h) - HISTORY_LIMIT]


# ---------- JSON extraction (defensive layer) ----------

def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


# ---------- Agent loop ----------

SYSTEM_PROMPT = """You are a data-analysis agent replying to a Telegram message.

Rules:
- Answer only the LATEST user message; earlier messages are context for multi-turn conversations.
- If the latest message is only a setup message (e.g. "I will send data next") with no real question,
  still reply with a short JSON ack, e.g. {"answer": "ok", "log_url": "PLACEHOLDER"} - never skip replying.
- Use the run_python tool to fetch or compute anything you are not 100% certain of. Never guess a number
  you could compute. For well-known published statistics, if fetching the source fails, answer from your
  own knowledge as a fallback rather than leaving it blank.
- Your FINAL reply must be ONLY the JSON object the question asks for - no prose, no markdown fences,
  nothing else before or after it.
- Match the requested shape exactly: same keys, same nesting, same value types (string vs number) as the
  question specifies. Never add extra keys beyond what's asked, except log_url which is always required.
- Always include "log_url": "PLACEHOLDER" in your final JSON - the code will replace it with the real URL.
"""

MAX_STEPS = 10
TIME_BUDGET_SECONDS = 210  # stay well under the ~300s grader timeout


def agent_reply(chat_id: int, user_text: str) -> str:
    deadline = time.time() + TIME_BUDGET_SECONDS
    append_history(chat_id, "user", user_text)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + get_history(chat_id)
    final_text = None

    for step in range(MAX_STEPS):
        time_left = deadline - time.time()
        use_tools = time_left > 15  # stop granting tool access once budget is nearly gone

        kwargs = dict(model=MODEL, messages=messages)
        if use_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            log_event({"chat_id": chat_id, "step": step, "error": str(e)})
            break

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                }
            )
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                code = args.get("code", "")
                result = run_python(code)
                log_event(
                    {
                        "chat_id": chat_id,
                        "step": step,
                        "tool_call": "run_python",
                        "code": code,
                        "result": result,
                    }
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
            continue
        else:
            final_text = msg.content or ""
            log_event({"chat_id": chat_id, "step": step, "final_text": final_text})
            break

    if final_text is None:
        # Time or steps ran out - force a final answer with no tools.
        try:
            messages.append(
                {
                    "role": "user",
                    "content": "Time is up. Reply now with ONLY the final JSON object, no tools.",
                }
            )
            resp = client.chat.completions.create(model=MODEL, messages=messages)
            final_text = resp.choices[0].message.content or ""
        except Exception as e:
            final_text = ""
            log_event({"chat_id": chat_id, "error": f"forced-answer failed: {e}"})

    parsed = extract_json(final_text)
    if parsed is None:
        parsed = {"answer": final_text.strip()}
    if "answer" not in parsed:
        parsed = {"answer": parsed}

    parsed["log_url"] = f"{BASE_URL}/run.jsonl"

    reply_str = json.dumps(parsed)
    append_history(chat_id, "assistant", reply_str)
    log_event({"chat_id": chat_id, "reply": reply_str})
    return reply_str


# ---------- Telegram long-polling ----------

def send_message(chat_id, text):
    try:
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=30,
        )
    except Exception as e:
        log_event({"error": f"send_message failed: {e}"})


def poll_loop():
    offset = None
    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset
            resp = requests.get(f"{TELEGRAM_API}/getUpdates", params=params, timeout=40)
            data = resp.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message")
                if not message or "text" not in message:
                    continue
                chat_id = message["chat"]["id"]
                text = message["text"]
                log_event({"chat_id": chat_id, "received": text})
                try:
                    reply = agent_reply(chat_id, text)
                except Exception as e:
                    reply = json.dumps(
                        {"answer": "internal error", "log_url": f"{BASE_URL}/run.jsonl"}
                    )
                    log_event(
                        {
                            "chat_id": chat_id,
                            "error": f"agent_reply crashed: {e}\n{traceback.format_exc()}",
                        }
                    )
                send_message(chat_id, reply)
        except Exception as e:
            log_event({"error": f"poll_loop error: {e}"})
            time.sleep(5)


def keep_warm_loop():
    while True:
        time.sleep(600)  # every 10 minutes
        try:
            requests.get(f"{BASE_URL}/health", timeout=20)
        except Exception:
            pass


@app.on_event("startup")
def startup():
    threading.Thread(target=poll_loop, daemon=True).start()
    threading.Thread(target=keep_warm_loop, daemon=True).start()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
