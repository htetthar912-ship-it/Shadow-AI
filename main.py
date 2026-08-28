import os
import re
import uuid
import time
import random
import base64
import asyncio
import datetime
from typing import Dict, List, Optional, Literal

import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

from database import init_db, get_db, User, UsageRecord, CreditTransaction
from database import register_user, set_user_as_admin


load_dotenv()
init_db()

app = FastAPI(title="Shadow AI")
templates = Jinja2Templates(directory="templates")
templates.env.cache = None
app.mount("/static", StaticFiles(directory="static"), name="static")


def env_or_default(key: str, default: str) -> str:
    val = os.environ.get(key, "").strip()
    return val if val else default


# ---------------------------------------------------------------------------
# OpenRouter — single provider, per business requirements (no VPN needed
# for Myanmar). Every model ID below MUST be a real slug from
# https://openrouter.ai/models — verify and edit these before going live.
# ---------------------------------------------------------------------------
if not os.environ.get("OPENROUTER_API_KEY"):
    raise RuntimeError("OPENROUTER_API_KEY is required in your .env file.")

from openai import OpenAI as _OpenAIClient
openrouter_client = _OpenAIClient(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
OR_HEADERS = {"HTTP-Referer": "https://shadow-ai.local", "X-Title": "Shadow AI"}

# ---------------------------------------------------------------------------
# MODEL CATALOG — verify every *_MODEL_ID against openrouter.ai/models.
# ---------------------------------------------------------------------------
TEXT_MODELS = {
    "deepseek-v4-flash": {
        "label": "DeepSeek V4 Flash", "tier": "free",
        "model_id": env_or_default("DEEPSEEK_MODEL_ID", "deepseek/deepseek-chat"),
    },
    "gpt-5.6-sol": {
        "label": "GPT-5.6 Sol", "tier": "premium",
        "model_id": env_or_default("GPT_SOL_MODEL_ID", "openai/gpt-4.1"),
    },
    "claude-sonnet-5": {
        "label": "Claude Sonnet 5", "tier": "premium",
        "model_id": env_or_default("CLAUDE_MODEL_ID", "anthropic/claude-sonnet-4.6"),
    },
    "gemini-3.7-flash": {
        "label": "Gemini 3.7 Flash", "tier": "premium",
        "model_id": env_or_default("GEMINI_MODEL_ID", "google/gemini-2.5-flash"),
    },
}

IMAGE_MODELS = {
    "nano-banana-2": {"label": "Nano Banana 2 (Free)", "credit_cost": 0, "free_quota": True,
                       "model_id": env_or_default("NANO_BANANA_MODEL_ID", "google/gemini-2.5-flash-image")},
    "midjourney": {"label": "Midjourney", "credit_cost": 15, "free_quota": False,
                   "model_id": env_or_default("MIDJOURNEY_MODEL_ID", "")},
    "flux-ultra": {"label": "FLUX.1 Ultra", "credit_cost": 15, "free_quota": False,
                   "model_id": env_or_default("FLUX_ULTRA_MODEL_ID", "black-forest-labs/flux-1.1-pro")},
    "dalle-4": {"label": "DALL-E 4", "credit_cost": 10, "free_quota": False,
                "model_id": env_or_default("DALLE4_MODEL_ID", "openai/gpt-image-1")},
}

VIDEO_MODELS = {
    "runway-gen45": {"label": "Runway Gen-4.5", "credit_cost": 70, "model_id": env_or_default("RUNWAY_MODEL_ID", "")},
    "wan-3": {"label": "Wan 3.0", "credit_cost": 35, "model_id": env_or_default("WAN3_MODEL_ID", "")},
    "seedance-mini": {"label": "Seedance Mini", "credit_cost": 35, "model_id": env_or_default("SEEDANCE_MODEL_ID", "")},
}

TEXT_PREMIUM_CREDIT_COST = int(env_or_default("TEXT_PREMIUM_CREDIT_COST", "2"))
CREDITS_PER_10000_MMK = 100  # 100 credits = 10,000 MMK -> 1 credit = 100 MMK

FREE_DAILY_TEXT_LIMIT = 10
PREMIUM_DAILY_TEXT_LIMIT = 50
DAILY_FREE_IMAGE_LIMIT = 2  # same for free & premium, per spec

LANGUAGE_INSTRUCTION = (
    "You are fully fluent in both Burmese (Myanmar) and English. Always "
    "reply in the same language the user writes in — natural, correct, "
    "idiomatic Burmese when they write in Burmese, clear natural English "
    "when they write in English. Mix naturally if they mix."
)
SYSTEM_PROMPT_CHAT = "You are Shadow AI, a sharp, warm, futuristic companion. Speak naturally and concisely. " + LANGUAGE_INSTRUCTION
SYSTEM_PROMPT_WORK = "You are Shadow AI in WORK MODE: a senior engineer who executes fully, however long the output needs to be, no placeholders. " + LANGUAGE_INSTRUCTION
SYSTEM_PROMPT_CODE = "You are Shadow AI in CODE MODE: respond with complete runnable code, fenced with the right language tag, no truncation. " + LANGUAGE_INSTRUCTION

THINKING_KEYWORDS = ("code", "debug", "fix", "algorithm", "architecture", "design", "build", "plan", "analy", "compare", "calculate", "solve", "refactor", "optimi", "strategy", "explain how", "write a", "create a", "implement", "why")


def needs_thinking(message: str, mode: str) -> bool:
    if mode in ("work", "code"):
        return True
    if len(message) > 140:
        return True
    return any(k in message.lower() for k in THINKING_KEYWORDS)


sessions: Dict[str, dict] = {}
projects: Dict[str, dict] = {}
library: Dict[str, dict] = {}
pending_codes: Dict[str, dict] = {}

ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "")


def today_str() -> str:
    return datetime.date.today().isoformat()


def is_premium_active(user: User) -> bool:
    if user.tier != "premium":
        return False
    if not user.premium_expires_at:
        return False
    try:
        return datetime.date.fromisoformat(user.premium_expires_at) >= datetime.date.today()
    except ValueError:
        return False


def get_user_by_identity(db, identity: str) -> Optional[User]:
    if "@" not in identity:
        return None  # anonymous users have no DB row -> always free, no credits
    return db.query(User).filter(User.email == identity).first()


def get_or_create_usage(db, identity: str) -> UsageRecord:
    row = db.query(UsageRecord).filter(UsageRecord.identity == identity, UsageRecord.date == today_str()).first()
    if not row:
        row = UsageRecord(identity=identity, date=today_str(), text_count=0, free_image_count=0)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def add_credit_transaction(db, identity: str, amount: int, reason: str, new_balance: int):
    db.add(CreditTransaction(identity=identity, amount=amount, reason=reason, balance_after=new_balance))
    db.commit()


class InsufficientCreditsError(Exception):
    def __init__(self, needed: int, have: int):
        self.needed = needed
        self.have = have


def deduct_credits(identity: str, amount: int, reason: str):
    """Atomic check-and-deduct. Raises InsufficientCreditsError if the user
    can't afford it — call this BEFORE triggering any paid model, never after."""
    if amount <= 0:
        return
    db = get_db()
    try:
        user = get_user_by_identity(db, identity)
        if not user:
            raise InsufficientCreditsError(needed=amount, have=0)
        if user.credit_balance < amount:
            raise InsufficientCreditsError(needed=amount, have=user.credit_balance)
        user.credit_balance -= amount
        db.commit()
        add_credit_transaction(db, identity, -amount, reason, user.credit_balance)
    finally:
        db.close()


def resolve_text_permission(identity: str, model_key: str) -> dict:
    """Returns {'allowed': bool, 'via': 'free_quota'|'premium_quota'|'credits', 'reason': str|None}"""
    model = TEXT_MODELS.get(model_key)
    if not model:
        return {"allowed": False, "reason": "Unknown model."}

    db = get_db()
    try:
        user = db.query(User).filter(User.email == identity).first()
        if user and user.is_admin == 1:
            return {"allowed": True, "via": "admin_bypass", "spent": 0}
        user = get_user_by_identity(db, identity)
        premium = bool(user and is_premium_active(user))
        usage = get_or_create_usage(db, identity)

        if model["tier"] == "free":
            limit = PREMIUM_DAILY_TEXT_LIMIT if premium else FREE_DAILY_TEXT_LIMIT
            if usage.text_count < limit:
                usage.text_count += 1
                db.commit()
                return {"allowed": True, "via": "quota", "remaining": limit - usage.text_count}
        else:
            if premium:
                if usage.text_count < PREMIUM_DAILY_TEXT_LIMIT:
                    usage.text_count += 1
                    db.commit()
                    return {"allowed": True, "via": "quota", "remaining": PREMIUM_DAILY_TEXT_LIMIT - usage.text_count}

        # Quota exhausted (or free user on a premium model) -> pay with credits
        if not user:
            return {"allowed": False, "reason": "Daily free limit reached. Log in and buy credits to continue."}
        if user.credit_balance < TEXT_PREMIUM_CREDIT_COST:
            return {"allowed": False, "reason": f"Daily limit reached and you have {user.credit_balance} credits (need {TEXT_PREMIUM_CREDIT_COST})."}
        user.credit_balance -= TEXT_PREMIUM_CREDIT_COST
        db.commit()
        add_credit_transaction(db, identity, -TEXT_PREMIUM_CREDIT_COST, f"text:{model_key}", user.credit_balance)
        return {"allowed": True, "via": "credits", "spent": TEXT_PREMIUM_CREDIT_COST}
    finally:
        db.close()


def resolve_image_permission(identity: str, model_key: str) -> dict:
    model = IMAGE_MODELS.get(model_key)
    if not model:
        return {"allowed": False, "reason": "Unknown model."}

    db = get_db()
    try:
        user = db.query(User).filter(User.email == identity).first()
        if user and user.is_admin == 1:
            return {"allowed": True, "via": "admin_bypass", "remaining": 9999}
        usage = get_or_create_usage(db, identity)
        if model["free_quota"]:
            if usage.free_image_count < DAILY_FREE_IMAGE_LIMIT:
                usage.free_image_count += 1
                db.commit()
                return {"allowed": True, "via": "quota", "remaining": DAILY_FREE_IMAGE_LIMIT - usage.free_image_count}
            return {"allowed": False, "reason": f"Free image quota ({DAILY_FREE_IMAGE_LIMIT}/day) used up. Try again tomorrow or pick a paid model."}

        cost = model["credit_cost"]
        user = get_user_by_identity(db, identity)
        if not user:
            return {"allowed": False, "reason": "Log in and buy credits to use this model."}
        if user.credit_balance < cost:
            return {"allowed": False, "reason": f"Need {cost} credits, you have {user.credit_balance}."}
        user.credit_balance -= cost
        db.commit()
        add_credit_transaction(db, identity, -cost, f"image:{model_key}", user.credit_balance)
        return {"allowed": True, "via": "credits", "spent": cost}
    finally:
        db.close()


def resolve_video_permission(identity: str, model_key: str) -> dict:
    model = VIDEO_MODELS.get(model_key)
    if not model:
        return {"allowed": False, "reason": "Unknown model."}
    cost = model["credit_cost"]
    db = get_db()
    try:
        
        user = get_user_by_identity(db, identity)
        if not user:
            return {"allowed": False, "reason": "Log in and buy credits to use video generation."}
        if user.is_admin == 1:
            return {"allowed": True, "via": "admin_bypass", "spent": 0}
        if user.credit_balance < cost:
            return {"allowed": False, "reason": f"Need {cost} credits, you have {user.credit_balance}."}
        user.credit_balance -= cost
        db.commit()
        add_credit_transaction(db, identity, -cost, f"video:{model_key}", user.credit_balance)
        return {"allowed": True, "via": "credits", "spent": cost}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Daily housekeeping job — the actual "reset" already happens for free
# because usage is looked up per calendar date; this job just prunes old
# rows so the table doesn't grow forever.
# ---------------------------------------------------------------------------
def daily_cleanup_job():
    cutoff = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
    db = get_db()
    try:
        db.query(UsageRecord).filter(UsageRecord.date < cutoff).delete()
        db.commit()
    finally:
        db.close()


scheduler = BackgroundScheduler()
scheduler.add_job(daily_cleanup_job, "cron", hour=0, minute=5)
scheduler.start()


def new_chat_session():
    return {"history": []}


def send_chat(session_obj, message, system_prompt, attachments, want_thinking, model_id):
    content = [{"type": "text", "text": message}]
    for att in attachments:
        if att["kind"] == "image":
            content.append({"type": "image_url", "image_url": {"url": f"data:{att['mime']};base64,{att['data']}"}})
        elif att.get("text"):
            content[0]["text"] += f"\n\n[Attached file: {att['filename']}]\n{att['text']}"
    session_obj["history"].append({"role": "user", "content": content})
    messages = [{"role": "system", "content": system_prompt}] + session_obj["history"]
    resp = openrouter_client.chat.completions.create(
        model=model_id, messages=messages, max_tokens=8000 if want_thinking else 2000, extra_headers=OR_HEADERS,
    )
    reply = resp.choices[0].message.content
    session_obj["history"].append({"role": "assistant", "content": reply})
    return reply


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Attachment(BaseModel):
    kind: Literal["image", "file"]
    filename: str
    mime: str
    data: Optional[str] = None
    text: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    mode: Literal["chat", "work", "code"] = "chat"
    attachments: List[Attachment] = []
    identity: str = "anonymous"
    text_model: str = "deepseek-v4-flash"


class ImageRequest(BaseModel):
    prompt: str
    identity: str = "anonymous"
    image_model: str = "nano-banana-2"


class VideoRequest(BaseModel):
    prompt: str
    identity: str = "anonymous"
    video_model: str = "wan-3"


class ProjectIn(BaseModel):
    name: str
    description: str = ""


class LibraryItemIn(BaseModel):
    title: str
    content: str
    language: str = "text"


class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: str


class VerifyRequest(BaseModel):
    email: str
    code: str


class SetPremiumRequest(BaseModel):
    email: str
    months: int = 1


class AddCreditsRequest(BaseModel):
    email: str
    credits: int
    note: str = ""


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
   # return templates.TemplateResponse("index.html", {"request": request, "provider": "openrouter"})
    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={"provider": "openrouter"}
)

@app.get("/api/model-catalog")
async def model_catalog():
    return {
        "text": {k: {"label": v["label"], "tier": v["tier"]} for k, v in TEXT_MODELS.items()},
        "image": {k: {"label": v["label"], "credit_cost": v["credit_cost"], "free_quota": v["free_quota"]} for k, v in IMAGE_MODELS.items()},
        "video": {k: {"label": v["label"], "credit_cost": v["credit_cost"]} for k, v in VIDEO_MODELS.items()},
        "credit_rate": {"credits": CREDITS_PER_10000_MMK, "mmk": 10000},
        "limits": {"free_text": FREE_DAILY_TEXT_LIMIT, "premium_text": PREMIUM_DAILY_TEXT_LIMIT, "free_image": DAILY_FREE_IMAGE_LIMIT},
    }


@app.post("/api/new-session")
async def new_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"engine_state": new_chat_session(), "mode": "chat"}
    return {"session_id": session_id}


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    message = payload.message.strip()
    if not message and not payload.attachments:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if payload.text_model not in TEXT_MODELS:
        raise HTTPException(status_code=400, detail="Unknown text model.")

    perm = resolve_text_permission(payload.identity, payload.text_model)
    if not perm["allowed"]:
        return {"limit_reached": True, "reason": perm["reason"]}

    session_id = payload.session_id
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"engine_state": new_chat_session(), "mode": payload.mode}

    session = sessions[session_id]
    session["mode"] = payload.mode
    system_prompt = {"work": SYSTEM_PROMPT_WORK, "code": SYSTEM_PROMPT_CODE}.get(payload.mode, SYSTEM_PROMPT_CHAT)
    thinking_flag = needs_thinking(message, payload.mode)
    model_id = TEXT_MODELS[payload.text_model]["model_id"]

    loop = asyncio.get_event_loop()

    def _run():
        return send_chat(session["engine_state"], message, system_prompt, [a.dict() for a in payload.attachments], thinking_flag, model_id)

    try:
        reply = await loop.run_in_executor(None, _run)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI provider error: {exc}")

    return {"session_id": session_id, "reply": reply, "thinking": None, "used_thinking_animation": thinking_flag, "billing": perm}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return {"ok": True}


@app.post("/api/image")
async def generate_image(payload: ImageRequest):
    if payload.image_model not in IMAGE_MODELS:
        raise HTTPException(status_code=400, detail="Unknown image model.")
    perm = resolve_image_permission(payload.identity, payload.image_model)
    if not perm["allowed"]:
        return {"supported": False, "limit_reached": True, "reason": perm["reason"]}

    model_id = IMAGE_MODELS[payload.image_model]["model_id"]
    if not model_id:
        return {"supported": False, "message": f"{IMAGE_MODELS[payload.image_model]['label']} isn't wired to a real OpenRouter model ID yet — set it in .env."}

    loop = asyncio.get_event_loop()

    def _run():
        resp = openrouter_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": [{"type": "text", "text": payload.prompt}]}],
            extra_headers=OR_HEADERS,
            modalities=["image", "text"],
        )
        msg = resp.choices[0].message
        images = getattr(msg, "images", None)
        if images:
            url = images[0]["image_url"]["url"]
            return url.split(",", 1)[1] if url.startswith("data:") else None
        return None

    try:
        b64 = await loop.run_in_executor(None, _run)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image provider error: {exc}")

    if not b64:
        return {"supported": False, "message": "Model responded without an image — check the model ID and that it supports image output on OpenRouter."}
    return {"supported": True, "image_base64": b64, "mime": "image/png", "billing": perm}


@app.post("/api/video")
async def generate_video(payload: VideoRequest):
    if payload.video_model not in VIDEO_MODELS:
        raise HTTPException(status_code=400, detail="Unknown video model.")
    perm = resolve_video_permission(payload.identity, payload.video_model)
    if not perm["allowed"]:
        return {"supported": False, "limit_reached": True, "reason": perm["reason"]}
    return {"supported": False, "message": f"{VIDEO_MODELS[payload.video_model]['label']} credits were deducted, but no video-generation call is wired yet — plug in the provider's API here.", "billing": perm}


@app.get("/api/account")
async def get_account(identity: str = "anonymous"):
    db = get_db()
    try:
        user = get_user_by_identity(db, identity)
        usage = get_or_create_usage(db, identity)
        premium = bool(user and is_premium_active(user))
        text_limit = PREMIUM_DAILY_TEXT_LIMIT if premium else FREE_DAILY_TEXT_LIMIT
        is_admin = bool(user and user.is_admin == 1)
        if is_admin:
            credit_display = 999999
            text_limit = 999999
            image_limit_display = 999999
        else:
            credit_display = user.credit_balance if user else 0
            image_limit_display = DAILY_FREE_IMAGE_LIMIT
        return {
            "logged_in": user is not None,
            "premium": premium,
            "premium_expires_at": user.premium_expires_at if user else "",
           "credit_balance": credit_display,
            "text_used": 0 if is_admin else usage.text_count,
            "text_limit": text_limit,
            "free_image_used": 0 if is_admin else usage.free_image_count,
            "free_image_limit": image_limit_display,
        }
    finally:
        db.close()


EMAIL_RE = re.compile(r"^[^@\s]+@gmail\.com$", re.IGNORECASE)


@app.post("/api/auth/register")
async def register(payload: RegisterRequest):
    email = payload.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Please use a valid gmail.com address.")
    db = get_db()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            db.add(User(email=email, first_name=payload.first_name.strip(), last_name=payload.last_name.strip()))
            db.commit()
    finally:
        db.close()
    code = f"{random.randint(0, 999999):06d}"
    pending_codes[email] = {"code": code, "expires_at": time.time() + 600}
    return {"email": email, "demo_code": code}


@app.post("/api/auth/verify")
async def verify(payload: VerifyRequest):
    email = payload.email.strip().lower()
    record = pending_codes.get(email)
    if not record or record["code"] != payload.code.strip():
        raise HTTPException(status_code=400, detail="Invalid or expired code.")
    if time.time() > record["expires_at"]:
        raise HTTPException(status_code=400, detail="Code expired, request a new one.")
    pending_codes.pop(email, None)
    db = get_db()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=400, detail="No account found for this email.")
        return {"user": {"email": user.email, "first_name": user.first_name, "last_name": user.last_name, "is_admin": bool(user.is_admin)}}
    finally:
        db.close()


@app.get("/api/projects")
async def list_projects():
    return {"projects": list(projects.values())}


@app.post("/api/projects")
async def create_project(payload: ProjectIn):
    pid = str(uuid.uuid4())
    projects[pid] = {"id": pid, "name": payload.name, "description": payload.description}
    return projects[pid]


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    projects.pop(project_id, None)
    return {"ok": True}


@app.get("/api/library")
async def list_library():
    return {"items": list(library.values())}


@app.post("/api/library")
async def create_library_item(payload: LibraryItemIn):
    lid = str(uuid.uuid4())
    library[lid] = {"id": lid, "title": payload.title, "content": payload.content, "language": payload.language}
    return library[lid]


@app.delete("/api/library/{item_id}")
async def delete_library_item(item_id: str):
    library.pop(item_id, None)
    return {"ok": True}


GUTENDEX_URL = "https://gutendex.com/books/"
FORMAT_PRIORITY = [("application/pdf", "PDF"), ("application/epub+zip", "EPUB"), ("application/x-mobipocket-ebook", "MOBI"), ("text/plain; charset=utf-8", "TXT"), ("text/plain; charset=us-ascii", "TXT"), ("text/html; charset=utf-8", "HTML")]


def _format_book(book: dict) -> dict:
    formats = book.get("formats", {})
    cover = formats.get("image/jpeg")
    download_url, format_label = None, None
    for mime, label in FORMAT_PRIORITY:
        if mime in formats:
            download_url, format_label = formats[mime], label
            break
    authors = ", ".join(a["name"] for a in book.get("authors", [])) or "Unknown Author"
    return {"id": book["id"], "title": book["title"], "authors": authors, "cover": cover, "download_url": download_url, "format_label": format_label}


@app.get("/api/books")
async def search_books(q: str = "", page: int = 1):
    params = {"page": page}
    if q:
        params["search"] = q
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(GUTENDEX_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Book library unreachable: {exc}")
    return {"results": [_format_book(b) for b in data.get("results", [])], "has_more": bool(data.get("next"))}


MOTIVATIONAL_QUERIES = ["acres of diamonds", "science of getting rich", "self help", "success", "as a man thinketh"]


@app.get("/api/books/motivational")
async def motivational_books():
    seen: Dict[int, dict] = {}
    async with httpx.AsyncClient(timeout=12) as client:
        for q in MOTIVATIONAL_QUERIES:
            try:
                resp = await client.get(GUTENDEX_URL, params={"search": q})
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                continue
            for book in data.get("results", [])[:6]:
                if book["id"] not in seen:
                    seen[book["id"]] = _format_book(book)
            if len(seen) >= 12:
                break
    return {"results": list(seen.values())[:12]}


@app.get("/api/books/myanmar")
async def myanmar_books():
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(GUTENDEX_URL, params={"languages": "my"})
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Book library unreachable: {exc}")
    return {"results": [_format_book(b) for b in data.get("results", [])]}


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------
def check_admin(x_admin_key: str):
    if not ADMIN_SECRET or x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key.")


@app.post("/api/admin/set-premium")
async def admin_set_premium(payload: SetPremiumRequest, x_admin_key: str = Header(default="")):
    check_admin(x_admin_key)
    db = get_db()
    try:
        user = db.query(User).filter(User.email == payload.email.lower()).first()
        if not user:
            raise HTTPException(status_code=404, detail="No account with this email — ask them to sign up first.")
        base = datetime.date.today()
        if user.tier == "premium" and user.premium_expires_at:
            try:
                existing = datetime.date.fromisoformat(user.premium_expires_at)
                if existing > base:
                    base = existing
            except ValueError:
                pass
        new_expiry = base + datetime.timedelta(days=30 * payload.months)
        user.tier = "premium"
        user.premium_expires_at = new_expiry.isoformat()
        db.commit()
        return {"ok": True, "email": user.email, "premium_expires_at": user.premium_expires_at}
    finally:
        db.close()


@app.post("/api/admin/revoke-premium")
async def admin_revoke_premium(payload: SetPremiumRequest, x_admin_key: str = Header(default="")):
    check_admin(x_admin_key)
    db = get_db()
    try:
        user = db.query(User).filter(User.email == payload.email.lower()).first()
        if not user:
            raise HTTPException(status_code=404, detail="No such user.")
        user.tier = "free"
        user.premium_expires_at = ""
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.post("/api/admin/add-credits")
async def admin_add_credits(payload: AddCreditsRequest, x_admin_key: str = Header(default="")):
    check_admin(x_admin_key)
    db = get_db()
    try:
        user = db.query(User).filter(User.email == payload.email.lower()).first()
        if not user:
            raise HTTPException(status_code=404, detail="No account with this email — ask them to sign up first.")
        user.credit_balance += payload.credits
        db.commit()
        add_credit_transaction(db, user.email, payload.credits, payload.note or "admin top-up", user.credit_balance)
        return {"ok": True, "email": user.email, "credit_balance": user.credit_balance}
    finally:
        db.close()


@app.get("/api/admin/users")
async def admin_list_users(x_admin_key: str = Header(default="")):
    check_admin(x_admin_key)
    db = get_db()
    try:
        users = db.query(User).all()
        return {"users": [{
            "email": u.email, "name": f"{u.first_name} {u.last_name}".strip(),
            "tier": u.tier, "premium_expires_at": u.premium_expires_at,
            "credit_balance": u.credit_balance, "is_admin": bool(u.is_admin),
        } for u in users]}
    finally:
        db.close()


@app.get("/api/admin/transactions")
async def admin_transactions(email: str, x_admin_key: str = Header(default="")):
    check_admin(x_admin_key)
    db = get_db()
    try:
        rows = db.query(CreditTransaction).filter(CreditTransaction.identity == email.lower()).order_by(CreditTransaction.id.desc()).limit(50).all()
        return {"transactions": [{"amount": r.amount, "reason": r.reason, "balance_after": r.balance_after, "at": r.created_at.isoformat()} for r in rows]}
    finally:
        db.close()


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel():
    return """
<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Shadow AI Admin</title>
<style>
body{background:#0B0B0F;color:#E5E7EB;font-family:sans-serif;padding:30px;max-width:900px;margin:0 auto;}
h1{color:#A855F7;} h3{color:#06B6D4;margin-top:28px;}
input,select,button{padding:10px;border-radius:8px;border:1px solid #A855F7;background:#111;color:#fff;margin:4px 0;width:100%;box-sizing:border-box;}
button{background:#06B6D4;color:#000;font-weight:700;cursor:pointer;border:none;}
.row{display:flex;gap:10px;} .row>*{flex:1;}
table{width:100%;border-collapse:collapse;margin-top:14px;} td,th{padding:8px;border-bottom:1px solid #333;text-align:left;font-size:.82rem;}
#lock{max-width:400px;margin:80px auto;text-align:center;}
.msg{font-size:.85rem;margin-top:6px;}
</style></head><body>

<div id="lock">
  <h1>🔒 Shadow AI Admin</h1>
  <input type="password" id="secret-input" placeholder="Admin secret">
  <button onclick="unlock()">Unlock</button>
</div>

<div id="panel" style="display:none;">
  <h1>⚡ Shadow AI Admin Panel</h1>

  <h3>1. Grant Premium (after Telegram payment confirmed)</h3>
  <input type="text" id="premium-email" placeholder="user@gmail.com">
  <div class="row">
    <input type="number" id="premium-months" value="1" min="1" placeholder="Months">
    <button onclick="setPremium()">Grant Premium</button>
    <button onclick="revokePremium()" style="background:#f87171;">Revoke</button>
  </div>
  <p class="msg" id="premium-msg"></p>

  <h3>2. Add Credits (after Telegram top-up payment confirmed)</h3>
  <input type="text" id="credit-email" placeholder="user@gmail.com">
  <div class="row">
    <input type="number" id="credit-amount" placeholder="Credits to add (100 credits = 10,000 MMK)">
    <button onclick="addCredits()">Add Credits</button>
  </div>
  <p class="msg" id="credit-msg"></p>

  <h3>Registered Users</h3>
  <button onclick="loadUsers()">Refresh</button>
  <table id="users-table"><thead><tr><th>Email</th><th>Name</th><th>Tier</th><th>Premium Until</th><th>Credits</th></tr></thead><tbody></tbody></table>
</div>

<script>
let secret = localStorage.getItem("shadow_admin_secret") || "";
function unlock() {
  secret = document.getElementById("secret-input").value;
  localStorage.setItem("shadow_admin_secret", secret);
  document.getElementById("lock").style.display = "none";
  document.getElementById("panel").style.display = "block";
  loadUsers();
}
if (secret) unlock();

async function setPremium() {
  const email = document.getElementById("premium-email").value.trim();
  const months = parseInt(document.getElementById("premium-months").value || "1");
  const res = await fetch("/api/admin/set-premium", { method: "POST", headers: { "Content-Type": "application/json", "x-admin-key": secret }, body: JSON.stringify({ email, months }) });
  const data = await res.json();
  document.getElementById("premium-msg").textContent = res.ok ? `✓ ${email} is premium until ${data.premium_expires_at}` : `✗ ${data.detail}`;
  loadUsers();
}
async function revokePremium() {
  const email = document.getElementById("premium-email").value.trim();
  const res = await fetch("/api/admin/revoke-premium", { method: "POST", headers: { "Content-Type": "application/json", "x-admin-key": secret }, body: JSON.stringify({ email }) });
  const data = await res.json();
  document.getElementById("premium-msg").textContent = res.ok ? `✓ ${email} reverted to free` : `✗ ${data.detail}`;
  loadUsers();
}
async function addCredits() {
  const email = document.getElementById("credit-email").value.trim();
  const credits = parseInt(document.getElementById("credit-amount").value || "0");
  const res = await fetch("/api/admin/add-credits", { method: "POST", headers: { "Content-Type": "application/json", "x-admin-key": secret }, body: JSON.stringify({ email, credits }) });
  const data = await res.json();
  document.getElementById("credit-msg").textContent = res.ok ? `✓ ${email} now has ${data.credit_balance} credits` : `✗ ${data.detail}`;
  loadUsers();
}
async function loadUsers() {
  const res = await fetch("/api/admin/users", { headers: { "x-admin-key": secret } });
  if (!res.ok) { document.getElementById("premium-msg").textContent = "✗ Invalid admin secret"; return; }
  const data = await res.json();
  document.querySelector("#users-table tbody").innerHTML = data.users.map(u =>
    `<tr><td>${u.email}</td><td>${u.name}</td><td>${u.tier}</td><td>${u.premium_expires_at || "—"}</td><td>${u.credit_balance}</td></tr>`).join("");
}
</script>
</body></html>
"""


@app.get("/health")
async def health():
    return {"status": "online", "provider": "openrouter"}


# ==========================================
# 📊 ADMIN DASHBOARD - BACKEND DATA & ACTIONS
# ==========================================
from fastapi.responses import HTMLResponse
from fastapi import Request

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard_page():
    """Admin Dashboard မျက်နှာပြင် (HTML) ကို ပြသမည့် လမ်းကြောင်း"""
    db = get_db() if 'get_db' in globals() else get_db()
    
   
    total_users = db.query(User).count()
    free_users = db.query(User).filter(User.tier == "free").count()
    premium_users = db.query(User).filter(User.tier == "premium").count()
    admin_users = db.query(User).filter(User.is_admin == 1).count()
    
    
    recent_users = db.query(User).order_by(User.id.desc()).limit(10).all()
    db.close()
    
   
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Shadow AI - Admin Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://jsdelivr.net" rel="stylesheet">
        <style>
            body {{ background-color: #0b0f19; color: #e2e8f0; font-family: sans-serif; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .card-custom {{ background: #111827; border: 1px solid #1f2937; border-radius: 12px; }}
            .table-custom {{ background: #111827; color: #e2e8f0; }}
            .table-custom th {{ background: #1f2937; color: #9ca3af; text-align: center; }}
            .table-custom td {{ border-color: #1f2937; text-align: center; vertical-align: middle; }}
            .btn-premium {{ background-color: #8b5cf6; color: white; }}
            .btn-premium:hover {{ background-color: #7c3aed; color: white; }}
            
            /* ✨ Screen ကြီးလျှင် ၄ ကွက် ပြိုင်တူပေါ်စေရန် */
            @media (min-width: 768px) {{
                .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }}
            }}
        </style>
    </head>
    <body class="py-5">
        <div class="container px-4">
            <h2 class="mb-4 text-center text-primary fw-bold">// SHADOW_AI_ADMIN_PANEL</h2>
            
            <!-- 📊 USER STATISTICS STATS (စာသားများ အလယ်ကပ်ထားပါသည်) -->
            <div class="stats-grid mb-5">
                <div class="card-custom p-4 text-center">
                    <h6 class="text-muted text-uppercase small mb-2">စုစုပေါင်း အသုံးပြုသူ</h6>
                    <h2 class="text-white fw-bold m-0">{total_users}</h2>
                </div>
                <div class="card-custom p-4 text-center border-info">
                    <h6 class="text-info text-uppercase small mb-2">Free Users</h6>
                    <h2 class="text-info fw-bold m-0">{free_users}</h2>
                </div>
                <div class="card-custom p-4 text-center border-warning">
                    <h6 class="text-warning text-uppercase small mb-2">Premium Users</h6>
                    <h2 class="text-warning fw-bold m-0">{premium_users}</h2>
                </div>
                <div class="card-custom p-4 text-center border-danger">
                    <h6 class="text-danger text-uppercase small mb-2">Admins</h6>
                    <h2 class="text-danger fw-bold m-0">{admin_users}</h2>
                </div>
            </div>

            <div class="row g-4">
                <!-- 🛠️ ACTIONS FORM (PREMIUM / CREDIT ပေးရန်အကွက် - စာသားများ အလယ်ကပ်ထားပါသည်) -->
                <div class="col-md-5">
                    <div class="card-custom p-4 mb-4 text-center">
                        <h5 class="mb-3 text-primary fw-bold">⭐ Premium အဆင့်မြှင့်တင်ရန်</h5>
                        <form action="/admin/action/upgrade" method="POST">
                            <div class="mb-3">
                                <label class="form-label text-muted small">User Email</label>
                                <input type="email" name="email" class="form-control bg-dark text-white border-secondary text-center" placeholder="example@gmail.com" required>
                            </div>
                            <div class="mb-3">
                                <label class="form-label text-muted small">သက်တမ်း (လအရေအတွက်)</label>
                                <select name="months" class="form-select bg-dark text-white border-secondary text-center">
                                    <option value="1">၁ လစာ (၃၀ ရက်)</option>
                                    <option value="3">၃ လစာ (၉၀ ရက်)</option>
                                    <option value="6">၆ လစာ (၁၈၀ ရက်)</option>
                                    <option value="12">၁ နှစ်စာ (၃၆၅ ရက်)</option>
                                </select>
                            </div>
                            <button type="submit" class="btn btn-premium w-100 fw-bold">Premium ထည့်ပေးမည်</button>
                        </form>
                    </div>

                    <div class="card-custom p-4 text-center">
                        <h5 class="mb-3 text-success fw-bold">💰 Credit (ဒင်္ဂါးပြား) ဖြည့်ပေးရန်</h5>
                        <form action="/admin/action/credit" method="POST">
                            <div class="mb-3">
                                <label class="form-label text-muted small">User Email</label>
                                <input type="email" name="email" class="form-control bg-dark text-white border-secondary text-center" placeholder="example@gmail.com" required>
                            </div>
                            <div class="mb-3">
                                <label class="form-label text-muted small">Credit ပမာဏ</label>
                                <input type="number" name="amount" class="form-control bg-dark text-white border-secondary text-center" placeholder="ဥပမာ - 100" required>
                            </div>
                            <button type="submit" class="btn btn-success w-100 fw-bold">Credit ဖြည့်ပေးမည်</button>
                        </form>
                    </div>
                </div>

                <!-- 📋 RECENT USERS LIST (မကြာသေးမီက ဆောက်ထားသော user များဇယား - အလယ်ကပ်ထားပါသည်) -->
                <div class="col-md-7">
                    <div class="card-custom p-4 h-100">
                        <h5 class="mb-3 text-muted text-center fw-bold">နောက်ဆုံးဖွင့်ထားသည့် အကောင့် ၁၀ ခု</h5>
                        <div class="table-responsive">
                            <table class="table table-custom table-hover m-0">
                                <thead>
                                    <tr>
                                        <th>Email</th>
                                        <th>Tier</th>
                                        <th>Credit</th>
                                        <th>Admin</th>
                                    </tr>
                                </thead>
                                <tbody>
    """
    for u in recent_users:
        tier_badge = f'<span class="badge bg-warning text-dark">Premium</span>' if u.tier == "premium" else f'<span class="badge bg-secondary">Free</span>'
        admin_badge = f'<span class="badge bg-danger">Yes</span>' if u.is_admin == 1 else f'<span class="badge bg-dark">-</span>'
        html_content += f"""
                                    <tr>
                                        <td class="small">{u.email}</td>
                                        <td>{tier_badge}</td>
                                        <td>{u.credit_balance}</td>
                                        <td>{admin_badge}</td>
                                    </tr>
        """
        
    html_content += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return html_content


from fastapi.param_functions import Form
from fastapi.responses import RedirectResponse

@app.post("/admin/action/upgrade")
async def handle_admin_upgrade(email: str = Form(...), months: int = Form(...)):
    from database import upgrade_to_premium
    upgrade_to_premium(email, months)
    return RedirectResponse(url="/admin/dashboard", status_code=303)


@app.post("/admin/action/credit")
async def handle_admin_credit(email: str = Form(...), amount: int = Form(...)):
    from database import add_user_credit
    add_user_credit(email, amount, reason="Admin Manual Top-up")
    return RedirectResponse(url="/admin/dashboard", status_code=303)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
   
    init_db() 
    
    
    admin_email = "htetthar912@gmail.com" 
    
    
    register_user(admin_email, "Admin", "User")
    
    
    result = set_user_as_admin(admin_email, status=1)
    print(result)    