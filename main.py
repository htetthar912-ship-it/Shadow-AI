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
from database import register_user, set_user_as_admin, upgrade_to_premium, revoke_premium, add_user_credit

load_dotenv()
init_db()

# One-line "just add the key and launch" admin bootstrap — runs every start,
# so the designated admin email always has admin rights without any manual
# script or button click.
_bootstrap_email = os.environ.get("ADMIN_BOOTSTRAP_EMAIL", "").strip().lower()
if _bootstrap_email:
    register_user(_bootstrap_email, "Admin", "")
    set_user_as_admin(_bootstrap_email, status=1)

app = FastAPI(title="Shadow AI")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def env_or_default(key: str, default: str) -> str:
    val = os.environ.get(key, "").strip()
    return val if val else default


if not os.environ.get("OPENROUTER_API_KEY"):
    raise RuntimeError("OPENROUTER_API_KEY is required in your .env file.")

from openai import OpenAI as _OpenAIClient
openrouter_client = _OpenAIClient(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
OR_HEADERS = {"HTTP-Referer": "https://shadow-ai.local", "X-Title": "Shadow AI"}

TEXT_MODELS = {
    "deepseek-v4-flash": {"label": "DeepSeek V4 Flash", "tier": "free", "model_id": env_or_default("DEEPSEEK_MODEL_ID", "deepseek/deepseek-chat")},
    "gpt-5.6-sol": {"label": "GPT-5.6 Sol", "tier": "premium", "model_id": env_or_default("GPT_SOL_MODEL_ID", "openai/gpt-4.1")},
    "claude-sonnet-5": {"label": "Claude Sonnet 5", "tier": "premium", "model_id": env_or_default("CLAUDE_MODEL_ID", "anthropic/claude-sonnet-4.6")},
    "gemini-3.7-flash": {"label": "Gemini 3.7 Flash", "tier": "premium", "model_id": env_or_default("GEMINI_MODEL_ID", "google/gemini-2.5-flash")},
}
IMAGE_MODELS = {
    "nano-banana-2": {"label": "Nano Banana 2 (Free)", "credit_cost": 0, "free_quota": True, "model_id": env_or_default("NANO_BANANA_MODEL_ID", "google/gemini-2.5-flash-image")},
    "midjourney": {"label": "Midjourney", "credit_cost": 15, "free_quota": False, "model_id": env_or_default("MIDJOURNEY_MODEL_ID", "")},
    "flux-ultra": {"label": "FLUX.1 Ultra", "credit_cost": 15, "free_quota": False, "model_id": env_or_default("FLUX_ULTRA_MODEL_ID", "black-forest-labs/flux-1.1-pro")},
    "dalle-4": {"label": "DALL-E 4", "credit_cost": 10, "free_quota": False, "model_id": env_or_default("DALLE4_MODEL_ID", "openai/gpt-image-1")},
}
VIDEO_MODELS = {
    "runway-gen45": {"label": "Runway Gen-4.5", "credit_cost": 70, "model_id": env_or_default("RUNWAY_MODEL_ID", "")},
    "wan-3": {"label": "Wan 3.0", "credit_cost": 35, "model_id": env_or_default("WAN3_MODEL_ID", "")},
    "seedance-mini": {"label": "Seedance Mini", "credit_cost": 35, "model_id": env_or_default("SEEDANCE_MODEL_ID", "")},
}

TEXT_PREMIUM_CREDIT_COST = int(env_or_default("TEXT_PREMIUM_CREDIT_COST", "2"))
CREDITS_PER_10000_MMK = 100
FREE_DAILY_TEXT_LIMIT = 10
PREMIUM_DAILY_TEXT_LIMIT = 50
FREE_DAILY_IMAGE_LIMIT = 2
PREMIUM_DAILY_IMAGE_LIMIT = 10

LANGUAGE_INSTRUCTION = (
    "You are fully fluent in both Burmese (Myanmar) and English. Always reply in the same "
    "language the user writes in — natural, correct, idiomatic Burmese when they write in "
    "Burmese, clear natural English when they write in English. Mix naturally if they mix."
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
    if user.tier != "premium" or not user.premium_expires_at:
        return False
    try:
        return datetime.date.fromisoformat(user.premium_expires_at) >= datetime.date.today()
    except ValueError:
        return False


def get_user_by_identity(db, identity: str) -> Optional[User]:
    if "@" not in identity:
        return None
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


def resolve_text_permission(identity: str, model_key: str) -> dict:
    model = TEXT_MODELS.get(model_key)
    if not model:
        return {"allowed": False, "reason": "Unknown model."}
    db = get_db()
    try:
        user = get_user_by_identity(db, identity)
        if user and user.is_admin == 1:
            return {"allowed": True, "via": "admin_bypass"}

        premium = bool(user and is_premium_active(user))
        usage = get_or_create_usage(db, identity)

        if model["tier"] == "free":
            limit = PREMIUM_DAILY_TEXT_LIMIT if premium else FREE_DAILY_TEXT_LIMIT
            if usage.text_count < limit:
                usage.text_count += 1
                db.commit()
                return {"allowed": True, "via": "quota", "remaining": limit - usage.text_count}
        elif premium and usage.text_count < PREMIUM_DAILY_TEXT_LIMIT:
            usage.text_count += 1
            db.commit()
            return {"allowed": True, "via": "quota", "remaining": PREMIUM_DAILY_TEXT_LIMIT - usage.text_count}

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
        user = get_user_by_identity(db, identity)
        if user and user.is_admin == 1:
            return {"allowed": True, "via": "admin_bypass"}

        premium = bool(user and is_premium_active(user))
        usage = get_or_create_usage(db, identity)

        if model["free_quota"]:
            limit = PREMIUM_DAILY_IMAGE_LIMIT if premium else FREE_DAILY_IMAGE_LIMIT
            if usage.free_image_count < limit:
                usage.free_image_count += 1
                db.commit()
                return {"allowed": True, "via": "quota", "remaining": limit - usage.free_image_count}
            return {"allowed": False, "reason": f"Free image quota ({limit}/day) used up. Try again tomorrow or pick a paid model."}

        cost = model["credit_cost"]
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
            return {"allowed": True, "via": "admin_bypass"}
        if user.credit_balance < cost:
            return {"allowed": False, "reason": f"Need {cost} credits, you have {user.credit_balance}."}
        user.credit_balance -= cost
        db.commit()
        add_credit_transaction(db, identity, -cost, f"video:{model_key}", user.credit_balance)
        return {"allowed": True, "via": "credits", "spent": cost}
    finally:
        db.close()


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
    resp = openrouter_client.chat.completions.create(model=model_id, messages=messages, max_tokens=8000 if want_thinking else 2000, extra_headers=OR_HEADERS)
    reply = resp.choices[0].message.content
    session_obj["history"].append({"role": "assistant", "content": reply})
    return reply


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
    return templates.TemplateResponse(request=request, name="index.html", context={"provider": "openrouter"})


@app.get("/api/model-catalog")
async def model_catalog():
    return {
        "text": {k: {"label": v["label"], "tier": v["tier"]} for k, v in TEXT_MODELS.items()},
        "image": {k: {"label": v["label"], "credit_cost": v["credit_cost"], "free_quota": v["free_quota"]} for k, v in IMAGE_MODELS.items()},
        "video": {k: {"label": v["label"], "credit_cost": v["credit_cost"]} for k, v in VIDEO_MODELS.items()},
        "credit_rate": {"credits": CREDITS_PER_10000_MMK, "mmk": 10000},
        "limits": {"free_text": FREE_DAILY_TEXT_LIMIT, "premium_text": PREMIUM_DAILY_TEXT_LIMIT, "free_image": FREE_DAILY_IMAGE_LIMIT, "premium_image": PREMIUM_DAILY_IMAGE_LIMIT},
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
            model=model_id, messages=[{"role": "user", "content": [{"type": "text", "text": payload.prompt}]}],
            extra_headers=OR_HEADERS, modalities=["image", "text"],
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
        return {"supported": False, "message": "Model responded without an image — check the model ID."}
    return {"supported": True, "image_base64": b64, "mime": "image/png", "billing": perm}


@app.post("/api/video")
async def generate_video(payload: VideoRequest):
    if payload.video_model not in VIDEO_MODELS:
        raise HTTPException(status_code=400, detail="Unknown video model.")
    perm = resolve_video_permission(payload.identity, payload.video_model)
    if not perm["allowed"]:
        return {"supported": False, "limit_reached": True, "reason": perm["reason"]}
    return {"supported": False, "message": f"{VIDEO_MODELS[payload.video_model]['label']} credits were deducted, but no video-generation call is wired yet.", "billing": perm}


@app.get("/api/account")
async def get_account(identity: str = "anonymous"):
    db = get_db()
    try:
        user = get_user_by_identity(db, identity)
        usage = get_or_create_usage(db, identity)
        premium = bool(user and is_premium_active(user))
        is_admin = bool(user and user.is_admin == 1)

        if is_admin:
            return {"logged_in": True, "premium": True, "is_admin": True, "premium_expires_at": "",
                    "credit_balance": 999999, "text_used": 0, "text_limit": 999999,
                    "free_image_used": 0, "free_image_limit": 999999}

        text_limit = PREMIUM_DAILY_TEXT_LIMIT if premium else FREE_DAILY_TEXT_LIMIT
        image_limit = PREMIUM_DAILY_IMAGE_LIMIT if premium else FREE_DAILY_IMAGE_LIMIT
        return {
            "logged_in": user is not None, "premium": premium, "is_admin": False,
            "premium_expires_at": user.premium_expires_at if user else "",
            "credit_balance": user.credit_balance if user else 0,
            "text_used": usage.text_count, "text_limit": text_limit,
            "free_image_used": usage.free_image_count, "free_image_limit": image_limit,
        }
    finally:
        db.close()


EMAIL_RE = re.compile(r"^[^@\s]+@gmail\.com$", re.IGNORECASE)


@app.post("/api/auth/register")
async def register(payload: RegisterRequest):
    email = payload.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Please use a valid gmail.com address.")
    register_user(email, payload.first_name.strip(), payload.last_name.strip())
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
# Admin — single secured panel. Every action requires x-admin-key header.
# ---------------------------------------------------------------------------
def check_admin(x_admin_key: str):
    if not ADMIN_SECRET or x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key.")


@app.get("/api/admin/stats")
async def admin_stats(x_admin_key: str = Header(default="")):
    check_admin(x_admin_key)
    db = get_db()
    try:
        return {
            "total": db.query(User).count(),
            "free": db.query(User).filter(User.tier == "free").count(),
            "premium": db.query(User).filter(User.tier == "premium").count(),
            "admins": db.query(User).filter(User.is_admin == 1).count(),
        }
    finally:
        db.close()


@app.get("/api/admin/users")
async def admin_list_users(x_admin_key: str = Header(default=""), q: str = ""):
    check_admin(x_admin_key)
    db = get_db()
    try:
        query = db.query(User)
        if q:
            like = f"%{q}%"
            query = query.filter(User.email.like(like))
        users = query.order_by(User.id.desc()).all()
        return {"users": [{
            "email": u.email, "name": f"{u.first_name} {u.last_name}".strip(),
            "tier": u.tier, "premium_expires_at": u.premium_expires_at,
            "credit_balance": u.credit_balance, "is_admin": bool(u.is_admin),
        } for u in users]}
    finally:
        db.close()


@app.post("/api/admin/set-premium")
async def admin_set_premium(payload: SetPremiumRequest, x_admin_key: str = Header(default="")):
    check_admin(x_admin_key)
    new_expiry = upgrade_to_premium(payload.email, payload.months)
    if new_expiry is None:
        raise HTTPException(status_code=404, detail="No account with this email — ask them to sign up first.")
    return {"ok": True, "email": payload.email.lower(), "premium_expires_at": new_expiry}


@app.post("/api/admin/revoke-premium")
async def admin_revoke_premium(payload: SetPremiumRequest, x_admin_key: str = Header(default="")):
    check_admin(x_admin_key)
    if not revoke_premium(payload.email):
        raise HTTPException(status_code=404, detail="No such user.")
    return {"ok": True}


@app.post("/api/admin/add-credits")
async def admin_add_credits(payload: AddCreditsRequest, x_admin_key: str = Header(default="")):
    check_admin(x_admin_key)
    new_balance = add_user_credit(payload.email, payload.credits, payload.note or "admin top-up")
    if new_balance is None:
        raise HTTPException(status_code=404, detail="No account with this email — ask them to sign up first.")
    return {"ok": True, "email": payload.email.lower(), "credit_balance": new_balance}


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel():
    return """
<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Shadow AI — Admin</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root { --purple:#A855F7; --cyan:#06B6D4; --bg:#0B0B0F; }
  * { box-sizing:border-box; }
  body { background:#000; color:#E5E7EB; font-family:'Rajdhani',sans-serif; margin:0; padding:24px; }
  h1 { font-family:'Orbitron',sans-serif; font-size:1.2rem; letter-spacing:.1em; background:linear-gradient(90deg,var(--purple),var(--cyan)); -webkit-background-clip:text; background-clip:text; color:transparent; }
  h2 { font-family:'Orbitron',sans-serif; font-size:.9rem; color:var(--cyan); letter-spacing:.08em; margin-top:0; }
  .wrap { max-width:1000px; margin:0 auto; }
  .grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:26px; }
  @media (max-width:700px) { .grid { grid-template-columns:repeat(2,1fr); } }
  .stat { background:var(--bg); border:1px solid rgba(168,85,247,.3); border-radius:14px; padding:18px; text-align:center; }
  .stat .num { font-family:'Orbitron',sans-serif; font-size:1.8rem; color:#fff; }
  .stat .lbl { font-size:.72rem; color:#7C7C8A; text-transform:uppercase; letter-spacing:.08em; margin-top:4px; }
  .panel { background:var(--bg); border:1px solid rgba(168,85,247,.25); border-radius:14px; padding:20px; margin-bottom:20px; }
  input,select,button { font-family:inherit; }
  input,select { width:100%; background:rgba(255,255,255,.05); border:1px solid rgba(168,85,247,.3); border-radius:8px; color:#fff; padding:10px; margin-bottom:10px; font-size:.9rem; }
  button { background:linear-gradient(90deg,var(--purple),var(--cyan)); color:#000; font-weight:700; border:none; border-radius:8px; padding:10px 16px; cursor:pointer; }
  button.danger { background:#f87171; }
  .row { display:flex; gap:10px; } .row>* { flex:1; }
  table { width:100%; border-collapse:collapse; margin-top:10px; }
  td,th { padding:9px; border-bottom:1px solid rgba(255,255,255,.08); text-align:left; font-size:.8rem; }
  th { color:#7C7C8A; text-transform:uppercase; font-size:.68rem; letter-spacing:.06em; }
  .badge { padding:2px 8px; border-radius:10px; font-size:.68rem; font-weight:700; }
  .badge.premium { background:rgba(251,191,36,.2); color:#fbbf24; }
  .badge.free { background:rgba(255,255,255,.08); color:#7C7C8A; }
  .badge.admin { background:rgba(248,113,113,.2); color:#f87171; }
  .msg { font-size:.82rem; margin:8px 0; }
  #lock { max-width:360px; margin:100px auto; text-align:center; }
</style></head><body>

<div id="lock">
  <h1>🔒 SHADOW AI ADMIN</h1>
  <input type="password" id="secret-input" placeholder="Admin secret key">
  <button style="width:100%" onclick="unlock()">Unlock</button>
</div>

<div id="panel" class="wrap" style="display:none;">
  <h1>⚡ SHADOW AI — ADMIN PANEL</h1>
  <div class="grid" id="stats-grid"></div>

  <div class="panel">
    <h2>👑 Grant Premium Subscription</h2>
    <input type="email" id="premium-email" placeholder="user@gmail.com">
    <div class="row">
      <select id="premium-months"><option value="1">1 month</option><option value="3">3 months</option><option value="6">6 months</option><option value="12">12 months</option></select>
      <button onclick="setPremium()">Grant Premium</button>
      <button class="danger" onclick="revokePremium()">Revoke</button>
    </div>
    <div class="msg" id="premium-msg"></div>
  </div>

  <div class="panel">
    <h2>💳 Add Credits (100 credits = 10,000 Ks)</h2>
    <input type="email" id="credit-email" placeholder="user@gmail.com">
    <div class="row">
      <input type="number" id="credit-amount" placeholder="Credits, e.g. 100">
      <button onclick="addCredits()">Add Credits</button>
    </div>
    <div class="msg" id="credit-msg"></div>
  </div>

  <div class="panel">
    <h2>Registered Users</h2>
    <input type="text" id="user-search" placeholder="Search by email..." oninput="loadUsers()">
    <table><thead><tr><th>Email</th><th>Name</th><th>Status</th><th>Premium Until</th><th>Credits</th></tr></thead><tbody id="users-body"></tbody></table>
  </div>
</div>

<script>
let secret = localStorage.getItem("shadow_admin_secret") || "";
function unlock() {
  secret = document.getElementById("secret-input").value;
  localStorage.setItem("shadow_admin_secret", secret);
  document.getElementById("lock").style.display = "none";
  document.getElementById("panel").style.display = "block";
  loadStats(); loadUsers();
}
if (secret) unlock();

async function loadStats() {
  const res = await fetch("/api/admin/stats", { headers: { "x-admin-key": secret } });
  if (!res.ok) { alert("Invalid admin secret"); localStorage.removeItem("shadow_admin_secret"); location.reload(); return; }
  const d = await res.json();
  document.getElementById("stats-grid").innerHTML = `
    <div class="stat"><div class="num">${d.total}</div><div class="lbl">Total Users</div></div>
    <div class="stat"><div class="num" style="color:#7C7C8A;">${d.free}</div><div class="lbl">Free</div></div>
    <div class="stat"><div class="num" style="color:#fbbf24;">${d.premium}</div><div class="lbl">Premium</div></div>
    <div class="stat"><div class="num" style="color:#f87171;">${d.admins}</div><div class="lbl">Admins</div></div>`;
}
async function setPremium() {
  const email = document.getElementById("premium-email").value.trim();
  const months = parseInt(document.getElementById("premium-months").value);
  const res = await fetch("/api/admin/set-premium", { method:"POST", headers:{"Content-Type":"application/json","x-admin-key":secret}, body: JSON.stringify({email, months}) });
  const d = await res.json();
  document.getElementById("premium-msg").textContent = res.ok ? `✓ ${email} is premium until ${d.premium_expires_at}` : `✗ ${d.detail}`;
  loadStats(); loadUsers();
}
async function revokePremium() {
  const email = document.getElementById("premium-email").value.trim();
  const res = await fetch("/api/admin/revoke-premium", { method:"POST", headers:{"Content-Type":"application/json","x-admin-key":secret}, body: JSON.stringify({email}) });
  const d = await res.json();
  document.getElementById("premium-msg").textContent = res.ok ? `✓ ${email} reverted to free` : `✗ ${d.detail}`;
  loadStats(); loadUsers();
}
async function addCredits() {
  const email = document.getElementById("credit-email").value.trim();
  const credits = parseInt(document.getElementById("credit-amount").value || "0");
  const res = await fetch("/api/admin/add-credits", { method:"POST", headers:{"Content-Type":"application/json","x-admin-key":secret}, body: JSON.stringify({email, credits}) });
  const d = await res.json();
  document.getElementById("credit-msg").textContent = res.ok ? `✓ ${email} now has ${d.credit_balance} credits` : `✗ ${d.detail}`;
  loadUsers();
}
async function loadUsers() {
  const q = document.getElementById("user-search").value.trim();
  const res = await fetch(`/api/admin/users?q=${encodeURIComponent(q)}`, { headers: { "x-admin-key": secret } });
  if (!res.ok) return;
  const d = await res.json();
  document.getElementById("users-body").innerHTML = d.users.map(u => `
    <tr><td>${u.email}</td><td>${u.name || "—"}</td>
    <td>${u.is_admin ? '<span class="badge admin">Admin</span>' : (u.tier === "premium" ? '<span class="badge premium">Premium</span>' : '<span class="badge free">Free</span>')}</td>
    <td>${u.premium_expires_at || "—"}</td><td>${u.credit_balance}</td></tr>`).join("");
}
</script>
</body></html>
"""

# =========================================================
# ADMIN DASHBOARD UI
# =========================================================


@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Shadow AI — Admin Console</title>

<style>
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

:root {
    --bg: #070912;
    --panel: #0d111d;
    --panel2: #111625;
    --border: rgba(255,255,255,.09);
    --text: #f5f7ff;
    --muted: #8992aa;
    --purple: #8b5cf6;
    --blue: #3b82f6;
    --green: #22c55e;
    --orange: #f59e0b;
    --red: #ef4444;
}

body {
    min-height: 100vh;
    background:
        radial-gradient(circle at 20% 0%, rgba(124,58,237,.16), transparent 28%),
        radial-gradient(circle at 85% 15%, rgba(59,130,246,.10), transparent 25%),
        var(--bg);
    color: var(--text);
    font-family: Inter, Arial, sans-serif;
}

/* ================= SIDEBAR ================= */

.sidebar {
    position: fixed;
    left: 0;
    top: 0;
    bottom: 0;
    width: 250px;
    background: rgba(8,11,20,.94);
    border-right: 1px solid var(--border);
    padding: 24px 15px;
    z-index: 100;
    backdrop-filter: blur(20px);
}

.logo {
    display: flex;
    align-items: center;
    gap: 13px;
    padding: 0 10px 30px;
}

.logo-icon {
    width: 44px;
    height: 44px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    font-weight: 900;
    background: linear-gradient(135deg,#8b5cf6,#38bdf8);
    box-shadow: 0 10px 35px rgba(139,92,246,.35);
}

.logo-title {
    font-size: 16px;
    font-weight: 800;
}

.logo-sub {
    color: #6f7890;
    font-size: 9px;
    letter-spacing: 2px;
    margin-top: 3px;
}

.section-title {
    color: #667087;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 0 11px;
    margin: 10px 0 10px;
}

.nav {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.nav button {
    border: 0;
    background: transparent;
    color: #929bb0;
    width: 100%;
    text-align: left;
    padding: 12px 13px;
    border-radius: 12px;
    cursor: pointer;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: .2s;
}

.nav button:hover {
    color: white;
    background: rgba(255,255,255,.05);
}

.nav button.active {
    color: white;
    background: linear-gradient(
        90deg,
        rgba(139,92,246,.20),
        rgba(139,92,246,.05)
    );
    border-left: 3px solid var(--purple);
}

.nav-icon {
    width: 18px;
    text-align: center;
}

.sidebar-bottom {
    position: absolute;
    bottom: 25px;
    left: 15px;
    right: 15px;
}

/* ================= MAIN ================= */

.main {
    margin-left: 250px;
    padding: 28px 32px 50px;
    min-height: 100vh;
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 28px;
}

.page-title {
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -.7px;
}

.page-subtitle {
    color: var(--muted);
    margin-top: 6px;
    font-size: 13px;
}

.admin-profile {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 9px 14px;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: rgba(255,255,255,.025);
}

.avatar {
    width: 36px;
    height: 36px;
    border-radius: 11px;
    background: linear-gradient(135deg,#8b5cf6,#60a5fa);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
}

.profile-name {
    font-size: 12px;
    font-weight: 700;
}

.profile-role {
    color: #737d95;
    font-size: 9px;
    margin-top: 2px;
}

/* ================= CARDS ================= */

.stats {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 15px;
    margin-bottom: 18px;
}

.card {
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.035),
            rgba(255,255,255,.012)
        );
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 20px;
    position: relative;
    overflow: hidden;
}

.card::after {
    content: "";
    position: absolute;
    width: 130px;
    height: 130px;
    right: -65px;
    top: -65px;
    background: rgba(139,92,246,.13);
    filter: blur(30px);
    border-radius: 50%;
}

.card-label {
    color: #818ba4;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}

.card-number {
    font-size: 31px;
    font-weight: 800;
    margin-top: 17px;
}

.card-foot {
    margin-top: 12px;
    font-size: 10px;
    color: #818ba4;
}

.green { color: var(--green); }
.purple { color: #a78bfa; }
.blue { color: #60a5fa; }
.orange { color: var(--orange); }

/* ================= GRID ================= */

.content-grid {
    display: grid;
    grid-template-columns: 1.7fr 1fr;
    gap: 17px;
}

.panel {
    background: rgba(10,14,25,.82);
    border: 1px solid var(--border);
    border-radius: 17px;
    overflow: hidden;
}

.panel-head {
    padding: 20px;
    border-bottom: 1px solid var(--border);
}

.panel-title {
    font-size: 14px;
    font-weight: 750;
}

.panel-sub {
    color: #69738b;
    font-size: 10px;
    margin-top: 5px;
}

/* ================= USERS ================= */

.table-wrap {
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th {
    color: #667087;
    font-size: 9px;
    letter-spacing: 1px;
    text-transform: uppercase;
    text-align: left;
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
}

td {
    padding: 15px 18px;
    border-bottom: 1px solid rgba(255,255,255,.045);
    font-size: 11px;
}

.user-email {
    font-weight: 650;
}

.user-name {
    color: #68738b;
    font-size: 9px;
    margin-top: 3px;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 5px 8px;
    border-radius: 7px;
    font-size: 9px;
    font-weight: 700;
}

.badge-premium {
    background: rgba(139,92,246,.13);
    color: #b59cff;
}

.badge-free {
    background: rgba(148,163,184,.09);
    color: #9ca7ba;
}

.badge-admin {
    background: rgba(245,158,11,.12);
    color: #fbbf24;
}

.empty {
    padding: 45px 20px;
    text-align: center;
    color: #69738b;
    font-size: 12px;
}

/* ================= QUICK ACTIONS ================= */

.actions {
    padding: 18px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 11px;
}

.action-btn {
    min-height: 92px;
    text-align: left;
    border: 1px solid var(--border);
    border-radius: 13px;
    background: rgba(255,255,255,.025);
    color: white;
    cursor: pointer;
    padding: 15px;
    transition: .2s;
}

.action-btn:hover {
    transform: translateY(-2px);
    border-color: rgba(139,92,246,.5);
    background: rgba(139,92,246,.08);
}

.action-icon {
    font-size: 16px;
    margin-bottom: 14px;
}

.action-title {
    font-size: 11px;
    font-weight: 700;
}

.action-sub {
    color: #69738b;
    font-size: 9px;
    margin-top: 4px;
}

/* ================= SEARCH ================= */

.toolbar {
    display: flex;
    gap: 10px;
    padding: 16px 18px;
    border-bottom: 1px solid var(--border);
}

.search {
    flex: 1;
    background: #080c16;
    border: 1px solid var(--border);
    border-radius: 10px;
    color: white;
    outline: none;
    padding: 10px 12px;
    font-size: 11px;
}

.search:focus {
    border-color: var(--purple);
}

.btn {
    border: 0;
    border-radius: 10px;
    padding: 10px 15px;
    cursor: pointer;
    color: white;
    font-weight: 700;
    background: linear-gradient(135deg,#7c3aed,#4f46e5);
}

.btn:hover {
    opacity: .9;
}

/* ================= PAGE ================= */

.page {
    display: none;
}

.page.active {
    display: block;
}

/* ================= MODAL ================= */

.modal-bg {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,.72);
    backdrop-filter: blur(7px);
    align-items: center;
    justify-content: center;
    z-index: 999;
}

.modal-bg.show {
    display: flex;
}

.modal {
    width: min(430px,92vw);
    background: #0d111d;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 23px;
    box-shadow: 0 30px 100px rgba(0,0,0,.55);
}

.modal h2 {
    font-size: 17px;
    margin-bottom: 8px;
}

.modal p {
    color: var(--muted);
    font-size: 11px;
    margin-bottom: 17px;
}

.modal input,
.modal select {
    width: 100%;
    background: #080c16;
    border: 1px solid var(--border);
    color: white;
    padding: 11px;
    border-radius: 9px;
    outline: none;
    margin-bottom: 10px;
}

.modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 9px;
    margin-top: 10px;
}

.cancel {
    background: rgba(255,255,255,.07);
}

/* ================= TOAST ================= */

.toast {
    position: fixed;
    right: 22px;
    bottom: 22px;
    background: #111827;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 13px 17px;
    font-size: 11px;
    display: none;
    z-index: 2000;
    box-shadow: 0 15px 40px rgba(0,0,0,.4);
}

.toast.show {
    display: block;
}

/* ================= RESPONSIVE ================= */

@media(max-width:1000px) {
    .stats {
        grid-template-columns: repeat(2,1fr);
    }

    .content-grid {
        grid-template-columns: 1fr;
    }
}

@media(max-width:720px) {
    .sidebar {
        width: 70px;
        padding: 15px 8px;
    }

    .logo-title,
    .logo-sub,
    .section-title,
    .nav button span:not(.nav-icon) {
        display: none;
    }

    .logo {
        justify-content: center;
        padding: 0 0 25px;
    }

    .nav button {
        justify-content: center;
    }

    .main {
        margin-left: 70px;
        padding: 20px 15px;
    }

    .stats {
        grid-template-columns: 1fr;
    }

    .topbar {
        align-items: flex-start;
    }

    .admin-profile {
        display: none;
    }
}
</style>
</head>

<body>

<!-- ================= SIDEBAR ================= -->

<aside class="sidebar">

    <div class="logo">
        <div class="logo-icon">S</div>
        <div>
            <div class="logo-title">SHADOW AI</div>
            <div class="logo-sub">ADMIN CONSOLE<br>-Thar Htet Swe </div>
        </div>
    </div>

    <div class="section-title">MAIN</div>

    <div class="nav">

        <button class="active" onclick="showPage('dashboard', this)">
            <span class="nav-icon">▦</span>
            <span>Dashboard</span>
        </button>

        <button onclick="showPage('users', this)">
            <span class="nav-icon">♙</span>
            <span>Users</span>
        </button>

        <button onclick="showPage('premium', this)">
            <span class="nav-icon">★</span>
            <span>Premium</span>
        </button>

        <button onclick="showPage('credits', this)">
            <span class="nav-icon">◇</span>
            <span>Credits</span>
        </button>

        <button onclick="showPage('analytics', this)">
            <span class="nav-icon">⌁</span>
            <span>Analytics</span>
        </button>

        <button onclick="showPage('transactions', this)">
            <span class="nav-icon">≡</span>
            <span>Transactions</span>
        </button>

    </div>

    <div class="section-title" style="margin-top:30px;">SYSTEM</div>

    <div class="nav">
        <button onclick="showPage('settings', this)">
            <span class="nav-icon">⚙</span>
            <span>Settings</span>
        </button>
    </div>

    <div class="sidebar-bottom">
        <div class="nav">
            <button onclick="logout()">
                <span class="nav-icon">↪</span>
                <span>Logout</span>
            </button>
        </div>
    </div>

</aside>

<!-- ================= MAIN ================= -->

<main class="main">

    <!-- DASHBOARD -->

    <section id="dashboard" class="page active">

        <div class="topbar">
            <div>
                <div class="page-title">Dashboard</div>
                <div class="page-subtitle">
                    Shadow AI platform overview
                </div>
            </div>

            <div class="admin-profile">
                <div class="avatar">A</div>
                <div>
                    <div class="profile-name">Shadow Admin</div>
                    <div class="profile-role">System Administrator</div>
                </div>
            </div>
        </div>

        <div class="stats">

            <div class="card">
                <div class="card-label">TOTAL USERS</div>
                <div id="totalUsers" class="card-number">—</div>
                <div class="card-foot green">● Registered accounts</div>
            </div>

            <div class="card">
                <div class="card-label">FREE USERS</div>
                <div id="freeUsers" class="card-number">—</div>
                <div class="card-foot purple">● Standard accounts</div>
            </div>

            <div class="card">
                <div class="card-label">PREMIUM USERS</div>
                <div id="premiumUsers" class="card-number">—</div>
                <div class="card-foot blue">● Premium accounts</div>
            </div>

            <div class="card">
                <div class="card-label">ADMINS</div>
                <div id="adminUsers" class="card-number">—</div>
                <div class="card-foot orange">● Administrators</div>
            </div>

        </div>

        <div class="content-grid">

            <div class="panel">

                <div class="panel-head">
                    <div class="panel-title">User Overview</div>
                    <div class="panel-sub">
                        Current registered accounts
                    </div>
                </div>

                <div class="table-wrap">

                    <table>
                        <thead>
                            <tr>
                                <th>User</th>
                                <th>Tier</th>
                                <th>Credits</th>
                                <th>Role</th>
                            </tr>
                        </thead>

                        <tbody id="dashboardUsers">
                        </tbody>
                    </table>

                    <div id="dashboardEmpty" class="empty">
                        Loading users...
                    </div>

                </div>

            </div>

            <div class="panel">

                <div class="panel-head">
                    <div class="panel-title">Quick Actions</div>
                    <div class="panel-sub">
                        Manage Shadow AI
                    </div>
                </div>

                <div class="actions">

                    <button class="action-btn"
                            onclick="showPageByName('users')">
                        <div class="action-icon">♙</div>
                        <div class="action-title">Manage Users</div>
                        <div class="action-sub">View all accounts</div>
                    </button>

                    <button class="action-btn"
                            onclick="openPremiumModal()">
                        <div class="action-icon">★</div>
                        <div class="action-title">Grant Premium</div>
                        <div class="action-sub">Upgrade an account</div>
                    </button>

                    <button class="action-btn"
                            onclick="openCreditModal()">
                        <div class="action-icon">◇</div>
                        <div class="action-title">Add Credits</div>
                        <div class="action-sub">Increase balance</div>
                    </button>

                    <button class="action-btn"
                            onclick="showPageByName('analytics')">
                        <div class="action-icon">⌁</div>
                        <div class="action-title">Analytics</div>
                        <div class="action-sub">View activity</div>
                    </button>

                </div>

            </div>

        </div>

    </section>


    <!-- USERS -->

    <section id="users" class="page">

        <div class="topbar">
            <div>
                <div class="page-title">Users</div>
                <div class="page-subtitle">
                    Manage Shadow AI accounts
                </div>
            </div>
        </div>

        <div class="panel">

            <div class="toolbar">
                <input
                    id="userSearch"
                    class="search"
                    placeholder="Search users by email..."
                    onkeydown="if(event.key==='Enter') loadUsers()"
                >

                <button class="btn" onclick="loadUsers()">
                    Search
                </button>

                <button class="btn" onclick="loadUsers()">
                    Refresh
                </button>
            </div>

            <div class="table-wrap">

                <table>
                    <thead>
                        <tr>
                            <th>Email</th>
                            <th>Name</th>
                            <th>Tier</th>
                            <th>Credits</th>
                            <th>Premium Until</th>
                            <th>Role</th>
                        </tr>
                    </thead>

                    <tbody id="usersTable"></tbody>
                </table>

                <div id="usersEmpty" class="empty">
                    Loading users...
                </div>

            </div>

        </div>

    </section>


    <!-- PREMIUM -->

    <section id="premium" class="page">

        <div class="topbar">
            <div>
                <div class="page-title">Premium</div>
                <div class="page-subtitle">
                    Manage premium subscriptions
                </div>
            </div>

            <button class="btn" onclick="openPremiumModal()">
                + Grant Premium
            </button>
        </div>

        <div class="panel">

            <div class="panel-head">
                <div class="panel-title">Premium Accounts</div>
                <div class="panel-sub">
                    Active premium users
                </div>
            </div>

            <div class="table-wrap">

                <table>
                    <thead>
                        <tr>
                            <th>Email</th>
                            <th>Name</th>
                            <th>Tier</th>
                            <th>Expires</th>
                        </tr>
                    </thead>

                    <tbody id="premiumTable"></tbody>

                </table>

            </div>

        </div>

    </section>


    <!-- CREDITS -->

    <section id="credits" class="page">

        <div class="topbar">
            <div>
                <div class="page-title">Credits</div>
                <div class="page-subtitle">
                    Manage user credit balances
                </div>
            </div>

            <button class="btn" onclick="openCreditModal()">
                + Add Credits
            </button>
        </div>

        <div class="panel">

            <div class="panel-head">
                <div class="panel-title">Credit Balances</div>
                <div class="panel-sub">
                    Current user balances
                </div>
            </div>

            <div class="table-wrap">

                <table>
                    <thead>
                        <tr>
                            <th>Email</th>
                            <th>Balance</th>
                            <th>Tier</th>
                        </tr>
                    </thead>

                    <tbody id="creditsTable"></tbody>

                </table>

            </div>

        </div>

    </section>


    <!-- ANALYTICS -->

    <section id="analytics" class="page">

        <div class="topbar">
            <div>
                <div class="page-title">Analytics</div>
                <div class="page-subtitle">
                    Shadow AI activity overview
                </div>
            </div>
        </div>

        <div class="stats">

            <div class="card">
                <div class="card-label">TOTAL USERS</div>
                <div id="analyticsUsers" class="card-number">—</div>
            </div>

            <div class="card">
                <div class="card-label">PREMIUM USERS</div>
                <div id="analyticsPremium" class="card-number">—</div>
            </div>

            <div class="card">
                <div class="card-label">CREDIT BALANCE</div>
                <div id="analyticsCredits" class="card-number">—</div>
            </div>

            <div class="card">
                <div class="card-label">AI REQUESTS</div>
                <div id="analyticsRequests" class="card-number">—</div>
            </div>

        </div>

        <div class="panel">

            <div class="panel-head">
                <div class="panel-title">Platform Activity</div>
                <div class="panel-sub">
                    Analytics data from Shadow AI
                </div>
            </div>

            <div class="empty" id="analyticsInfo">
                Loading analytics...
            </div>

        </div>

    </section>


    <!-- TRANSACTIONS -->

    <section id="transactions" class="page">

        <div class="topbar">
            <div>
                <div class="page-title">Transactions</div>
                <div class="page-subtitle">
                    Credit transaction history
                </div>
            </div>
        </div>

        <div class="panel">

            <div class="panel-head">
                <div class="panel-title">Transaction History</div>
                <div class="panel-sub">
                    Credit activity
                </div>
            </div>

            <div class="empty">
                Transaction history will appear here.
            </div>

        </div>

    </section>


    <!-- SETTINGS -->

    <section id="settings" class="page">

        <div class="topbar">
            <div>
                <div class="page-title">Settings</div>
                <div class="page-subtitle">
                    Admin console settings
                </div>
            </div>
        </div>

        <div class="panel">

            <div class="panel-head">
                <div class="panel-title">Admin Configuration</div>
                <div class="panel-sub">
                    Shadow AI administration
                </div>
            </div>

            <div style="padding:20px;">

                <button class="btn" onclick="changeAdminKey()">
                    Change Admin Key
                </button>

            </div>

        </div>

    </section>

</main>


<!-- ================= PREMIUM MODAL ================= -->

<div id="premiumModal" class="modal-bg">

    <div class="modal">

        <h2>Grant Premium</h2>

        <p>
            Upgrade a Shadow AI account.
        </p>

        <input
            id="premiumEmail"
            placeholder="User email"
        >

        <select id="premiumMonths">
            <option value="1">1 Month</option>
            <option value="3">3 Months</option>
            <option value="6">6 Months</option>
            <option value="12">12 Months</option>
        </select>

        <div class="modal-actions">

            <button
                class="btn cancel"
                onclick="closeModals()">
                Cancel
            </button>

            <button
                class="btn"
                onclick="grantPremium()">
                Grant Premium
            </button>

        </div>

    </div>

</div>


<!-- ================= CREDIT MODAL ================= -->

<div id="creditModal" class="modal-bg">

    <div class="modal">

        <h2>Add Credits</h2>

        <p>
            Increase a user's Shadow AI credit balance.
        </p>

        <input
            id="creditEmail"
            placeholder="User email"
        >

        <input
            id="creditAmount"
            type="number"
            min="1"
            placeholder="Credit amount"
        >

        <input
            id="creditReason"
            placeholder="Reason (optional)"
        >

        <div class="modal-actions">

            <button
                class="btn cancel"
                onclick="closeModals()">
                Cancel
            </button>

            <button
                class="btn"
                onclick="addCredits()">
                Add Credits
            </button>

        </div>

    </div>

</div>


<div id="toast" class="toast"></div>


<script>

/* =====================================================
   SHADOW AI ADMIN CONSOLE
===================================================== */

let usersCache = [];


/* ================= ADMIN KEY ================= */

function getAdminKey() {

    let key = localStorage.getItem("shadow_admin_key");

    if (!key) {

        key = prompt(
            "Enter Shadow AI Admin Key:"
        );

        if (key) {
            localStorage.setItem(
                "shadow_admin_key",
                key
            );
        }
    }

    return key || "";
}


/* ================= API ================= */

async function apiFetch(url, options = {}) {

    const key = getAdminKey();

    options.headers = {
        ...(options.headers || {}),
        "x-admin-key": key,
        "Content-Type": "application/json"
    };

    const response = await fetch(
        url,
        options
    );

    if (response.status === 401 ||
        response.status === 403) {

        localStorage.removeItem(
            "shadow_admin_key"
        );

        throw new Error(
            "Admin authentication failed."
        );
    }

    if (!response.ok) {

        let text = "";

        try {
            text = await response.text();
        } catch {}

        throw new Error(
            text || `HTTP ${response.status}`
        );
    }

    return response.json();
}


/* ================= NAVIGATION ================= */

function showPage(name, button) {

    document
        .querySelectorAll(".page")
        .forEach(page => {
            page.classList.remove("active");
        });

    const page =
        document.getElementById(name);

    if (page) {
        page.classList.add("active");
    }

    document
        .querySelectorAll(".nav button")
        .forEach(btn => {
            btn.classList.remove("active");
        });

    if (button) {
        button.classList.add("active");
    }

    if (name === "users") {
        loadUsers();
    }

    if (name === "premium") {
        loadPremium();
    }

    if (name === "credits") {
        loadCredits();
    }

    if (name === "analytics") {
        loadAnalytics();
    }
}


function showPageByName(name) {

    const buttons =
        document.querySelectorAll(".nav button");

    for (const button of buttons) {

        if (
            button.innerText
                .toLowerCase()
                .includes(name.toLowerCase())
        ) {

            showPage(name, button);
            return;
        }
    }

    showPage(name);
}


/* ================= DASHBOARD ================= */

async function loadDashboard() {

    try {

        const data =
            await apiFetch("/api/admin/stats");

        console.log(
            "STATS:",
            data
        );

        const stats =
            data.stats || data;

        const total =
            stats.total_users ??
            stats.users ??
            0;

        const free =
            stats.free_users ??
            stats.free ??
            0;

        const premium =
            stats.premium_users ??
            stats.premium ??
            0;

        const admins =
            stats.admins ??
            stats.admin_users ??
            0;

        document.getElementById(
            "totalUsers"
        ).textContent = total;

        document.getElementById(
            "freeUsers"
        ).textContent = free;

        document.getElementById(
            "premiumUsers"
        ).textContent = premium;

        document.getElementById(
            "adminUsers"
        ).textContent = admins;

    } catch (error) {

        console.error(
            "Stats error:",
            error
        );

        /*
         * If stats endpoint has a different
         * response format, user count is
         * still loaded from /users.
         */
    }

    await loadUsers(true);
}


/* ================= USERS ================= */

async function loadUsers(dashboardOnly = false) {

    try {

        const search =
            document.getElementById(
                "userSearch"
            )?.value || "";

        let url =
            "/api/admin/users";

        if (search.trim()) {

            url +=
                "?q=" +
                encodeURIComponent(
                    search.trim()
                );
        }

        const data =
            await apiFetch(url);

        console.log(
            "USERS:",
            data
        );

        usersCache =
            data.users || [];

        renderUsers(
            usersCache,
            dashboardOnly
        );

        /*
         * Even if /stats response is different,
         * calculate the cards directly from users.
         */
        updateStatsFromUsers(
            usersCache
        );

    } catch (error) {

        console.error(
            "Users error:",
            error
        );

        if (!dashboardOnly) {

            document.getElementById(
                "usersEmpty"
            ).textContent =
                error.message;
        }

        document.getElementById(
            "dashboardEmpty"
        ).textContent =
            "Unable to load users.";

    }
}


/* ================= CALCULATE STATS ================= */

function updateStatsFromUsers(users) {

    const total =
        users.length;

    const premium =
        users.filter(
            u =>
                String(
                    u.tier || ""
                ).toLowerCase() ===
                "premium"
        ).length;

    const admins =
        users.filter(
            u =>
                Boolean(u.is_admin)
        ).length;

    const free =
        total - premium;

    document.getElementById(
        "totalUsers"
    ).textContent = total;

    document.getElementById(
        "freeUsers"
    ).textContent = free;

    document.getElementById(
        "premiumUsers"
    ).textContent = premium;

    document.getElementById(
        "adminUsers"
    ).textContent = admins;
}


/* ================= RENDER USERS ================= */

function renderUsers(
    users,
    dashboardOnly = false
) {

    const dashboardBody =
        document.getElementById(
            "dashboardUsers"
        );

    const usersBody =
        document.getElementById(
            "usersTable"
        );

    const dashboardEmpty =
        document.getElementById(
            "dashboardEmpty"
        );

    const usersEmpty =
        document.getElementById(
            "usersEmpty"
        );

    dashboardBody.innerHTML = "";
    usersBody.innerHTML = "";

    if (!users.length) {

        dashboardEmpty.style.display =
            "block";

        dashboardEmpty.textContent =
            "No users found.";

        usersEmpty.style.display =
            "block";

        usersEmpty.textContent =
            "No users found.";

        return;
    }

    dashboardEmpty.style.display =
        "none";

    usersEmpty.style.display =
        "none";


    users.forEach(user => {

        const email =
            user.email || "—";

        const name =
            user.name || "User";

        const tier =
            String(
                user.tier || "free"
            ).toLowerCase();

        const credits =
            user.credit_balance ??
            0;

        const premiumUntil =
            user.premium_expires_at ||
            "—";

        const admin =
            user.is_admin
                ? "Admin"
                : "User";


        /* DASHBOARD */

        const row =
            document.createElement("tr");

        row.innerHTML = `
            <td>
                <div class="user-email">
                    ${escapeHtml(email)}
                </div>
                <div class="user-name">
                    ${escapeHtml(name)}
                </div>
            </td>

            <td>
                ${
                    tier === "premium"
                    ? '<span class="badge badge-premium">PREMIUM</span>'
                    : '<span class="badge badge-free">FREE</span>'
                }
            </td>

            <td>
                ${credits}
            </td>

            <td>
                ${
                    user.is_admin
                    ? '<span class="badge badge-admin">ADMIN</span>'
                    : 'User'
                }
            </td>
        `;

        dashboardBody.appendChild(row);


        /* USERS TABLE */

        const fullRow =
            document.createElement("tr");

        fullRow.innerHTML = `
            <td>
                <div class="user-email">
                    ${escapeHtml(email)}
                </div>
            </td>

            <td>
                ${escapeHtml(name)}
            </td>

            <td>
                ${
                    tier === "premium"
                    ? '<span class="badge badge-premium">PREMIUM</span>'
                    : '<span class="badge badge-free">FREE</span>'
                }
            </td>

            <td>
                ${credits}
            </td>

            <td>
                ${escapeHtml(
                    String(premiumUntil)
                )}
            </td>

            <td>
                ${
                    user.is_admin
                    ? '<span class="badge badge-admin">ADMIN</span>'
                    : 'User'
                }
            </td>
        `;

        usersBody.appendChild(
            fullRow
        );

    });
}


/* ================= PREMIUM ================= */

async function loadPremium() {

    await loadUsers();

    const body =
        document.getElementById(
            "premiumTable"
        );

    body.innerHTML = "";

    const premiumUsers =
        usersCache.filter(
            u =>
                String(
                    u.tier || ""
                ).toLowerCase() ===
                "premium"
        );

    premiumUsers.forEach(user => {

        const row =
            document.createElement("tr");

        row.innerHTML = `
            <td>${escapeHtml(user.email || "")}</td>
            <td>${escapeHtml(user.name || "")}</td>
            <td>
                <span class="badge badge-premium">
                    PREMIUM
                </span>
            </td>
            <td>
                ${escapeHtml(
                    String(
                        user.premium_expires_at ||
                        "—"
                    )
                )}
            </td>
        `;

        body.appendChild(row);
    });
}


/* ================= CREDITS ================= */

async function loadCredits() {

    await loadUsers();

    const body =
        document.getElementById(
            "creditsTable"
        );

    body.innerHTML = "";

    usersCache.forEach(user => {

        const row =
            document.createElement("tr");

        row.innerHTML = `
            <td>
                ${escapeHtml(
                    user.email || ""
                )}
            </td>

            <td>
                <strong>
                    ${user.credit_balance ?? 0}
                </strong>
            </td>

            <td>
                ${
                    String(
                        user.tier || ""
                    ).toLowerCase() ===
                    "premium"
                    ? '<span class="badge badge-premium">PREMIUM</span>'
                    : '<span class="badge badge-free">FREE</span>'
                }
            </td>
        `;

        body.appendChild(row);
    });
}


/* ================= ANALYTICS ================= */

async function loadAnalytics() {

    await loadUsers();

    const total =
        usersCache.length;

    const premium =
        usersCache.filter(
            u =>
                String(
                    u.tier || ""
                ).toLowerCase() ===
                "premium"
        ).length;

    const credits =
        usersCache.reduce(
            (sum, u) =>
                sum +
                Number(
                    u.credit_balance || 0
                ),
            0
        );

    document.getElementById(
        "analyticsUsers"
    ).textContent = total;

    document.getElementById(
        "analyticsPremium"
    ).textContent = premium;

    document.getElementById(
        "analyticsCredits"
    ).textContent = credits;

    document.getElementById(
        "analyticsRequests"
    ).textContent = "—";

    document.getElementById(
        "analyticsInfo"
    ).textContent =
        "User and credit analytics loaded from Shadow AI database.";
}


/* ================= PREMIUM MODAL ================= */

function openPremiumModal() {

    document
        .getElementById(
            "premiumModal"
        )
        .classList.add("show");
}


function openCreditModal() {

    document
        .getElementById(
            "creditModal"
        )
        .classList.add("show");
}


function closeModals() {

    document
        .querySelectorAll(
            ".modal-bg"
        )
        .forEach(
            modal =>
                modal.classList.remove(
                    "show"
                )
        );
}


/* ================= GRANT PREMIUM ================= */

async function grantPremium() {

    const email =
        document.getElementById(
            "premiumEmail"
        ).value.trim();

    const months =
        Number(
            document.getElementById(
                "premiumMonths"
            ).value
        );

    if (!email) {

        showToast(
            "Enter a user email."
        );

        return;
    }

    try {

        /*
         * IMPORTANT:
         * This endpoint must exist in your
         * main.py.
         *
         * If your endpoint uses a different
         * name/body, tell me the endpoint code
         * and I will match it exactly.
         */

        await apiFetch(
            "/api/admin/set-premium",
            {
                method: "POST",
                body: JSON.stringify({
                    email: email,
                    months: months
                })
            }
        );

        closeModals();

        showToast(
            "Premium granted successfully."
        );

        await loadDashboard();

    } catch (error) {

        showToast(
            "Premium error: " +
            error.message
        );
    }
}


/* ================= ADD CREDITS ================= */

async function addCredits() {

    const email =
        document.getElementById(
            "creditEmail"
        ).value.trim();

    const amount =
        Number(
            document.getElementById(
                "creditAmount"
            ).value
        );

    const reason =
        document.getElementById(
            "creditReason"
        ).value.trim();


    if (!email) {

        showToast(
            "Enter a user email."
        );

        return;
    }

    if (!amount || amount <= 0) {

        showToast(
            "Enter a valid credit amount."
        );

        return;
    }


    try {

        /*
         * This endpoint name may already
         * exist in your backend.
         */

        await apiFetch(
            "/api/admin/add-credits",
            {
                method: "POST",
                body: JSON.stringify({
                    email: email,
                    amount: amount,
                    reason: reason
                })
            }
        );

        closeModals();

        showToast(
            "Credits added successfully."
        );

        await loadDashboard();

    } catch (error) {

        showToast(
            "Credit error: " +
            error.message
        );
    }
}


/* ================= LOGOUT ================= */

function logout() {

    localStorage.removeItem(
        "shadow_admin_key"
    );

    showToast(
        "Admin key removed."
    );

    setTimeout(
        () => location.reload(),
        700
    );
}


/* ================= CHANGE KEY ================= */

function changeAdminKey() {

    localStorage.removeItem(
        "shadow_admin_key"
    );

    const key =
        prompt(
            "Enter new Shadow AI Admin Key:"
        );

    if (key) {

        localStorage.setItem(
            "shadow_admin_key",
            key
        );

        showToast(
            "Admin key updated."
        );

        loadDashboard();
    }
}


/* ================= TOAST ================= */

function showToast(message) {

    const toast =
        document.getElementById(
            "toast"
        );

    toast.textContent =
        message;

    toast.classList.add(
        "show"
    );

    setTimeout(
        () =>
            toast.classList.remove(
                "show"
            ),
        3000
    );
}


/* ================= HTML ESCAPE ================= */

function escapeHtml(value) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


/* ================= START ================= */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        console.log(
            "Shadow AI Admin Console loaded."
        );

        loadDashboard();

    }
);

</script>

</body>
</html>
"""



@app.get("/health")
async def health():
    return {"status": "online", "provider": "openrouter"}
@app.get("/api/version")
async def get_version():
    return {"version": os.environ.get("APP_VERSION", "1.0.0")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
