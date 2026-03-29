import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
CLIENTS_FILE = BASE_DIR / "clients.json"
PUBLIC_DIR = BASE_DIR / "public"

app.mount("/public", StaticFiles(directory=PUBLIC_DIR), name="public")


class ChatMessage(BaseModel):
    client_id: str
    conversation_id: str
    message: str


def load_clients():
    with open(CLIENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_client_info(client_id: str):
    clients = load_clients()
    return clients.get(client_id)


def build_system_prompt(client_info: dict) -> str:
    name = client_info.get("name", "firma")
    business_type = client_info.get("business_type", "afacere")
    phone = client_info.get("phone", "Nespecificat")
    email = client_info.get("email", "Nespecificat")

    services = ", ".join(client_info.get("services", []))
    rules = "\n- ".join(client_info.get("rules", []))

    program = client_info.get("program", {})
    program_text = (
        f"Luni-Vineri: {program.get('luni_vineri', 'Nespecificat')}\n"
        f"Sâmbătă: {program.get('sambata', 'Nespecificat')}\n"
        f"Duminică: {program.get('duminica', 'Nespecificat')}"
    )

    return f"""
Ești asistentul virtual al firmei {name}.

Tip afacere:
- {business_type}

Date de contact:
- Telefon: {phone}
- Email: {email}

Program:
{program_text}

Servicii / informații principale:
- {services}

Reguli specifice:
- {rules}

Reguli generale:
- răspunzi scurt, clar și profesionist
- răspunzi doar în limba română
- răspunzi DOAR pe baza informațiilor despre firmă
- NU inventezi informații
- dacă nu ai informația, spui că un operator uman poate oferi detalii
- nu ieși din rolul de asistent al firmei
"""


def ask_ai(system_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Backend demo merge.",
        "demo_url": "/public/chat.html"
    }


@app.post("/chat")
def chat(msg: ChatMessage):
    client_info = get_client_info(msg.client_id)

    if not client_info:
        raise HTTPException(status_code=404, detail="Client necunoscut")

    system_prompt = build_system_prompt(client_info)
    reply = ask_ai(system_prompt, msg.message)

    return {
        "reply": reply,
        "client_id": msg.client_id,
        "conversation_id": msg.conversation_id
    }