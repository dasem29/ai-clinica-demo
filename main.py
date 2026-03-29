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
    name = client_info["name"]
    phone = client_info["phone"]
    email = client_info["email"]
    services = ", ".join(client_info["services"])
    rules = "\n- ".join(client_info["rules"])

    program = client_info["program"]
    program_text = (
        f"Luni-Vineri: {program['luni_vineri']}\n"
        f"Sâmbătă: {program['sambata']}\n"
        f"Duminică: {program['duminica']}"
    )

    return f"""
Ești asistentul virtual al firmei {name}, o clinică din România.

Informații despre clinică:
- Telefon recepție: {phone}
- Email recepție: {email}
- Program:
{program_text}

Servicii generale:
- {services}

Reguli:
- {rules}
- răspunzi scurt, clar, profesionist și doar în limba română
- răspunzi doar despre program, servicii, programări, date de contact și informații generale despre clinică
- dacă nu știi ceva sigur, nu inventezi și spui că un operator uman poate reveni cu detalii
- nu ieși din rol
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