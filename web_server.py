#!/usr/bin/env python3
"""
web_server.py - High-Performance Streaming Web Server for TunisiaLLM
Provides a modern chat interface with real-time SSE token streaming,
tool execution (safe calculator & Tunisian search), and hardware telemetry.
"""

import os
import sys
import json
import time
import zipfile
import argparse
import webbrowser
from typing import Optional, List, Dict, Generator, Tuple

import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from utils import load_config, format_time, format_bytes
import importlib
inf_mod = importlib.import_module("07_inference")
TunisiaLLMInference = inf_mod.TunisiaLLMInference


# ==============================================================================
# Checkpoint Resolution Helper
# ==============================================================================

def resolve_checkpoint(custom_path: Optional[str] = None) -> Tuple[str, str]:
    """
    Find best available checkpoint, auto-extracting tunisiallm_model.zip if needed.
    Returns (checkpoint_path, display_name).
    """
    if custom_path and os.path.isfile(custom_path):
        return custom_path, os.path.basename(custom_path)

    # 1. Check inside extracted tunisiallm_model directory
    extracted_sft = os.path.join(REPO_ROOT, "tunisiallm_model", "sft_checkpoint_best.pt")
    if os.path.isfile(extracted_sft) and os.path.getsize(extracted_sft) > 1000000:
        return extracted_sft, "tunisiallm_model (SFT Best)"

    # 2. Check in checkpoints directory
    ckpt_dir = os.path.join(REPO_ROOT, "checkpoints")
    candidates = [
        ("sft_checkpoint_best.pt", "SFT Checkpoint Best"),
        ("sft_checkpoint_final.pt", "SFT Checkpoint Final"),
        ("checkpoint_best.pt", "Pretrained Best"),
        ("checkpoint_latest.pt", "Pretrained Latest"),
    ]
    for filename, label in candidates:
        full_path = os.path.join(ckpt_dir, filename)
        if os.path.isfile(full_path) and os.path.getsize(full_path) > 1000000:
            return full_path, label

    # 3. Check for tunisiallm_model.zip
    zip_path = os.path.join(REPO_ROOT, "tunisiallm_model.zip")
    if os.path.isfile(zip_path):
        target_dir = os.path.join(REPO_ROOT, "tunisiallm_model")
        print(f"[*] Extracting {zip_path} into {target_dir}...")
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        if os.path.isfile(extracted_sft):
            return extracted_sft, "Extracted Zip (SFT Best)"

    # Default fallback
    return os.path.join(ckpt_dir, "sft_checkpoint_best.pt"), "Default (SFT Best)"


# ==============================================================================
# FastAPI Application & Streaming Endpoints
# ==============================================================================

app = FastAPI(title="TunisiaLLM Web Interface", version="1.0.0")
engine: Optional[TunisiaLLMInference] = None
checkpoint_info: Dict[str, str] = {}


class ChatRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = "إنت مساعد ذكي بالدارجة التونسية. جاوب بكل طلاقة ووضوح."
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_new_tokens: Optional[int] = 256
    repetition_penalty: Optional[float] = 1.15
    chat_history: Optional[List[Dict[str, str]]] = None


@app.get("/api/status")
async def get_status():
    """Return runtime hardware status, active model, and device information."""
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    return JSONResponse({
        "status": "online",
        "preset": "80M",
        "parameters": "78.12M",
        "checkpoint_name": checkpoint_info.get("name", "Active Model"),
        "checkpoint_path": checkpoint_info.get("path", ""),
        "device": str(engine.device) if engine else "cpu",
        "device_name": device_name,
        "cuda_available": torch.cuda.is_available()
    })


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Server-Sent Events (SSE) streaming chat endpoint with real-time token delivery."""
    if engine is None:
        return JSONResponse({"error": "Engine not initialized"}, status_code=500)

    # Build prompt context if history exists
    user_prompt = req.prompt.strip()

    def event_stream() -> Generator[str, None, None]:
        try:
            tokens_stream = engine.stream_generate(
                prompt=user_prompt,
                system_prompt=req.system_prompt or "إنت مساعد ذكي بالدارجة التونسية.",
                max_new_tokens=req.max_new_tokens or 256,
                temperature=req.temperature or 0.7,
                top_p=req.top_p or 0.9,
                repetition_penalty=req.repetition_penalty or 1.15
            )

            for token in tokens_stream:
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            err_data = json.dumps({"error": str(e)})
            yield f"data: {err_data}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Static Frontend Mounting
WEB_DIR = os.path.join(REPO_ROOT, "web")
os.makedirs(WEB_DIR, exist_ok=True)

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

app.mount("/", StaticFiles(directory=WEB_DIR), name="static")


# ==============================================================================
# Server Initialization & Launch
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="TunisiaLLM Web Chat Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on (default: 7860)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    global engine, checkpoint_info

    print("=" * 80)
    print(" TUNISIALLM CHAT INTERFACE & INFERENCE SERVER")
    print("=" * 80)

    cfg = load_config(os.path.join(REPO_ROOT, "config.yaml"))
    ckpt_path, ckpt_name = resolve_checkpoint(args.checkpoint)
    checkpoint_info = {"path": ckpt_path, "name": ckpt_name}

    # Ensure matching 16k tokenizer is selected
    tok_candidates = [
        os.path.join(os.path.dirname(ckpt_path), "tokenizer.json"),
        os.path.join(REPO_ROOT, "tunisiallm_model", "tokenizer.json"),
        os.path.join(REPO_ROOT, "tokenizer.json")
    ]
    for tc in tok_candidates:
        if os.path.isfile(tc) and os.path.getsize(tc) > 100000:
            cfg["paths"]["tokenizer_path"] = tc
            break

    print(f"[*] Checkpoint Selected: {ckpt_path} ({ckpt_name})")
    print(f"[*] Tokenizer Selected:  {cfg['paths']['tokenizer_path']}")
    print(f"[*] Target Device:       {args.device}")

    # Initialize Engine
    engine = TunisiaLLMInference(cfg, checkpoint_path=ckpt_path, device=args.device)

    import socket
    import urllib.request

    # Check if port is already running TunisiaLLM
    def is_port_in_use(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.4)
            return s.connect_ex((host, port)) == 0

    selected_port = args.port
    if is_port_in_use(args.host, selected_port):
        # Test if it's already TunisiaLLM
        try:
            with urllib.request.urlopen(f"http://{args.host}:{selected_port}/api/status", timeout=1.0) as resp:
                data = json.loads(resp.read().decode())
                if data.get("preset") == "80M":
                    print(f"\n[i] TunisiaLLM server is ALREADY active at http://{args.host}:{selected_port}!")
                    print(f"[i] Checkpoint in use: {data.get('checkpoint_name')}")
                    if not args.no_browser:
                        webbrowser.open(f"http://{args.host}:{selected_port}")
                    print("[i] Opened in browser. You can talk to the model now!\n")
                    return
        except Exception:
            pass

        # If it's a different application, find next free port
        for p in range(args.port + 1, args.port + 15):
            if not is_port_in_use(args.host, p):
                print(f"[!] Port {args.port} is busy. Automatically switching to free port: {p}")
                selected_port = p
                break

    url = f"http://{args.host}:{selected_port}"
    print("\n" + "=" * 80)
    print(f"[OK] TunisiaLLM UI ready at: {url}")
    print("=" * 80 + "\n")

    if not args.no_browser:
        def open_browser():
            time.sleep(1.2)
            webbrowser.open(url)
        import threading
        threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=selected_port, log_level="warning")


if __name__ == "__main__":
    main()
