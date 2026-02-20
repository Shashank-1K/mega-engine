"""
GROQ MEGA ENGINE - Core AI Engine
8 Keys × 20 Models = AI Powerhouse
"""

import os
import re
import json
import time
import base64
import random
import threading
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from groq import Groq


# ═══════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    "Text Chat - Fast": {
        "llama-3.1-8b-instant": {"desc": "Llama 3.1 8B - Ultra fast", "speed": "⚡⚡⚡", "quality": "⭐⭐"},
        "allam-2-7b": {"desc": "ALLaM 2 7B - Arabic/English", "speed": "⚡⚡⚡", "quality": "⭐⭐"},
    },
    "Text Chat - Powerful": {
        "llama-3.3-70b-versatile": {"desc": "Llama 3.3 70B - Best overall", "speed": "⚡⚡", "quality": "⭐⭐⭐⭐"},
        "qwen/qwen3-32b": {"desc": "Qwen3 32B - Strong reasoning", "speed": "⚡⚡", "quality": "⭐⭐⭐⭐"},
        "openai/gpt-oss-120b": {"desc": "GPT-OSS 120B - Largest", "speed": "⚡", "quality": "⭐⭐⭐⭐⭐"},
        "openai/gpt-oss-20b": {"desc": "GPT-OSS 20B - Balanced", "speed": "⚡⚡", "quality": "⭐⭐⭐"},
        "moonshotai/kimi-k2-instruct": {"desc": "Kimi K2 - Strong coder", "speed": "⚡⚡", "quality": "⭐⭐⭐⭐"},
        "moonshotai/kimi-k2-instruct-0905": {"desc": "Kimi K2 0905", "speed": "⚡⚡", "quality": "⭐⭐⭐⭐"},
    },
    "Compound (Agentic)": {
        "groq/compound": {"desc": "Compound - Agentic AI", "speed": "⚡", "quality": "⭐⭐⭐⭐"},
        "groq/compound-mini": {"desc": "Compound Mini - Fast agent", "speed": "⚡⚡", "quality": "⭐⭐⭐"},
    },
    "Vision (Multimodal)": {
        "meta-llama/llama-4-scout-17b-16e-instruct": {"desc": "Llama 4 Scout - Vision", "speed": "⚡⚡", "quality": "⭐⭐⭐"},
        "meta-llama/llama-4-maverick-17b-128e-instruct": {"desc": "Llama 4 Maverick - Best vision", "speed": "⚡", "quality": "⭐⭐⭐⭐⭐"},
    },
    "Safety / Guard": {
        "meta-llama/llama-prompt-guard-2-22m": {"desc": "Prompt Guard 22M", "speed": "⚡⚡⚡", "quality": "⭐⭐"},
        "meta-llama/llama-prompt-guard-2-86m": {"desc": "Prompt Guard 86M", "speed": "⚡⚡⚡", "quality": "⭐⭐⭐"},
        "meta-llama/llama-guard-4-12b": {"desc": "Llama Guard 4 12B", "speed": "⚡⚡", "quality": "⭐⭐⭐⭐"},
        "openai/gpt-oss-safeguard-20b": {"desc": "GPT Safeguard 20B", "speed": "⚡⚡", "quality": "⭐⭐⭐⭐"},
    },
    "Audio STT (Whisper)": {
        "whisper-large-v3": {"desc": "Whisper V3 - Best accuracy", "speed": "⚡⚡", "quality": "⭐⭐⭐⭐⭐"},
        "whisper-large-v3-turbo": {"desc": "Whisper V3 Turbo - Faster", "speed": "⚡⚡⚡", "quality": "⭐⭐⭐⭐"},
    },
    "Audio TTS (Orpheus)": {
        "canopylabs/orpheus-v1-english": {"desc": "Orpheus English TTS", "speed": "⚡⚡", "quality": "⭐⭐⭐⭐"},
        "canopylabs/orpheus-arabic-saudi": {"desc": "Orpheus Arabic TTS", "speed": "⚡⚡", "quality": "⭐⭐⭐⭐"},
    },
}

TTS_VOICES = {
    "canopylabs/orpheus-v1-english": {
        "voices": ["autumn", "diana", "hannah", "austin", "daniel", "troy"],
        "default": "diana",
    },
    "canopylabs/orpheus-arabic-saudi": {
        "voices": ["fahad", "sultan", "noura", "lulwa", "aisha"],
        "default": "noura",
    },
}

ALL_CHAT_MODELS = []
for cat in ["Text Chat - Fast", "Text Chat - Powerful", "Compound (Agentic)", "Vision (Multimodal)"]:
    ALL_CHAT_MODELS.extend(MODEL_REGISTRY.get(cat, {}).keys())


def get_all_models_flat():
    models = []
    for cat_models in MODEL_REGISTRY.values():
        models.extend(cat_models.keys())
    return models


# ═══════════════════════════════════════════════════════════════════
# KEY ROUTER (Load Balancer)
# ═══════════════════════════════════════════════════════════════════
class KeyRouter:
    def __init__(self, keys: list):
        self.keys = keys
        self.clients = {}
        self.lock = threading.Lock()
        self.call_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.rate_limit_until = defaultdict(float)
        self.total_tokens = defaultdict(int)
        self.last_used = defaultdict(float)
        self.response_times = defaultdict(list)
        self._index = 0

        for i, key in enumerate(keys):
            label = f"Key_{i+1}"
            try:
                self.clients[label] = Groq(api_key=key)
            except Exception:
                self.clients[label] = None

    def get_client(self, strategy="round_robin"):
        with self.lock:
            now = time.time()
            n = len(self.keys)
            available = []

            for i in range(n):
                label = f"Key_{i+1}"
                if self.clients.get(label) and self.rate_limit_until[label] <= now:
                    available.append((label, self.clients[label], i))

            if not available:
                min_wait = min(
                    (self.rate_limit_until[f"Key_{i+1}"] - now for i in range(n)),
                    default=1
                )
                if min_wait > 0:
                    time.sleep(min(min_wait + 0.1, 5))
                label = f"Key_1"
                self.call_counts[label] += 1
                return label, self.clients.get(label)

            if strategy == "round_robin":
                for attempt in range(n):
                    idx = (self._index + attempt) % n
                    label = f"Key_{idx+1}"
                    if any(a[0] == label for a in available):
                        self._index = (idx + 1) % n
                        self.call_counts[label] += 1
                        self.last_used[label] = now
                        return label, self.clients[label]
                label, client, _ = available[0]
            elif strategy == "least_used":
                available.sort(key=lambda x: self.call_counts[x[0]])
                label, client, _ = available[0]
            elif strategy == "fastest":
                def avg_time(lbl):
                    times = self.response_times[lbl]
                    return sum(times[-10:]) / len(times[-10:]) if times else 999999
                available.sort(key=lambda x: avg_time(x[0]))
                label, client, _ = available[0]
            else:
                label, client, _ = random.choice(available)

            self.call_counts[label] += 1
            self.last_used[label] = now
            return label, client

    def report_error(self, label, error_str):
        with self.lock:
            self.error_counts[label] += 1
            if "429" in error_str or "rate_limit" in error_str.lower():
                self.rate_limit_until[label] = time.time() + 15

    def report_success(self, label, tokens=0, response_time_ms=0):
        with self.lock:
            self.total_tokens[label] += tokens
            if response_time_ms > 0:
                self.response_times[label].append(response_time_ms)
                if len(self.response_times[label]) > 100:
                    self.response_times[label] = self.response_times[label][-50:]

    def get_stats(self):
        stats = {}
        for i, key in enumerate(self.keys):
            label = f"Key_{i+1}"
            times = self.response_times[label]
            avg_ms = sum(times[-20:]) / len(times[-20:]) if times else 0
            stats[label] = {
                "calls": self.call_counts[label],
                "errors": self.error_counts[label],
                "tokens": self.total_tokens[label],
                "avg_response_ms": round(avg_ms),
                "masked_key": key[:8] + "••••" + key[-4:],
                "is_available": self.rate_limit_until[label] <= time.time(),
            }
        return stats

    def validate_keys(self):
        results = {}
        for i, key in enumerate(self.keys):
            label = f"Key_{i+1}"
            try:
                if self.clients.get(label):
                    models = self.clients[label].models.list()
                    results[label] = {
                        "valid": True,
                        "models": len(models.data),
                        "masked": key[:8] + "••••" + key[-4:],
                    }
                else:
                    results[label] = {"valid": False, "error": "Client init failed"}
            except Exception as e:
                results[label] = {"valid": False, "error": str(e)[:100]}
        return results


# ═══════════════════════════════════════════════════════════════════
# MEGA ENGINE
# ═══════════════════════════════════════════════════════════════════
class GroqMegaEngine:
    def __init__(self, router: KeyRouter):
        self.router = router
        self.call_log = []
        self.stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "errors": 0,
            "start_time": datetime.now().isoformat(),
        }

    def _clean_response(self, text):
        if not text:
            return ""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def _log_call(self, method, model, key, success, tokens, time_ms, error=None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "model": model,
            "key": key,
            "success": success,
            "tokens": tokens,
            "time_ms": time_ms,
            "error": error,
        }
        self.call_log.append(entry)
        if len(self.call_log) > 1000:
            self.call_log = self.call_log[-500:]

    # ─── CHAT ───
    def chat(self, prompt, model="llama-3.3-70b-versatile",
             system="You are a helpful AI assistant.",
             max_tokens=2048, temperature=0.7,
             max_retries=3, strategy="round_robin"):

        if "qwen3" in model.lower():
            system += " /no_think"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(max_retries):
            label, client = self.router.get_client(strategy)
            if not client:
                continue
            try:
                t0 = time.time()
                resp = client.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                )
                elapsed = round((time.time() - t0) * 1000)
                reply = self._clean_response(resp.choices[0].message.content)
                tokens = resp.usage.total_tokens if resp.usage else 0

                self.router.report_success(label, tokens, elapsed)
                self.stats["total_calls"] += 1
                self.stats["total_tokens"] += tokens
                self._log_call("chat", model, label, True, tokens, elapsed)

                return {
                    "success": True, "response": reply,
                    "model": model, "key": label,
                    "tokens": tokens, "time_ms": elapsed,
                    "attempt": attempt + 1,
                }
            except Exception as e:
                self.router.report_error(label, str(e))
                self.stats["errors"] += 1
                self._log_call("chat", model, label, False, 0, 0, str(e)[:200])
                if attempt == max_retries - 1:
                    return {"success": False, "error": str(e)[:300],
                            "model": model, "key": label}
                time.sleep(1)

    # ─── MULTI-TURN CHAT ───
    def chat_multi(self, messages, model="llama-3.3-70b-versatile",
                   max_tokens=2048, temperature=0.7, max_retries=3):

        if "qwen3" in model.lower():
            if messages and messages[0]["role"] == "system":
                if "/no_think" not in messages[0]["content"]:
                    messages[0]["content"] += " /no_think"

        for attempt in range(max_retries):
            label, client = self.router.get_client()
            if not client:
                continue
            try:
                t0 = time.time()
                resp = client.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                )
                elapsed = round((time.time() - t0) * 1000)
                reply = self._clean_response(resp.choices[0].message.content)
                tokens = resp.usage.total_tokens if resp.usage else 0

                self.router.report_success(label, tokens, elapsed)
                self.stats["total_calls"] += 1
                self.stats["total_tokens"] += tokens

                return {
                    "success": True, "response": reply,
                    "model": model, "key": label,
                    "tokens": tokens, "time_ms": elapsed,
                }
            except Exception as e:
                self.router.report_error(label, str(e))
                self.stats["errors"] += 1
                if attempt == max_retries - 1:
                    return {"success": False, "error": str(e)[:300]}
                time.sleep(1)

    # ─── CONSENSUS ───
    def consensus(self, prompt, models=None,
                  system="You are a helpful AI assistant. Be precise.",
                  max_tokens=1024, judge_model="llama-3.3-70b-versatile"):

        if models is None:
            models = [
                "llama-3.3-70b-versatile",
                "qwen/qwen3-32b",
                "openai/gpt-oss-120b",
                "moonshotai/kimi-k2-instruct",
            ]

        responses = {}

        def ask(m):
            return m, self.chat(prompt, model=m, system=system, max_tokens=max_tokens)

        with ThreadPoolExecutor(max_workers=min(len(models), 8)) as ex:
            futures = {ex.submit(ask, m): m for m in models}
            for f in as_completed(futures):
                m, r = f.result()
                if r["success"]:
                    responses[m] = r

        if not responses:
            return {"success": False, "error": "All models failed"}

        if len(responses) == 1:
            r = list(responses.values())[0]
            return {"success": True, "response": r["response"],
                    "method": "single", "individual": responses}

        judge_prompt = f'User asked: "{prompt}"\n\nAnswers:\n\n'
        for i, (m, r) in enumerate(responses.items(), 1):
            short = m.split("/")[-1] if "/" in m else m
            judge_prompt += f"--- Answer {i} ({short}) ---\n{r['response']}\n\n"
        judge_prompt += (
            "Pick the BEST answer. Synthesize the best parts into one "
            "superior response. Reply with the combined best answer only."
        )

        judge = self.chat(judge_prompt, model=judge_model, max_tokens=2000, temperature=0.0)

        return {
            "success": True,
            "response": judge.get("response", ""),
            "method": "consensus",
            "individual": {m: r["response"] for m, r in responses.items()},
            "judge_model": judge_model,
        }

    # ─── BATCH ───
    def batch_process(self, prompts, model="llama-3.1-8b-instant",
                      system="You are a helpful assistant.",
                      max_workers=8, progress_callback=None):

        results = [None] * len(prompts)
        completed = [0]
        lock = threading.Lock()

        def process(idx_prompt):
            idx, prompt = idx_prompt
            r = self.chat(prompt, model=model, system=system)
            r["index"] = idx
            with lock:
                completed[0] += 1
                if progress_callback:
                    progress_callback(completed[0], len(prompts))
            return idx, r

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(process, (i, p)): i for i, p in enumerate(prompts)}
            for f in as_completed(futures):
                idx, r = f.result()
                results[idx] = r

        return results

    # ─── CHAIN OF THOUGHT ───
    def chain_of_thought(self, task, models=None, progress_callback=None):

        if models is None:
            models = [
                "llama-3.3-70b-versatile",
                "qwen/qwen3-32b",
                "openai/gpt-oss-120b",
                "moonshotai/kimi-k2-instruct",
            ]

        # Plan steps
        plan = self.chat(
            f'Break this task into 3-5 steps:\nTask: {task}\n\n'
            f'Reply as JSON: {{"steps": ["step1", "step2", ...]}}',
            model="llama-3.3-70b-versatile", max_tokens=500, temperature=0.0
        )

        steps = [task]
        if plan["success"]:
            try:
                match = re.search(r'\{[\s\S]*\}', plan["response"])
                if match:
                    steps = json.loads(match.group()).get("steps", [task])
            except Exception:
                pass

        chain = []
        context = f"Task: {task}\n\n"

        for i, step in enumerate(steps):
            m = models[i % len(models)]
            r = self.chat(
                f"{context}\nCurrent step ({i+1}/{len(steps)}): {step}\n"
                f"Complete this step. Build on previous results.",
                model=m, max_tokens=1500
            )
            if r["success"]:
                context += f"\n--- Step {i+1} ---\n{r['response']}\n"
                chain.append({"step": step, "model": m, "response": r["response"],
                              "tokens": r.get("tokens", 0), "time_ms": r.get("time_ms", 0)})
            else:
                chain.append({"step": step, "model": m, "error": r.get("error", "")})

            if progress_callback:
                progress_callback(i + 1, len(steps) + 1)

        # Synthesize
        final = self.chat(
            f"{context}\nSynthesize ALL step results into one complete answer for: {task}",
            model="llama-3.3-70b-versatile", max_tokens=2000
        )
        if progress_callback:
            progress_callback(len(steps) + 1, len(steps) + 1)

        return {
            "success": True, "task": task, "steps": chain,
            "final_answer": final.get("response", ""),
            "total_tokens": sum(s.get("tokens", 0) for s in chain),
        }

    # ─── SAFE CHAT ───
    def safe_chat(self, prompt, model="llama-3.3-70b-versatile",
                  system="You are a helpful assistant."):

        # Check input
        label1, client1 = self.router.get_client()
        input_safety = "unknown"
        try:
            if client1:
                g = client1.chat.completions.create(
                    model="meta-llama/llama-guard-4-12b",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50, temperature=0.0,
                )
                input_safety = g.choices[0].message.content.strip().lower()
        except Exception:
            pass

        if "unsafe" in input_safety:
            return {"success": False, "blocked": True,
                    "reason": "Input flagged unsafe", "safety": input_safety}

        result = self.chat(prompt, model=model, system=system)
        if not result["success"]:
            return result

        # Check output
        label2, client2 = self.router.get_client()
        output_safety = "unknown"
        try:
            if client2:
                g2 = client2.chat.completions.create(
                    model="meta-llama/llama-guard-4-12b",
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": result["response"]},
                    ],
                    max_tokens=50, temperature=0.0,
                )
                output_safety = g2.choices[0].message.content.strip().lower()
        except Exception:
            pass

        result["input_safety"] = input_safety
        result["output_safety"] = output_safety
        result["safety_checked"] = True
        return result

    # ─── VISION ───
    def see(self, image_bytes, prompt="Describe this image in detail.",
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            max_tokens=1024):

        img_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        label, client = self.router.get_client()
        if not client:
            return {"success": False, "error": "No client available"}
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"}}
                    ]
                }],
                max_tokens=max_tokens, temperature=0.3,
            )
            elapsed = round((time.time() - t0) * 1000)
            reply = resp.choices[0].message.content.strip()
            tokens = resp.usage.total_tokens if resp.usage else 0
            self.router.report_success(label, tokens, elapsed)
            self.stats["total_calls"] += 1
            self.stats["total_tokens"] += tokens
            return {"success": True, "response": reply, "model": model,
                    "key": label, "tokens": tokens, "time_ms": elapsed}
        except Exception as e:
            self.router.report_error(label, str(e))
            return {"success": False, "error": str(e)[:300]}

    # ─── STT ───
    def listen(self, audio_bytes, filename="audio.mp3",
               model="whisper-large-v3-turbo"):

        label, client = self.router.get_client()
        if not client:
            return {"success": False, "error": "No client available"}
        try:
            t0 = time.time()
            tx = client.audio.transcriptions.create(
                model=model,
                file=(filename, audio_bytes),
                language="en",
                response_format="verbose_json",
            )
            elapsed = round((time.time() - t0) * 1000)
            result = {"success": True, "text": tx.text, "model": model,
                      "key": label, "time_ms": elapsed}
            if hasattr(tx, 'duration'):
                result["duration"] = tx.duration
            if hasattr(tx, 'segments') and tx.segments:
                segs = []
                for s in tx.segments:
                    if isinstance(s, dict):
                        segs.append(s)
                    else:
                        segs.append({"start": s.start, "end": s.end, "text": s.text})
                result["segments"] = segs
            self.router.report_success(label, 0, elapsed)
            return result
        except Exception as e:
            self.router.report_error(label, str(e))
            return {"success": False, "error": str(e)[:300]}

    # ─── TTS ───
    def speak(self, text, voice="diana",
              model="canopylabs/orpheus-v1-english"):

        valid = TTS_VOICES.get(model, {}).get("voices", ["diana"])
        if voice not in valid:
            voice = TTS_VOICES.get(model, {}).get("default", valid[0])

        label, client = self.router.get_client()
        if not client:
            return {"success": False, "error": "No client available"}
        try:
            t0 = time.time()
            resp = client.audio.speech.create(
                model=model, input=text, voice=voice, response_format="wav",
            )
            audio = resp.read() if hasattr(resp, 'read') else (
                resp.content if hasattr(resp, 'content') else bytes(resp))
            elapsed = round((time.time() - t0) * 1000)
            self.router.report_success(label, 0, elapsed)
            return {"success": True, "audio_bytes": audio,
                    "size_kb": round(len(audio) / 1024, 1),
                    "voice": voice, "model": model, "key": label,
                    "time_ms": elapsed}
        except Exception as e:
            self.router.report_error(label, str(e))
            return {"success": False, "error": str(e)[:300]}

    # ─── DEBATE ───
    def debate(self, topic, rounds=2, models=None, progress_callback=None):

        if models is None:
            models = ["llama-3.3-70b-versatile", "qwen/qwen3-32b", "openai/gpt-oss-120b"]

        log = []
        ctx = f"Debate topic: {topic}\n\n"
        total_steps = rounds * len(models) + 1
        step = 0

        for rnd in range(rounds):
            for m in models:
                short = m.split("/")[-1] if "/" in m else m
                r = self.chat(
                    f"{ctx}\nYou are debating. Round {rnd+1}. Take a clear position. "
                    f"Respond to previous speakers. Be concise (3-5 sentences).",
                    model=m, max_tokens=500, temperature=0.8
                )
                if r["success"]:
                    log.append({"round": rnd+1, "model": m, "short": short,
                                "argument": r["response"], "time_ms": r.get("time_ms", 0)})
                    ctx += f"\n[{short}, R{rnd+1}]: {r['response']}\n"
                step += 1
                if progress_callback:
                    progress_callback(step, total_steps)

        verdict = self.chat(
            f"{ctx}\nAs judge, summarize the debate. Key points from each side. "
            f"Strongest arguments. Your verdict.",
            model="llama-3.3-70b-versatile", max_tokens=1000, temperature=0.0
        )
        if progress_callback:
            progress_callback(total_steps, total_steps)

        return {"success": True, "topic": topic, "log": log,
                "verdict": verdict.get("response", "")}

    # ─── COMPARE MODELS ───
    def compare_models(self, prompt, models=None, progress_callback=None):

        if models is None:
            models = ALL_CHAT_MODELS[:6]

        results = {}

        def ask(m):
            return m, self.chat(prompt, model=m)

        with ThreadPoolExecutor(max_workers=min(len(models), 8)) as ex:
            futures = {ex.submit(ask, m): m for m in models}
            done = 0
            for f in as_completed(futures):
                m, r = f.result()
                results[m] = r
                done += 1
                if progress_callback:
                    progress_callback(done, len(models))

        return results

    # ─── JSON MODE ───
    def json_chat(self, prompt, model="llama-3.3-70b-versatile",
                  system="Respond with valid JSON only.", max_tokens=1024):

        if "qwen3" in model.lower():
            system += " /no_think"
            prompt += " /no_think"

        label, client = self.router.get_client()
        if not client:
            return {"success": False, "error": "No client available"}
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens, temperature=0.0,
                response_format={"type": "json_object"},
            )
            elapsed = round((time.time() - t0) * 1000)
            raw = resp.choices[0].message.content
            tokens = resp.usage.total_tokens if resp.usage else 0
            self.router.report_success(label, tokens, elapsed)

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None

            return {"success": True, "response": raw, "parsed": parsed,
                    "model": model, "key": label, "tokens": tokens, "time_ms": elapsed}
        except Exception as e:
            self.router.report_error(label, str(e))
            return {"success": False, "error": str(e)[:300]}

    def get_stats(self):
        return {
            "engine": self.stats,
            "keys": self.router.get_stats(),
            "call_log": self.call_log[-50:],
        }