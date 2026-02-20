"""
🚀 GROQ MEGA ENGINE - Streamlit UI (SECURED)
Production-ready with encrypted admin access
"""

import streamlit as st
import json
import time
import os
import hashlib
import hmac
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from groq import Groq

from engine import (
    KeyRouter, GroqMegaEngine, MODEL_REGISTRY, TTS_VOICES,
    ALL_CHAT_MODELS, get_all_models_flat
)

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🚀 Groq Mega Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
    }
    .mega-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        padding: 10px;
    }
    .sub-header {
        text-align: center;
        color: #8888aa;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    .user-msg {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .bot-msg {
        background: #1e1e3f;
        border: 1px solid #3a3a6a;
        color: #e0e0ff;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
    }
    .msg-meta {
        font-size: 0.7rem;
        color: #6666aa;
        margin-top: 4px;
    }
    .lock-container {
        text-align: center;
        padding: 40px;
        background: linear-gradient(135deg, #1a1a3a, #2a2a5a);
        border: 2px solid #3a3a6a;
        border-radius: 20px;
        margin: 20px auto;
        max-width: 500px;
    }
    .lock-icon {
        font-size: 4rem;
        margin-bottom: 15px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-active {
        background: #00ff8822;
        color: #00ff88;
        border: 1px solid #00ff8844;
    }
    .badge-locked {
        background: #ff444422;
        color: #ff4444;
        border: 1px solid #ff444444;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 🔒 SECURITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def get_api_keys():
    """Load API keys from Streamlit secrets (never exposed in UI)."""
    try:
        return list(st.secrets["groq"]["keys"])
    except Exception:
        return []


def get_admin_password():
    """Get admin password from secrets."""
    try:
        return st.secrets["groq"]["admin_password"]
    except Exception:
        return "admin123"  # Fallback for local dev
    
def get_user_password():
    """Get user password from secrets."""
    try:
        return st.secrets["groq"]["user_password"]
    except Exception:
        return "user123"  # Fallback for local dev


def verify_user_password(input_password: str) -> bool:
    """Verify user password securely."""
    correct = get_user_password()
    return hmac.compare_digest(
        hash_password(input_password),
        hash_password(correct)
    )


def hash_password(password: str) -> str:
    """Hash password with SHA-256 + salt."""
    salt = "groq_mega_engine_2025"
    return hashlib.sha256(f"{salt}{password}{salt}".encode()).hexdigest()


def verify_password(input_password: str) -> bool:
    """Verify admin password securely."""
    correct = get_admin_password()
    # Use hmac.compare_digest to prevent timing attacks
    return hmac.compare_digest(
        hash_password(input_password),
        hash_password(correct)
    )


def mask_key(key: str) -> str:
    """Mask API key for display: gsk_zUn6••••••••8rRf"""
    if len(key) < 12:
        return "••••••••"
    return key[:8] + "••••••••" + key[-4:]


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════
def init_session():
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "router" not in st.session_state:
        st.session_state.router = None
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = []
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "admin_unlocked" not in st.session_state:
        st.session_state.admin_unlocked = False
    if "admin_unlock_time" not in st.session_state:
        st.session_state.admin_unlock_time = None
    if "failed_attempts" not in st.session_state:
        st.session_state.failed_attempts = 0
    if "lockout_until" not in st.session_state:
        st.session_state.lockout_until = 0
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "strategy" not in st.session_state:
        st.session_state.strategy = "round_robin"

    if "user_authenticated" not in st.session_state:
        st.session_state.user_authenticated = False
    if "user_failed_attempts" not in st.session_state:
        st.session_state.user_failed_attempts = 0
    if "user_lockout_until" not in st.session_state:
        st.session_state.user_lockout_until = 0

init_session()


# ═══════════════════════════════════════════════════════════════════
# AUTO-INITIALIZE ENGINE (from secrets, no UI exposure)
# ═══════════════════════════════════════════════════════════════════
def auto_initialize():
    """Auto-initialize engine from secrets on first load."""
    if not st.session_state.initialized:
        keys = get_api_keys()
        if keys:
            st.session_state.api_keys = keys
            st.session_state.router = KeyRouter(keys)
            st.session_state.engine = GroqMegaEngine(st.session_state.router)
            st.session_state.initialized = True
            return True
        return False
    return True

engine_ready = auto_initialize()



# ═══════════════════════════════════════════════════════════════════
# 🔐 APP-LEVEL USER AUTHENTICATION GATE
# ═══════════════════════════════════════════════════════════════════
if not st.session_state.user_authenticated:
    # Show ONLY the login screen, nothing else
    st.markdown("")
    st.markdown("")

    # Centered login container
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 5rem; margin-bottom: 10px;">🚀</div>
            <h1 style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 2.5rem; font-weight: 900;">
                GROQ MEGA ENGINE
            </h1>
            <p style="color: #6666aa; font-size: 1.1rem; margin-bottom: 30px;">
                Multi-Key AI Powerhouse
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a3a, #2a2a5a);
                    border: 2px solid #3a3a6a; border-radius: 20px;
                    padding: 30px; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 10px;">🔐</div>
            <h3 style="color: #aaaacc; margin-bottom: 5px;">Authentication Required</h3>
            <p style="color: #6666aa; font-size: 0.9rem;">
                Enter your password to access the application
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Check lockout
        now = time.time()
        if st.session_state.user_lockout_until > now:
            remaining = int(st.session_state.user_lockout_until - now)
            st.error(f"🔒 Too many failed attempts. Try again in {remaining} seconds.")
            st.stop()

        with st.form("user_login_form"):
            user_password_input = st.text_input(
                "🔑 Password:",
                type="password",
                placeholder="Enter access password...",
            )
            login_submitted = st.form_submit_button(
                "🔓 Enter", use_container_width=True
            )

            if login_submitted:
                if not user_password_input:
                    st.warning("Please enter a password.")
                elif verify_user_password(user_password_input):
                    st.session_state.user_authenticated = True
                    st.session_state.user_failed_attempts = 0
                    st.success("✅ Welcome! Loading engine...")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.session_state.user_failed_attempts += 1
                    remaining_attempts = 5 - st.session_state.user_failed_attempts

                    if st.session_state.user_failed_attempts >= 5:
                        st.session_state.user_lockout_until = time.time() + 300
                        st.session_state.user_failed_attempts = 0
                        st.error("🔒 Too many attempts! Locked for 5 minutes.")
                    else:
                        st.error(
                            f"❌ Wrong password. {remaining_attempts} attempts remaining."
                        )

        st.markdown("""
        <div style="text-align: center; color: #444; font-size: 0.75rem; margin-top: 30px;">
            🚀 Groq Mega Engine v2.0 | Secured
        </div>
        """, unsafe_allow_html=True)

    st.stop()  # ← THIS IS CRITICAL: stops rendering everything below


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR (Clean - no keys shown)
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚀 Groq Mega Engine")

    if engine_ready:
        num_keys = len(st.session_state.api_keys)
        num_models = len(get_all_models_flat())

        st.markdown(
            f'<span class="status-badge badge-active">● ONLINE</span>',
            unsafe_allow_html=True
        )
        # st.markdown(f"""
        # - 🔑 **{num_keys}** API Keys loaded
        # - 🤖 **{num_models}** Models available
        # - ⚡ **{num_keys * num_models}** Total workers
        # """)

        st.markdown(f"""
        - 🤖 **{num_models}** Models available
        - ⚡ **{num_keys * num_models}** Total workers
        """)

        # Strategy selector
        strategy = st.selectbox(
            "⚖️ Load Balancing",
            ["round_robin", "least_used", "fastest", "random"],
            index=0,
            help="How requests are distributed across keys"
        )
        st.session_state.strategy = strategy

        # Quick stats (no sensitive info)
        if st.session_state.engine:
            s = st.session_state.engine.stats
            st.markdown("---")
            st.markdown("### 📊 Session Stats")
            st.metric("Calls", s["total_calls"])
            st.metric("Tokens", f"{s['total_tokens']:,}")
            if s["total_calls"] > 0:
                rate = ((s['total_calls'] - s['errors']) / s['total_calls']) * 100
                st.metric("Success", f"{rate:.0f}%")
    else:
        st.markdown(
            f'<span class="status-badge badge-locked">● OFFLINE</span>',
            unsafe_allow_html=True
        )
        st.error("⚠️ No API keys configured!")
        st.markdown("""
        **Setup Instructions:**
        1. Create `.streamlit/secrets.toml`
        2. Add your Groq API keys
        3. Restart the app
        
        Or on **Streamlit Cloud**:
        Settings → Secrets
        """)

    # Admin lock/unlock
    st.markdown("---")
    st.markdown("### 🔒 Admin Panel")

    if st.session_state.admin_unlocked:
        unlock_time = st.session_state.admin_unlock_time
        if unlock_time:
            elapsed = (datetime.now() - unlock_time).seconds
            remaining = max(0, 1800 - elapsed)  # 30 min session
            if remaining <= 0:
                st.session_state.admin_unlocked = False
                st.warning("Session expired. Please re-authenticate.")
                st.rerun()
            else:
                mins = remaining // 60
                st.success(f"🔓 Unlocked ({mins}m remaining)")

        if st.button("🔒 Lock Admin Panel", use_container_width=True):
            st.session_state.admin_unlocked = False
            st.rerun()
    else:
        st.info("🔒 Admin features locked")
        st.caption("Dashboard & Key Health require authentication")

    # App-level logout
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.user_authenticated = False
        st.session_state.admin_unlocked = False
        st.session_state.chat_history = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="mega-header">🚀 GROQ MEGA ENGINE</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Key AI Powerhouse | '
    'Chat • Vision • Audio • Consensus • Debate • Batch</div>',
    unsafe_allow_html=True
)

if not engine_ready:
    st.markdown("---")
    st.error("⚠️ Engine not initialized. Please configure API keys in secrets.")
    st.markdown("""
    ### Setup Guide

    **Option 1: Local Development**
    Create `.streamlit/secrets.toml`:
    ```toml
    [groq]
    keys = ["gsk_your_key_1", "gsk_your_key_2"]
    admin_password = "YourSecurePassword"
    ```

    **Option 2: Streamlit Cloud**
    Go to App Settings → Secrets and paste the same TOML content.
    """)
    st.stop()

engine = st.session_state.engine
router = st.session_state.router


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "💬 Chat",
    "🧠 Consensus",
    "🔗 Chain of Thought",
    "⚡ Batch Process",
    "🎭 Debate",
    "🔀 Compare Models",
    "🖼️ Vision",
    "🎤 Audio STT",
    "🔊 Audio TTS",
    "🛡️ Safe Chat",
    "📋 JSON Mode",
    "🔒 Dashboard",
    "🔒 Key Health",
    "📖 Model Registry",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ═══════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### 💬 AI Chat")

    col_settings, col_chat = st.columns([1, 3])

    with col_settings:
        st.markdown("#### ⚙️ Settings")
        chat_model = st.selectbox(
            "Model", ALL_CHAT_MODELS,
            index=ALL_CHAT_MODELS.index("llama-3.3-70b-versatile")
            if "llama-3.3-70b-versatile" in ALL_CHAT_MODELS else 0,
            key="chat_model_select"
        )
        chat_system = st.text_area(
            "System Prompt",
            value="You are a helpful AI assistant.",
            height=80, key="chat_sys"
        )
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key="chat_temp")
        max_tokens = st.slider("Max Tokens", 100, 8000, 2048, 100, key="chat_max_tok")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    with col_chat:
        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-msg">🧑 {msg["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    meta = msg.get("meta", "")
                    st.markdown(
                        f'<div class="bot-msg">🤖 {msg["content"]}'
                        f'<div class="msg-meta">{meta}</div></div>',
                        unsafe_allow_html=True
                    )

        user_input = st.chat_input("Type your message...", key="chat_input")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            messages = [{"role": "system", "content": chat_system}]
            for h in st.session_state.chat_history:
                messages.append({"role": h["role"], "content": h["content"]})

            with st.spinner(f"🤖 {chat_model.split('/')[-1]} thinking..."):
                result = engine.chat_multi(
                    messages, model=chat_model,
                    max_tokens=max_tokens, temperature=temperature
                )

            if result["success"]:
                # Don't show key name to regular users
                meta = (f"⏱️ {result.get('time_ms', 0)}ms | "
                        f"📊 {result.get('tokens', 0)} tokens")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["response"],
                    "meta": meta,
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ Error: {result.get('error', 'Unknown')}",
                    "meta": "",
                })
            st.rerun()


# ═══════════════════════════════════════════════════════════════════
# TAB 2: CONSENSUS
# ═══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### 🧠 Multi-Model Consensus")
    st.info("Ask multiple models the same question. A judge synthesizes the best answer.")

    consensus_prompt = st.text_area(
        "Your question:",
        placeholder="What will be the most transformative technology of 2030?",
        height=100, key="consensus_q"
    )
    col1, col2 = st.columns(2)
    with col1:
        consensus_models = st.multiselect(
            "Models to consult:", ALL_CHAT_MODELS,
            default=["llama-3.3-70b-versatile", "qwen/qwen3-32b",
                     "openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct"],
            key="consensus_models"
        )
    with col2:
        judge_model = st.selectbox(
            "Judge model:", ALL_CHAT_MODELS,
            index=ALL_CHAT_MODELS.index("llama-3.3-70b-versatile")
            if "llama-3.3-70b-versatile" in ALL_CHAT_MODELS else 0,
            key="consensus_judge"
        )

    if st.button("🧠 Get Consensus", type="primary", key="consensus_btn"):
        if not consensus_prompt.strip():
            st.warning("Please enter a question!")
        else:
            progress = st.progress(0, text="Consulting models...")
            with st.spinner("Getting responses..."):
                result = engine.consensus(
                    consensus_prompt, models=consensus_models,
                    judge_model=judge_model,
                )
            progress.progress(100, text="Done!")

            if result["success"]:
                st.markdown("#### 📋 Individual Responses:")
                individual = result.get("individual", {})
                cols = st.columns(min(len(individual), 3))
                for i, (model, resp) in enumerate(individual.items()):
                    short = model.split("/")[-1] if "/" in model else model
                    with cols[i % len(cols)]:
                        with st.expander(f"🤖 {short}", expanded=False):
                            st.write(resp if isinstance(resp, str) else
                                     resp.get("response", str(resp)))
                st.markdown("---")
                st.markdown("#### ⚖️ Consensus Answer:")
                st.success(result["response"])
            else:
                st.error(f"Failed: {result.get('error', 'Unknown')}")


# ═══════════════════════════════════════════════════════════════════
# TAB 3: CHAIN OF THOUGHT
# ═══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 🔗 Chain of Thought")
    st.info("Break complex tasks into steps. Different models per step.")

    cot_task = st.text_area(
        "Complex task:", placeholder="Design a marketing strategy...",
        height=100, key="cot_task"
    )
    cot_models = st.multiselect(
        "Models (cycled):", ALL_CHAT_MODELS,
        default=["llama-3.3-70b-versatile", "qwen/qwen3-32b",
                 "openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct"],
        key="cot_models"
    )

    if st.button("🔗 Execute Chain", type="primary", key="cot_btn"):
        if not cot_task.strip():
            st.warning("Enter a task!")
        else:
            progress = st.progress(0, text="Planning...")

            def cot_progress(c, t):
                progress.progress(min(c / max(t, 1), 1.0), text=f"Step {c}/{t}")

            with st.spinner("Processing..."):
                result = engine.chain_of_thought(
                    cot_task, models=cot_models, progress_callback=cot_progress
                )

            if result["success"]:
                for i, step in enumerate(result["steps"], 1):
                    short = step.get("model", "?").split("/")[-1]
                    with st.expander(
                        f"Step {i}: {step.get('step', '')[:60]} ({short})",
                        expanded=False
                    ):
                        if "response" in step:
                            st.write(step["response"])
                        else:
                            st.error(step.get("error", "Failed"))
                st.markdown("---")
                st.markdown("#### 📝 Final Answer:")
                st.success(result["final_answer"])


# ═══════════════════════════════════════════════════════════════════
# TAB 4: BATCH
# ═══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### ⚡ Parallel Batch Processing")
    num_keys = len(st.session_state.api_keys)
    st.info(f"Process multiple prompts simultaneously! ({num_keys} parallel workers)")

    batch_mode = st.radio(
        "Input mode:", ["Manual", "Template", "CSV Upload"],
        horizontal=True, key="batch_mode"
    )
    prompts = []

    if batch_mode == "Manual":
        batch_text = st.text_area(
            "Prompts (one per line):",
            placeholder="What is Python?\nWhat is JavaScript?",
            height=150, key="batch_manual"
        )
        prompts = [p.strip() for p in batch_text.strip().split("\n") if p.strip()]
    elif batch_mode == "Template":
        template = st.text_input(
            "Template ({item}):", value="Explain {item} in one paragraph.",
            key="batch_template"
        )
        items_text = st.text_area(
            "Items:", value="quantum computing\nblockchain\nAI",
            height=100, key="batch_items"
        )
        items = [i.strip() for i in items_text.strip().split("\n") if i.strip()]
        prompts = [template.replace("{item}", item) for item in items]
    elif batch_mode == "CSV Upload":
        uploaded = st.file_uploader("CSV (column 'prompt'):", type=["csv"], key="batch_csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            if "prompt" in df.columns:
                prompts = df["prompt"].dropna().tolist()
                st.write(f"Loaded {len(prompts)} prompts")

    batch_model = st.selectbox("Model:", ALL_CHAT_MODELS, key="batch_model")
    batch_workers = st.slider("Workers:", 1, max(num_keys, 1), num_keys, key="batch_workers")

    if prompts:
        st.write(f"📋 {len(prompts)} prompts ready")

    if st.button(f"⚡ Process {len(prompts)} prompts", type="primary", key="batch_btn"):
        if not prompts:
            st.warning("No prompts!")
        else:
            progress = st.progress(0)

            def bp(c, t):
                progress.progress(min(c / max(t, 1), 1.0), text=f"{c}/{t}")

            t0 = time.time()
            results = engine.batch_process(
                prompts, model=batch_model,
                max_workers=batch_workers, progress_callback=bp
            )
            elapsed = time.time() - t0
            successes = sum(1 for r in results if r and r.get("success"))
            st.success(f"✅ {successes}/{len(prompts)} in {elapsed:.1f}s "
                       f"({len(prompts)/max(elapsed,0.01):.1f}/sec)")

            rows = []
            for i, (p, r) in enumerate(zip(prompts, results)):
                rows.append({
                    "Index": i + 1,
                    "Prompt": p[:50],
                    "Status": "✅" if r and r.get("success") else "❌",
                    "Response": (r.get("response", "")[:100] if r and r.get("success")
                                 else r.get("error", "")[:50] if r else "Failed"),
                    "Time (ms)": r.get("time_ms", 0) if r else 0,
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.download_button("📥 Download CSV", df.to_csv(index=False),
                               "batch_results.csv", "text/csv", key="batch_dl")


# ═══════════════════════════════════════════════════════════════════
# TAB 5: DEBATE
# ═══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 🎭 AI Debate Arena")

    debate_topic = st.text_input(
        "Topic:", placeholder="Will AI replace most jobs?", key="debate_topic"
    )
    col1, col2 = st.columns(2)
    with col1:
        debate_models = st.multiselect(
            "Debaters:", ALL_CHAT_MODELS,
            default=["llama-3.3-70b-versatile", "qwen/qwen3-32b", "openai/gpt-oss-120b"],
            key="debate_models"
        )
    with col2:
        debate_rounds = st.slider("Rounds:", 1, 5, 2, key="debate_rounds")

    if st.button("🎭 Start Debate", type="primary", key="debate_btn"):
        if not debate_topic.strip():
            st.warning("Enter a topic!")
        else:
            progress = st.progress(0)

            def dp(c, t):
                progress.progress(min(c / max(t, 1), 1.0))

            with st.spinner("Debating..."):
                result = engine.debate(
                    debate_topic, rounds=debate_rounds,
                    models=debate_models, progress_callback=dp
                )
            if result["success"]:
                for entry in result["log"]:
                    with st.expander(
                        f"R{entry['round']} - 🤖 {entry.get('short', '?')}", expanded=True
                    ):
                        st.write(entry["argument"])
                st.markdown("---")
                st.markdown("#### ⚖️ Verdict:")
                st.success(result["verdict"])


# ═══════════════════════════════════════════════════════════════════
# TAB 6: COMPARE
# ═══════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### 🔀 Model Comparison")

    compare_prompt = st.text_area(
        "Prompt:", placeholder="Explain quantum entanglement.",
        height=80, key="compare_prompt"
    )
    compare_models = st.multiselect(
        "Models:", ALL_CHAT_MODELS, default=ALL_CHAT_MODELS[:4], key="compare_models"
    )

    if st.button("🔀 Compare", type="primary", key="compare_btn"):
        if not compare_prompt.strip() or len(compare_models) < 2:
            st.warning("Need prompt + 2+ models!")
        else:
            progress = st.progress(0)

            def cp(c, t):
                progress.progress(min(c / max(t, 1), 1.0))

            results = engine.compare_models(
                compare_prompt, models=compare_models, progress_callback=cp
            )
            cols = st.columns(min(len(results), 3))
            for i, (m, r) in enumerate(results.items()):
                short = m.split("/")[-1] if "/" in m else m
                with cols[i % len(cols)]:
                    st.markdown(f"**🤖 {short}**")
                    if r["success"]:
                        st.write(r["response"])
                        st.caption(f"⏱️ {r.get('time_ms', 0)}ms | 📊 {r.get('tokens', 0)} tok")
                    else:
                        st.error(r.get("error", "")[:100])

            speed_data = [
                {"Model": m.split("/")[-1] if "/" in m else m,
                 "Time (ms)": r.get("time_ms", 0), "Tokens": r.get("tokens", 0)}
                for m, r in results.items() if r["success"]
            ]
            if speed_data:
                fig = px.bar(pd.DataFrame(speed_data), x="Model", y="Time (ms)",
                             title="⚡ Speed Comparison", color="Tokens")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                                  paper_bgcolor='rgba(0,0,0,0)', font_color='#aac')
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 7: VISION
# ═══════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 🖼️ Vision Analysis")

    vision_model = st.selectbox(
        "Model:", list(MODEL_REGISTRY.get("Vision (Multimodal)", {}).keys()),
        key="vision_model"
    )
    uploaded_img = st.file_uploader(
        "Upload image:", type=["png", "jpg", "jpeg", "gif", "webp"], key="vision_upload"
    )
    vision_prompt = st.text_input(
        "Question:", value="Describe this image in detail.", key="vision_prompt"
    )

    if uploaded_img:
        st.image(uploaded_img, use_container_width=True)
        if st.button("🖼️ Analyze", type="primary", key="vision_btn"):
            with st.spinner("Analyzing..."):
                result = engine.see(uploaded_img.getvalue(), vision_prompt, vision_model)
            if result["success"]:
                st.write(result["response"])
                st.caption(f"⏱️ {result.get('time_ms', 0)}ms | 📊 {result.get('tokens', 0)} tok")
            else:
                st.error(result.get("error", "Failed"))


# ═══════════════════════════════════════════════════════════════════
# TAB 8: STT
# ═══════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("### 🎤 Speech to Text")

    stt_model = st.selectbox(
        "Model:", list(MODEL_REGISTRY.get("Audio STT (Whisper)", {}).keys()), key="stt_model"
    )
    uploaded_audio = st.file_uploader(
        "Upload audio:", type=["mp3", "wav", "m4a", "ogg", "flac"], key="stt_upload"
    )

    if uploaded_audio:
        st.audio(uploaded_audio)
        if st.button("🎤 Transcribe", type="primary", key="stt_btn"):
            with st.spinner("Transcribing..."):
                result = engine.listen(uploaded_audio.getvalue(), uploaded_audio.name, stt_model)
            if result["success"]:
                st.text_area("📝 Result:", value=result["text"], height=200, key="stt_res")
                st.caption(f"⏱️ {result.get('time_ms', 0)}ms | Duration: {result.get('duration', '?')}s")
                if result.get("segments"):
                    with st.expander("📋 Segments"):
                        for s in result["segments"]:
                            st.write(f"[{s.get('start',0):.1f}s-{s.get('end',0):.1f}s] {s.get('text','')}")
            else:
                st.error(result.get("error", "Failed"))


# ═══════════════════════════════════════════════════════════════════
# TAB 9: TTS
# ═══════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown("### 🔊 Text to Speech")

    tts_model = st.selectbox(
        "Model:", list(MODEL_REGISTRY.get("Audio TTS (Orpheus)", {}).keys()), key="tts_model"
    )
    voices = TTS_VOICES.get(tts_model, {}).get("voices", ["diana"])
    tts_voice = st.selectbox("Voice:", voices, key="tts_voice")
    tts_text = st.text_area(
        "Text:", value="Hello! I am the Groq Mega Engine.", height=100, key="tts_text"
    )

    if st.button("🔊 Generate", type="primary", key="tts_btn"):
        if not tts_text.strip():
            st.warning("Enter text!")
        else:
            with st.spinner("Generating..."):
                result = engine.speak(tts_text, tts_voice, tts_model)
            if result["success"]:
                st.audio(result["audio_bytes"], format="audio/wav")
                st.caption(f"🗣️ {result['voice']} | ⏱️ {result.get('time_ms', 0)}ms | {result['size_kb']}KB")
                st.download_button("📥 Download", result["audio_bytes"],
                                   "speech.wav", "audio/wav", key="tts_dl")
            else:
                st.error(result.get("error", "Failed"))


# ═══════════════════════════════════════════════════════════════════
# TAB 10: SAFE CHAT
# ═══════════════════════════════════════════════════════════════════
with tabs[9]:
    st.markdown("### 🛡️ Safety-Checked Chat")
    st.info("Guard models check input & output automatically.")

    safe_model = st.selectbox("Model:", ALL_CHAT_MODELS, key="safe_model")
    safe_prompt = st.text_area("Message:", placeholder="How to make pasta?",
                               height=80, key="safe_prompt")

    if st.button("🛡️ Send", type="primary", key="safe_btn"):
        if not safe_prompt.strip():
            st.warning("Enter a message!")
        else:
            with st.spinner("Checking safety..."):
                result = engine.safe_chat(safe_prompt, model=safe_model)

            if result.get("blocked"):
                st.error(f"🚫 BLOCKED: {result.get('reason')}")
            elif result.get("success"):
                c1, c2 = st.columns(2)
                with c1:
                    s = result.get("input_safety", "unknown")
                    st.markdown(f"**Input:** :{'green' if 'safe' in str(s) else 'red'}[{s}]")
                with c2:
                    s = result.get("output_safety", "unknown")
                    st.markdown(f"**Output:** :{'green' if 'safe' in str(s) else 'red'}[{s}]")
                st.write(result["response"])
            else:
                st.error(result.get("error", "Failed"))


# ═══════════════════════════════════════════════════════════════════
# TAB 11: JSON MODE
# ═══════════════════════════════════════════════════════════════════
with tabs[10]:
    st.markdown("### 📋 JSON Output")

    json_model = st.selectbox("Model:", ALL_CHAT_MODELS, key="json_model")
    json_prompt = st.text_area(
        "Prompt:", value="List 5 programming languages as JSON.", height=80, key="json_prompt"
    )

    if st.button("📋 Generate JSON", type="primary", key="json_btn"):
        with st.spinner("Generating..."):
            result = engine.json_chat(json_prompt, model=json_model)
        if result["success"]:
            st.caption(f"⏱️ {result.get('time_ms', 0)}ms | 📊 {result.get('tokens', 0)} tok")
            if result.get("parsed"):
                st.json(result["parsed"])
            else:
                st.code(result.get("response", ""), language="json")
        else:
            st.error(result.get("error", "Failed"))


# ═══════════════════════════════════════════════════════════════════
# TAB 12: 🔒 DASHBOARD (ADMIN ONLY)
# ═══════════════════════════════════════════════════════════════════
with tabs[11]:
    st.markdown("### 🔒 System Dashboard")

    if not st.session_state.admin_unlocked:
        # ─── LOGIN GATE ───
        st.markdown("""
        <div class="lock-container">
            <div class="lock-icon">🔒</div>
            <h3 style="color: #aaaacc;">Admin Authentication Required</h3>
            <p style="color: #6666aa;">Enter your admin password to access
            system dashboard, key health, and usage statistics.</p>
        </div>
        """, unsafe_allow_html=True)

        # Check lockout
        now = time.time()
        if st.session_state.lockout_until > now:
            remaining = int(st.session_state.lockout_until - now)
            st.error(f"🔒 Too many failed attempts. Try again in {remaining} seconds.")
        else:
            with st.form("admin_login_dashboard"):
                password = st.text_input(
                    "🔑 Admin Password:",
                    type="password",
                    placeholder="Enter admin password...",
                )
                submitted = st.form_submit_button("🔓 Unlock", use_container_width=True)

                if submitted:
                    if verify_password(password):
                        st.session_state.admin_unlocked = True
                        st.session_state.admin_unlock_time = datetime.now()
                        st.session_state.failed_attempts = 0
                        st.success("✅ Authenticated! Refreshing...")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.session_state.failed_attempts += 1
                        remaining = 5 - st.session_state.failed_attempts

                        if st.session_state.failed_attempts >= 5:
                            # Lock out for 5 minutes
                            st.session_state.lockout_until = time.time() + 300
                            st.session_state.failed_attempts = 0
                            st.error("🔒 Too many attempts! Locked for 5 minutes.")
                        else:
                            st.error(f"❌ Wrong password. {remaining} attempts remaining.")
    else:
        # ─── DASHBOARD CONTENT (UNLOCKED) ───
        st.success("🔓 Admin access granted")

        if st.button("🔄 Refresh Dashboard", key="dash_refresh"):
            st.rerun()

        stats = engine.get_stats()
        es = stats["engine"]
        ks = stats["keys"]

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Calls", es["total_calls"])
        with c2:
            st.metric("Total Tokens", f"{es['total_tokens']:,}")
        with c3:
            st.metric("Errors", es["errors"])
        with c4:
            rate = ((es['total_calls'] - es['errors']) / max(es['total_calls'], 1)) * 100
            st.metric("Success Rate", f"{rate:.1f}%")

        # Key usage (masked)
        st.markdown("#### 🔑 Key Usage Distribution")
        key_data = []
        for label, s in ks.items():
            key_data.append({
                "Key": label,
                "Masked": s["masked_key"],
                "Calls": s["calls"],
                "Errors": s["errors"],
                "Tokens": s["tokens"],
                "Avg (ms)": s["avg_response_ms"],
                "Status": "🟢 Active" if s["is_available"] else "🔴 Cooldown",
            })

        if key_data:
            df = pd.DataFrame(key_data)
            st.dataframe(df, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(df, x="Key", y="Calls", title="Calls per Key",
                             color="Errors", color_continuous_scale="RdYlGn_r")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                                  paper_bgcolor='rgba(0,0,0,0)', font_color='#aac')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.bar(df, x="Key", y="Tokens", title="Tokens per Key",
                              color="Avg (ms)", color_continuous_scale="Viridis")
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                                   paper_bgcolor='rgba(0,0,0,0)', font_color='#aac')
                st.plotly_chart(fig2, use_container_width=True)

        # Call log
        st.markdown("#### 📜 Recent Call Log")
        log = stats.get("call_log", [])
        if log:
            log_df = pd.DataFrame(log[-20:][::-1])
            # Remove key column from public view - show only to admin
            st.dataframe(log_df, use_container_width=True, height=300)
        else:
            st.info("No calls yet.")


# ═══════════════════════════════════════════════════════════════════
# TAB 13: 🔒 KEY HEALTH (ADMIN ONLY)
# ═══════════════════════════════════════════════════════════════════
with tabs[12]:
    st.markdown("### 🔒 API Key Health")

    if not st.session_state.admin_unlocked:
        st.markdown("""
        <div class="lock-container">
            <div class="lock-icon">🔒</div>
            <h3 style="color: #aaaacc;">Admin Access Required</h3>
            <p style="color: #6666aa;">Go to the Dashboard tab to authenticate first.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("🔓 Admin access granted")

        if st.button("🔍 Check All Keys", type="primary", key="health_btn"):
            with st.spinner("Validating..."):
                health = router.validate_keys()
            for label, info in health.items():
                if info["valid"]:
                    st.success(f"✅ {label}: {info['masked']} — {info['models']} models")
                else:
                    st.error(f"❌ {label}: {info.get('error', 'Invalid')[:80]}")

        st.markdown("---")
        st.markdown("#### 🧪 Test Model on All Keys")

        test_model_sel = st.selectbox("Model:", get_all_models_flat(), key="health_model")

        if st.button("🧪 Test", key="health_test_btn"):
            with st.spinner("Testing..."):
                for i, key in enumerate(st.session_state.api_keys):
                    label = f"Key_{i+1}"
                    try:
                        client = Groq(api_key=key)
                        t0 = time.time()

                        stt_list = list(MODEL_REGISTRY.get("Audio STT (Whisper)", {}).keys())
                        tts_list = list(MODEL_REGISTRY.get("Audio TTS (Orpheus)", {}).keys())

                        if test_model_sel in stt_list:
                            st.info(f"{label}: STT - upload audio to test")
                        elif test_model_sel in tts_list:
                            v = TTS_VOICES.get(test_model_sel, {}).get("default", "diana")
                            client.audio.speech.create(
                                model=test_model_sel, input="Test",
                                voice=v, response_format="wav",
                            )
                            ms = round((time.time() - t0) * 1000)
                            st.success(f"✅ {label}: TTS works ({ms}ms)")
                        else:
                            sys_msg = "Reply in 5 words."
                            if "qwen3" in test_model_sel.lower():
                                sys_msg += " /no_think"
                            resp = client.chat.completions.create(
                                model=test_model_sel,
                                messages=[
                                    {"role": "system", "content": sys_msg},
                                    {"role": "user", "content": "Hello"},
                                ],
                                max_tokens=20, temperature=0.0,
                            )
                            ms = round((time.time() - t0) * 1000)
                            reply = resp.choices[0].message.content.strip()[:50]
                            st.success(f"✅ {label}: {ms}ms — \"{reply}\"")

                    except Exception as e:
                        st.error(f"❌ {label}: {str(e)[:100]}")
                    time.sleep(0.5)


# ═══════════════════════════════════════════════════════════════════
# TAB 14: MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════
with tabs[13]:
    st.markdown("### 📖 Model Registry")

    for category, models in MODEL_REGISTRY.items():
        with st.expander(f"📂 {category} ({len(models)} models)", expanded=True):
            for model_id, info in models.items():
                c1, c2, c3, c4 = st.columns([3, 4, 1, 1])
                with c1:
                    short = model_id.split("/")[-1] if "/" in model_id else model_id
                    st.markdown(f"**{short}**")
                with c2:
                    st.write(info["desc"])
                with c3:
                    st.write(info["speed"])
                with c4:
                    st.write(info["quality"])

    st.markdown("---")
    st.markdown("#### 🗣️ TTS Voices")
    for model, config in TTS_VOICES.items():
        short = model.split("/")[-1]
        st.write(f"**{short}**: {', '.join(config['voices'])} (default: {config['default']})")


# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #555; font-size: 0.8rem;">'
    '🚀 Groq Mega Engine v2.0 | Secured & Production Ready | '
    f'{datetime.now().strftime("%Y-%m-%d %H:%M")}'
    '</div>',
    unsafe_allow_html=True
)