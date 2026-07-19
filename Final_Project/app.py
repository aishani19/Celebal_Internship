import os
import sys
import torch
import torch.nn.functional as F
import streamlit as st

# Paths
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

LOG_DIR  = os.path.abspath("./logs")
DATA_DIR = os.path.abspath("./data")

CKPT_50M = os.path.join(LOG_DIR, "model_50M_wikitext.pt")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(
    page_title="GPT-2 · 50M",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; background: #0b0d14; color: #e2e8f0; }
code, pre { font-family: 'JetBrains Mono', monospace; }

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

.gpt-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00f2fe 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.gpt-subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 2.5rem;
}

/* Status pill */
.pill {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.pill-training { background: rgba(16,185,129,.15); color: #10b981; border: 1px solid rgba(16,185,129,.35); }
.pill-ready    { background: rgba(139,92,246,.15);  color: #a78bfa; border: 1px solid rgba(139,92,246,.35); }
.pill-idle     { background: rgba(100,116,139,.15); color: #94a3b8; border: 1px solid rgba(100,116,139,.35); }

/* Text input */
textarea {
    background: #141824 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
}

/* Output card */
.output-card {
    background: #141824;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.2rem;
    line-height: 1.75;
    font-size: 1rem;
}
.output-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.token-highlight {
    color: #00f2fe;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.prompt-text {
    color: #94a3b8;
}

/* Generate button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    border: none !important;
    border-radius: 10px !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.4) !important;
}

/* Token table */
.token-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
}
.token-table th {
    color: #64748b;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.07em;
    padding: 0.4rem 0.8rem;
    border-bottom: 1px solid #334155;
    text-align: left;
}
.token-table td {
    padding: 0.35rem 0.8rem;
    border-bottom: 1px solid #1e293b;
    color: #e2e8f0;
}
.token-table tr:first-child td { color: #00f2fe; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

if "model" not in st.session_state:
    st.session_state.model = None

@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    from model_50M import GPT
    model = GPT(config=ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, device

st.markdown('<div class="gpt-title">GPT-2 Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="gpt-subtitle">From-scratch transformer · next-token prediction</div>', unsafe_allow_html=True)

if os.path.exists(CKPT_50M):
    FINAL_CKPT = CKPT_50M
    st.markdown(f'<p style="text-align:center"><span class="pill pill-ready">&#x2705; Model ready (50M WikiText-103) &mdash; enter a prompt and generate!</span></p>', unsafe_allow_html=True)
else:
    FINAL_CKPT = None
    st.markdown('<p style="text-align:center"><span class="pill pill-idle">&#x23F3; Checkpoint not found. Run train_50M.py first!</span></p>', unsafe_allow_html=True)

st.write("")


prompt = st.text_area(
    "Enter your prompt",
    placeholder="Type something and press Generate…",
    height=130,
    label_visibility="collapsed",
)

top_k_n = st.slider("How many top predictions to show", min_value=1, max_value=20, value=5)

generate_btn = st.button("Generate ✨", disabled=(FINAL_CKPT is None))

# Output placeholder
output_area = st.empty()

if generate_btn:
    if not prompt.strip():
        output_area.warning("Please enter a prompt first.")
    elif FINAL_CKPT is None:
        output_area.error("No checkpoint found. Run `python train_50M.py` first.")
    else:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")

            with st.spinner("Loading model ..."):
                model, device = load_model(FINAL_CKPT)

            tokens = enc.encode(prompt.strip())
            if len(tokens) == 0:
                output_area.warning("Prompt produced no tokens after encoding.")
            else:
                input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

                with torch.no_grad():
                    with torch.autocast(
                        device_type="cuda" if "cuda" in device else "cpu",
                        dtype=torch.bfloat16,
                        enabled=("cuda" in device),
                    ):
                        logits, _ = model(input_ids)

                # Logits for the NEXT token (last position)
                next_logits = logits[0, -1, :]           # (vocab_size,)
                probs = F.softmax(next_logits, dim=-1)

                top_probs, top_indices = torch.topk(probs, top_k_n)
                top_probs   = top_probs.cpu().tolist()
                top_indices = top_indices.cpu().tolist()

                # Decode top tokens
                top_tokens = [enc.decode([idx]) for idx in top_indices]
                best_token = top_tokens[0]

                # Build table rows HTML
                rows_html = ""
                for rank, (tok, prob) in enumerate(zip(top_tokens, top_probs)):
                    tok_display = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    rows_html += f"""<tr>
  <td>#{rank+1}</td>
  <td>"{tok_display}"</td>
  <td>{prob*100:.2f}%</td>
</tr>"""

                best_display = best_token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                prompt_display = prompt.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                output_area.markdown(f"""
<div class="output-card">
  <div class="output-label">Prompt</div>
  <div class="prompt-text">{prompt_display}</div>
  <br/>
  <div class="output-label">Next predicted token</div>
  <span class="token-highlight">"{best_display}"</span>
  <br/><br/>
  <div class="output-label">Top-{top_k_n} predictions</div>
  <table class="token-table">
    <thead>
      <tr><th>Rank</th><th>Token</th><th>Probability</th></tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</div>
""", unsafe_allow_html=True)

        except Exception as exc:
            output_area.error(f"Generation error: {exc}")


