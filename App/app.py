# Gradio chat app for FitCoach — runs on Hugging Face Spaces with ZeroGPU.
# Models load on CPU at startup; GPU is acquired per request inside @spaces.GPU.

import os
from threading import Thread

import gradio as gr
import spaces
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_3B_ID = "Harsh-k-007/fitcoach-3b"
BASE_8B_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_8B_ID = "Harsh-k-007/fitcoach-8b-adapter"
HF_TOKEN = os.environ.get("HF_TOKEN")

ENABLE_8B = True
ENABLE_3B = True  # set False if startup logs show "Killed" due to CPU RAM limit

PAD_TOKEN_ID = 128004  # <|finetune_right_pad_id|>
MAX_NEW_TOKENS = 512
GPU_DURATION = 120

SYSTEM_PROMPT = (
    "You are FitCoach, a friendly and knowledgeable fitness and nutrition coaching "
    "assistant. You help users with meal plans and workout plans only. "
    "You do not give advice on injuries, medical conditions, or anything outside "
    "fitness and nutrition. "
    "Conduct a conversational intake by asking one question at a time. "
    "Acknowledge each answer before asking the next question. "
    "For meal plans, collect: goal, age/height/weight, dietary restrictions, activity level. "
    "For workout plans, collect: goal, experience level, days per week, equipment access. "
    "Once intake is complete, generate a clear structured plan. "
    "Keep responses practical, structured, motivating, and easy to follow."
)

# ---------------------------------------------------------------------------
# Load models on CPU at startup (ZeroGPU: no GPU at module level)
# ---------------------------------------------------------------------------
MODELS = {}

if ENABLE_8B:
    print("Loading FitCoach 8B (base + LoRA adapter) on CPU...")
    tok_8b = AutoTokenizer.from_pretrained(BASE_8B_ID, token=HF_TOKEN)
    tok_8b.pad_token = "<|finetune_right_pad_id|>"
    base_8b = AutoModelForCausalLM.from_pretrained(
        BASE_8B_ID,
        dtype=torch.float16,
        device_map="cpu",
        token=HF_TOKEN,
    )
    model_8b = PeftModel.from_pretrained(base_8b, ADAPTER_8B_ID, token=HF_TOKEN)
    model_8b.eval()
    MODELS["FitCoach 8B"] = (model_8b, tok_8b)
    print("FitCoach 8B ready")

if ENABLE_3B:
    print("Loading FitCoach 3B (merged) on CPU...")
    tok_3b = AutoTokenizer.from_pretrained(MODEL_3B_ID, token=HF_TOKEN)
    tok_3b.pad_token = "<|finetune_right_pad_id|>"
    model_3b = AutoModelForCausalLM.from_pretrained(
        MODEL_3B_ID,
        dtype=torch.float16,
        device_map="cpu",
        token=HF_TOKEN,
    )
    model_3b.eval()
    MODELS["FitCoach 3B"] = (model_3b, tok_3b)
    print("FitCoach 3B ready")

if not MODELS:
    raise RuntimeError("No models enabled.")

MODEL_CHOICES = list(MODELS.keys())
DEFAULT_MODEL = MODEL_CHOICES[0]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_content(content):
    """History content can be a string or a list of blocks, always return str."""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", "") or item.get("content", ""))
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p)
    if isinstance(content, str):
        return content
    return str(content)

# ---------------------------------------------------------------------------
# Generation (GPU only inside this function)
# ---------------------------------------------------------------------------
@spaces.GPU(duration=GPU_DURATION)
def generate_stream(messages, model_choice):
    model, tok = MODELS.get(model_choice, MODELS[DEFAULT_MODEL])

    encoded = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # Current transformers returns BatchEncoding, older versions return a tensor
    if isinstance(encoded, torch.Tensor):
        input_ids = encoded
    else:
        input_ids = encoded["input_ids"]

    model.to("cuda")
    input_ids = input_ids.to("cuda")

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=PAD_TOKEN_ID,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    try:
        for piece in streamer:
            partial += piece
            yield partial
    finally:
        thread.join()
        model.to("cpu")
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Chat function for gr.ChatInterface
# ---------------------------------------------------------------------------
def respond(message, history, model_choice):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history:
        role = turn.get("role", "user")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": extract_content(turn.get("content", ""))})

    messages.append({"role": "user", "content": message})

    for partial in generate_stream(messages, model_choice):
        yield partial

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
HERO = """
<div class="fc-shell">
  <div class="fc-hero-card">
    <div class="fc-badge">🌱 AI fitness & nutrition coach</div>
    <h1>FitCoach</h1>
    <p class="fc-subtitle">Build smarter workout and meal plans through a quick guided chat.</p>
    <div class="fc-stats">
      <div><strong>1:1</strong><span>guided intake</span></div>
      <div><strong>2</strong><span>model options</span></div>
      <div><strong>Live</strong><span>streaming replies</span></div>
    </div>
  </div>
</div>
"""

PLACEHOLDER = """
<div class="fc-welcome">
  <div class="fc-orb">🌱</div>
  <h2>What are we building today?</h2>
  <p>Tell me your goal and I’ll ask a few quick questions, one at a time, then create a clear plan for you.</p>
  <div class="fc-chips">
    <span>💪 Build me a workout plan</span>
    <span>🥗 I want a fat-loss meal plan</span>
    <span>🏠 Train 3 days a week at home</span>
    <span>⚡ Improve strength and stamina</span>
  </div>
</div>
"""

DISCLAIMER = """
<div class="fc-disclaimer">
  FitCoach can make mistakes. Verify important health information and speak to a qualified professional for medical concerns.
</div>
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --fc-bg-1: #07130f;
  --fc-bg-2: #10251b;
  --fc-card: rgba(15, 25, 20, 0.78);
  --fc-card-2: rgba(255, 255, 255, 0.06);
  --fc-border: rgba(183, 255, 197, 0.18);
  --fc-text: #f4fff7;
  --fc-muted: rgba(244, 255, 247, 0.68);
  --fc-green: #9dff9d;
  --fc-lime: #d7ff6f;
  --fc-shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
}

body,
.gradio-container {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  background:
    radial-gradient(circle at 20% 10%, rgba(157, 255, 157, 0.18), transparent 28%),
    radial-gradient(circle at 80% 0%, rgba(215, 255, 111, 0.12), transparent 24%),
    linear-gradient(135deg, var(--fc-bg-1), var(--fc-bg-2) 48%, #050806) !important;
  color: var(--fc-text) !important;
}

.gradio-container *,
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container label {
  font-family: 'Inter', system-ui, sans-serif;
}

h1, h2, h3,
.fc-hero-card h1,
.fc-welcome h2,
.fc-stats strong {
  font-family: 'Space Grotesk', 'Inter', sans-serif !important;
}

.gradio-container {
  max-width: 1120px !important;
  margin: 0 auto !important;
  padding: 24px 18px 34px !important;
}

#fc-app {
  max-width: 980px;
  margin: 0 auto;
}

.fc-shell {
  padding: 18px 0 14px;
}

.fc-hero-card {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--fc-border);
  border-radius: 32px;
  padding: 34px 34px 28px;
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.11), rgba(255, 255, 255, 0.035)),
    linear-gradient(135deg, rgba(80, 255, 130, 0.08), rgba(215, 255, 111, 0.04));
  box-shadow: var(--fc-shadow);
  backdrop-filter: blur(18px);
}

.fc-hero-card::after {
  content: "";
  position: absolute;
  width: 280px;
  height: 280px;
  right: -90px;
  top: -110px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(157, 255, 157, 0.26), transparent 66%);
  pointer-events: none;
}

.fc-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  width: fit-content;
  padding: 8px 13px;
  border: 1px solid rgba(157, 255, 157, 0.28);
  border-radius: 999px;
  background: rgba(157, 255, 157, 0.08);
  color: var(--fc-green);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.2px;
}

.fc-hero-card h1 {
  margin: 16px 0 8px;
  font-size: clamp(42px, 7vw, 72px);
  font-weight: 700;
  line-height: 0.98;
  letter-spacing: -2.5px;
  background: linear-gradient(120deg, #f4fff7 30%, #9dff9d 70%, #d7ff6f 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  color: var(--fc-green);
}

.fc-subtitle {
  max-width: 640px;
  margin: 0;
  font-size: 17px;
  font-weight: 400;
  line-height: 1.65;
  letter-spacing: 0.1px;
  color: var(--fc-muted);
}

.fc-stats {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 26px;
}

.fc-stats div {
  padding: 16px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 20px;
  background: rgba(0, 0, 0, 0.18);
}

.fc-stats strong {
  display: block;
  font-size: 24px;
  font-weight: 700;
  letter-spacing: -0.5px;
  color: var(--fc-lime);
}

.fc-stats span {
  display: block;
  margin-top: 5px;
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: var(--fc-muted);
}

#fc-chat-card {
  border: 1px solid var(--fc-border) !important;
  border-radius: 30px !important;
  background: rgba(8, 12, 10, 0.62) !important;
  box-shadow: var(--fc-shadow) !important;
  backdrop-filter: blur(18px);
  padding: 14px !important;
}

#fc-chat-card .wrap,
#fc-chat-card .contain,
#fc-chat-card .block {
  border-radius: 22px !important;
}

#fc-chatbot {
  min-height: 560px !important;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  overflow: hidden !important;
}

/* Flatten every intermediate container inside the chat card so the
   glass card itself is the only visible box. */
#fc-chat-card .block,
#fc-chat-card .form,
#fc-chat-card .gap,
#fc-chat-card .panel,
#fc-chat-card .bubble-wrap,
#fc-chat-card fieldset {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

#fc-chatbot .message {
  border-radius: 18px !important;
  font-size: 15px !important;
  line-height: 1.7 !important;
  letter-spacing: 0.1px !important;
}

#fc-chatbot .user,
#fc-chatbot [data-testid="user"] {
  background: linear-gradient(135deg, rgba(157, 255, 157, 0.22), rgba(215, 255, 111, 0.13)) !important;
  border: 1px solid rgba(157, 255, 157, 0.22) !important;
}

#fc-chatbot .bot,
#fc-chatbot [data-testid="bot"] {
  background: rgba(255, 255, 255, 0.065) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

.fc-welcome {
  min-height: 430px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 38px 22px;
}

.fc-orb {
  display: grid;
  place-items: center;
  width: 86px;
  height: 86px;
  margin-bottom: 20px;
  border-radius: 28px;
  background:
    radial-gradient(circle at 30% 25%, rgba(255, 255, 255, 0.35), transparent 26%),
    linear-gradient(135deg, rgba(157, 255, 157, 0.26), rgba(215, 255, 111, 0.14));
  border: 1px solid rgba(157, 255, 157, 0.24);
  box-shadow: 0 18px 55px rgba(77, 255, 124, 0.16);
  font-size: 42px;
}

.fc-welcome h2 {
  margin: 0 0 10px;
  font-size: clamp(26px, 4vw, 38px);
  font-weight: 600;
  line-height: 1.08;
  letter-spacing: -1.2px;
  color: var(--fc-text);
}

.fc-welcome p {
  max-width: 620px;
  margin: 0 0 22px;
  color: var(--fc-muted);
  font-size: 15.5px;
  line-height: 1.7;
}

.fc-chips {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
  max-width: 760px;
}

.fc-chips span {
  display: inline-flex;
  align-items: center;
  padding: 10px 14px;
  border: 1px solid rgba(157, 255, 157, 0.20);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.055);
  color: rgba(244, 255, 247, 0.88);
  font-size: 13.5px;
  white-space: nowrap;
}

#fc-chat-card #fc-models {
  margin: 4px 4px 10px !important;
  padding: 12px 16px !important;
  border: 1px solid rgba(157, 255, 157, 0.16) !important;
  border-radius: 18px !important;
  background: rgba(255, 255, 255, 0.045) !important;
}

#fc-models .wrap {
  gap: 10px !important;
}

#fc-input textarea,
textarea[data-testid="textbox"] {
  border-radius: 18px !important;
  border: 1px solid rgba(157, 255, 157, 0.18) !important;
  background: rgba(255, 255, 255, 0.07) !important;
  color: var(--fc-text) !important;
  min-height: 54px !important;
}

button.primary,
button[aria-label="Submit"] {
  border-radius: 18px !important;
  background: linear-gradient(135deg, #9dff9d, #d7ff6f) !important;
  color: #07130f !important;
  border: none !important;
  font-weight: 800 !important;
  box-shadow: 0 12px 30px rgba(157, 255, 157, 0.18) !important;
}

.fc-disclaimer {
  margin: 12px auto 0;
  max-width: 900px;
  text-align: center;
  color: rgba(244, 255, 247, 0.52);
  font-size: 12.5px;
  line-height: 1.5;
}

footer {
  opacity: 0.55 !important;
}

@media (max-width: 720px) {
  .gradio-container {
    padding: 14px 10px 28px !important;
  }

  .fc-hero-card {
    padding: 26px 22px 22px;
    border-radius: 26px;
  }

  .fc-stats {
    grid-template-columns: 1fr;
  }

  #fc-chatbot {
    min-height: 520px !important;
  }

  .fc-chips span {
    white-space: normal;
  }
}
"""

THEME = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="lime",
    neutral_hue="slate",
    radius_size="lg",
    text_size="md",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="transparent",
    block_background_fill="rgba(255, 255, 255, 0.04)",
    block_border_color="rgba(157, 255, 157, 0.16)",
    input_background_fill="rgba(255, 255, 255, 0.07)",
    button_primary_background_fill="linear-gradient(135deg, #9dff9d, #d7ff6f)",
    button_primary_text_color="#07130f",
)

with gr.Blocks() as demo:
    with gr.Column(elem_id="fc-app"):
        gr.HTML(HERO)

        with gr.Group(elem_id="fc-chat-card"):
            model_radio = gr.Radio(
                MODEL_CHOICES,
                value=DEFAULT_MODEL,
                label="Choose model",
                info="Use 8B for stronger answers, 3B for lighter responses.",
                elem_id="fc-models",
            )

            gr.ChatInterface(
                fn=respond,
                chatbot=gr.Chatbot(
                    placeholder=PLACEHOLDER,
                    show_label=False,
                    elem_id="fc-chatbot",
                    height=560,
                ),
                textbox=gr.Textbox(
                    placeholder="Ask FitCoach to build a workout plan, meal plan, or fitness routine...",
                    show_label=False,
                    elem_id="fc-input",
                    container=False,
                ),
                additional_inputs=[model_radio],
            )

        gr.HTML(DISCLAIMER)

demo.queue().launch(theme=THEME, css=CSS)
