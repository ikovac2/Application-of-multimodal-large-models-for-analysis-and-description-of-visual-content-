"""
Leather Defect Detection — GUI demonstracija
Učitaj sliku i pozovi CLIP / Moondream / LLaVA-7B / LLaVA-13B

POKRETANJE:
    U Anaconda Promptu, u folderu gdje je ovaj fajl:
    python leather_defect_gui.py

NAPOMENA: Ollama mora biti pokrenuta u pozadini
prije pokretanja ove aplikacije.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # fix za OMP Error #15 (konflikt PyTorch/numpy)

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import base64
import requests
import time

# ============================================================
# Tamna tema — paleta boja
# ============================================================
BG_DARK      = "#1a1d29"
BG_PANEL     = "#232838"
BG_CARD      = "#2a3045"
ACCENT       = "#5b8def"
ACCENT_HOVER = "#4a7ad9"
TEXT_PRIMARY = "#e8eaf0"
TEXT_SECOND  = "#8b92a8"
GREEN        = "#4caf80"
ORANGE       = "#e6a23c"
PURPLE       = "#9d6fd6"
PURPLE_DARK  = "#7c5bb0"
RED          = "#e6614c"
BORDER       = "#363c52"

# ============================================================
# CLIP setup (učitava se jednom pri pokretanju aplikacije)
# ============================================================
print("Učitavam CLIP model... (može potrajati 1-2 minute)")
import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

DEFECT_PROMPTS = [
    "a leather surface with visible color defects or discoloration",
    "a leather surface with grain off or damaged surface texture",
    "a leather surface with folding marks or creases",
    "a leather surface with pin holes or small holes",
    "a leather surface with cuts or scratches or scars",
    "a leather surface with wrinkles or neck wrinkles",
    "a leather surface with loose grain or growth marks",
    "a clean leather surface without any visible defects"
]
DEFECT_LABELS = [
    "color", "grain_off", "folding_marks", "pin_holes",
    "cuts_scars", "wrinkles", "loose_grain", "defect_free"
]

text_tokens = clip.tokenize(DEFECT_PROMPTS).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

print("✅ CLIP spreman!")

VQA_PROMPT = """You are an expert in leather quality inspection.
Analyze this leather sample image carefully.

Answer these three questions:
1. DEFECT_PRESENT: Are there any visible defects on this leather surface? Answer only Yes or No.
2. DEFECT_TYPE: If yes, what type of defects do you see?
   Choose from: color, grain_off, folding_marks, pin_holes, cuts_scars, wrinkles, loose_grain, other.
   If no defects, write: none
3. SEVERITY: Rate the severity: None, Minor, or Major.

Format your answer exactly like this:
DEFECT_PRESENT: Yes/No
DEFECT_TYPE: type1, type2
SEVERITY: None/Minor/Major
DESCRIPTION: one sentence describing what you see"""


def classify_with_clip(image_path):
    """Vraća (top_label, top_prob, top3_list, verdict_text)."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_features = clip_model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            similarities = (img_features @ text_features.T).squeeze(0)
            probs = similarities.softmax(dim=-1).cpu().numpy()

        sorted_idx = probs.argsort()[::-1]
        top3 = [(DEFECT_LABELS[i], float(probs[i])) for i in sorted_idx[:3]]
        top_label = DEFECT_LABELS[sorted_idx[0]]
        is_clean = (top_label == "defect_free")
        return {
            "ok": True, "model": "CLIP", "time": None,
            "top3": top3, "is_clean": is_clean, "top_label": top_label
        }
    except Exception as e:
        return {"ok": False, "model": "CLIP", "error": str(e)}


def query_ollama(image_path, model_name):
    """Šalje sliku Ollama modelu, vraća parsiran rezultat dict."""
    try:
        with open(image_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        payload = {
            "model": model_name, "prompt": VQA_PROMPT,
            "images": [img_b64], "stream": False
        }
        start = time.time()
        r = requests.post('http://localhost:11434/api/generate', json=payload, timeout=300)
        elapsed = time.time() - start

        if r.status_code != 200:
            return {"ok": False, "model": model_name,
                    "error": f"HTTP {r.status_code} — model možda nije instaliran (ollama pull {model_name})"}

        raw = r.json().get('response', '').strip()
        parsed = {"defect_present": None, "defect_type": None, "severity": None, "description": None}
        for line in raw.split('\n'):
            line = line.strip()
            if line.upper().startswith('DEFECT_PRESENT:'):
                val = line.split(':', 1)[1].strip().lower()
                parsed["defect_present"] = 'yes' in val
            elif line.upper().startswith('DEFECT_TYPE:'):
                parsed["defect_type"] = line.split(':', 1)[1].strip()
            elif line.upper().startswith('SEVERITY:'):
                parsed["severity"] = line.split(':', 1)[1].strip()
            elif line.upper().startswith('DESCRIPTION:'):
                parsed["description"] = line.split(':', 1)[1].strip()

        return {"ok": True, "model": model_name, "time": elapsed, "raw": raw, "parsed": parsed}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "model": model_name, "error": "Ollama nije pokrenuta! Otvori Ollama iz Start menija."}
    except Exception as e:
        return {"ok": False, "model": model_name, "error": str(e)}


# ============================================================
# GUI aplikacija
# ============================================================
class DefectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Leather Defect Detection — Multimodalni modeli demo")
        self.root.geometry("1200x780")
        self.root.configure(bg=BG_DARK)
        self.image_path = None
        self.result_cards = []

        self._build_header()
        self._build_body()

    # ---------- Header ----------
    def _build_header(self):
        header = tk.Frame(self.root, bg=BG_PANEL, height=60)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="🧵  Leather Defect Detection", font=("Segoe UI", 15, "bold"),
                 bg=BG_PANEL, fg=TEXT_PRIMARY).pack(side=tk.LEFT, padx=20)
        tk.Label(header, text="Seminarski rad — multimodalni VLM modeli", font=("Segoe UI", 10),
                 bg=BG_PANEL, fg=TEXT_SECOND).pack(side=tk.LEFT)

    # ---------- Body layout ----------
    def _build_body(self):
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        # ---- LEFT PANEL ----
        left = tk.Frame(body, bg=BG_PANEL, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 16))
        left.pack_propagate(False)

        pad = {"padx": 16}

        tk.Label(left, text="UČITAVANJE", font=("Segoe UI", 9, "bold"),
                 bg=BG_PANEL, fg=TEXT_SECOND).pack(anchor="w", pady=(16, 6), **pad)

        self._make_button(left, "📁  Učitaj sliku", ACCENT, ACCENT_HOVER,
                           self.load_image).pack(fill=tk.X, **pad)

        # Image preview card
        preview_card = tk.Frame(left, bg=BG_CARD, highlightbackground=BORDER,
                                 highlightthickness=1)
        preview_card.pack(fill=tk.X, pady=12, **pad)

        self.image_label = tk.Label(preview_card, text="Nema učitane slike",
                                     bg=BG_CARD, fg=TEXT_SECOND, font=("Segoe UI", 9),
                                     width=30, height=11)
        self.image_label.pack(pady=10)

        self.filename_label = tk.Label(preview_card, text="", font=("Segoe UI", 9, "bold"),
                                        bg=BG_CARD, fg=TEXT_PRIMARY)
        self.filename_label.pack(pady=(0, 10))

        tk.Frame(left, bg=BORDER, height=1).pack(fill=tk.X, pady=8, **pad)

        tk.Label(left, text="POZOVI MODEL", font=("Segoe UI", 9, "bold"),
                 bg=BG_PANEL, fg=TEXT_SECOND).pack(anchor="w", pady=(4, 6), **pad)

        self._make_button(left, "⚡  CLIP   ·  ~1s", GREEN, "#3d9468",
                           lambda: self.run_model("clip")).pack(fill=tk.X, pady=3, **pad)
        self._make_button(left, "🌙  Moondream   ·  ~5s", ORANGE, "#cc8f33",
                           lambda: self.run_model("moondream")).pack(fill=tk.X, pady=3, **pad)
        self._make_button(left, "🔮  LLaVA-7B   ·  ~60s", PURPLE, PURPLE_DARK,
                           lambda: self.run_model("llava:7b-v1.6")).pack(fill=tk.X, pady=3, **pad)
        self._make_button(left, "🔮  LLaVA-13B   ·  ~90s", PURPLE_DARK, "#664a96",
                           lambda: self.run_model("llava:13b")).pack(fill=tk.X, pady=3, **pad)

        tk.Frame(left, bg=BORDER, height=1).pack(fill=tk.X, pady=12, **pad)

        self._make_button(left, "🚀  Pokreni SVE modele redom", RED, "#c94f3d",
                           self.run_all_models).pack(fill=tk.X, **pad)

        self.status_label = tk.Label(left, text="", font=("Segoe UI", 9),
                                      bg=BG_PANEL, fg=ACCENT, wraplength=260, justify="left")
        self.status_label.pack(anchor="w", pady=(10, 0), **pad)

        # ---- RIGHT PANEL (results, scrollable) ----
        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(right, text="Rezultati", font=("Segoe UI", 13, "bold"),
                 bg=BG_DARK, fg=TEXT_PRIMARY).pack(anchor="w", pady=(0, 10))

        # Scrollable canvas for result cards
        canvas_frame = tk.Frame(right, bg=BG_DARK)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg=BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.results_container = tk.Frame(self.canvas, bg=BG_DARK)

        self.results_container.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.results_container, anchor="nw", width=820)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.empty_hint = tk.Label(
            self.results_container, text="Učitaj sliku i pozovi model da vidiš rezultate ovdje.",
            font=("Segoe UI", 10), bg=BG_DARK, fg=TEXT_SECOND
        )
        self.empty_hint.pack(pady=40)

    # ---------- Helpers ----------
    def _make_button(self, parent, text, bg, hover_bg, command):
        btn = tk.Button(parent, text=text, font=("Segoe UI", 10, "bold"),
                         bg=bg, fg="white", activebackground=hover_bg, activeforeground="white",
                         relief=tk.FLAT, bd=0, pady=9, cursor="hand2", command=command)
        btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Slike", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")]
        )
        if not path:
            return
        self.image_path = path

        img = Image.open(path).convert("RGB")
        img.thumbnail((230, 230))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

        self.filename_label.config(text=os.path.basename(path))

        # Clear results
        for w in self.results_container.winfo_children():
            w.destroy()
        self.result_cards = []
        tk.Label(self.results_container, text="✓ Slika učitana — odaberi model za analizu.",
                 font=("Segoe UI", 10), bg=BG_DARK, fg=GREEN).pack(pady=20, anchor="w")

    def set_status(self, text):
        self.status_label.config(text=text)
        self.root.update_idletasks()

    # ---------- Result card rendering ----------
    def _add_card(self, result):
        card = tk.Frame(self.results_container, bg=BG_CARD, highlightbackground=BORDER,
                         highlightthickness=1)
        card.pack(fill=tk.X, pady=6, padx=2)

        inner = tk.Frame(card, bg=BG_CARD)
        inner.pack(fill=tk.X, padx=18, pady=14)

        # Header row: model name + badge + time
        header_row = tk.Frame(inner, bg=BG_CARD)
        header_row.pack(fill=tk.X)

        model_colors = {
            "CLIP": GREEN, "moondream": ORANGE,
            "llava:7b-v1.6": PURPLE, "llava:13b": PURPLE_DARK
        }
        color = model_colors.get(result.get("model"), ACCENT)

        dot = tk.Label(header_row, text="●", font=("Segoe UI", 12), bg=BG_CARD, fg=color)
        dot.pack(side=tk.LEFT)
        tk.Label(header_row, text=result.get("model", "?"), font=("Segoe UI", 12, "bold"),
                 bg=BG_CARD, fg=TEXT_PRIMARY).pack(side=tk.LEFT, padx=(6, 0))

        if result.get("time"):
            tk.Label(header_row, text=f"  {result['time']:.0f}s", font=("Segoe UI", 9),
                     bg=BG_CARD, fg=TEXT_SECOND).pack(side=tk.LEFT)

        if not result.get("ok"):
            tk.Label(inner, text=f"⚠  {result.get('error', 'Nepoznata greška')}",
                     font=("Segoe UI", 10), bg=BG_CARD, fg=RED, wraplength=740,
                     justify="left").pack(anchor="w", pady=(10, 0))
            return

        # CLIP-style result
        if result["model"] == "CLIP":
            verdict_text = "ČISTO — bez defekta" if result["is_clean"] else f"DEFEKT  ·  {result['top_label']}"
            verdict_color = GREEN if result["is_clean"] else RED
            verdict_frame = tk.Frame(inner, bg=BG_CARD)
            verdict_frame.pack(fill=tk.X, pady=(12, 10))
            tk.Label(verdict_frame, text=("✓ " if result["is_clean"] else "⚠ ") + verdict_text,
                     font=("Segoe UI", 12, "bold"), bg=BG_CARD, fg=verdict_color).pack(anchor="w")

            tk.Label(inner, text="Top 3 predikcije", font=("Segoe UI", 9, "bold"),
                     bg=BG_CARD, fg=TEXT_SECOND).pack(anchor="w", pady=(4, 6))

            for label, prob in result["top3"]:
                row = tk.Frame(inner, bg=BG_CARD)
                row.pack(fill=tk.X, pady=2)
                tk.Label(row, text=label, font=("Segoe UI", 10), bg=BG_CARD,
                         fg=TEXT_PRIMARY, width=16, anchor="w").pack(side=tk.LEFT)

                bar_bg = tk.Frame(row, bg=BORDER, height=14, width=300)
                bar_bg.pack(side=tk.LEFT, padx=8)
                bar_bg.pack_propagate(False)
                bar_fill_w = max(int(300 * prob), 2)
                tk.Frame(bar_bg, bg=color, height=14, width=bar_fill_w).place(x=0, y=0)

                tk.Label(row, text=f"{prob*100:.1f}%", font=("Segoe UI", 9),
                         bg=BG_CARD, fg=TEXT_SECOND).pack(side=tk.LEFT)

        # Ollama-style (moondream / llava) result
        else:
            p = result["parsed"]
            present = p["defect_present"]
            verdict_color = RED if present else GREEN
            verdict_text = "DEFEKT DETEKTOVAN" if present else "ČISTO — bez defekta"
            verdict_icon = "⚠" if present else "✓"

            tk.Label(inner, text=f"{verdict_icon}  {verdict_text}", font=("Segoe UI", 12, "bold"),
                     bg=BG_CARD, fg=verdict_color).pack(anchor="w", pady=(12, 8))

            grid = tk.Frame(inner, bg=BG_CARD)
            grid.pack(fill=tk.X, pady=(0, 6))

            def field(label, value):
                row = tk.Frame(grid, bg=BG_CARD)
                row.pack(fill=tk.X, pady=2)
                tk.Label(row, text=label, font=("Segoe UI", 9, "bold"), bg=BG_CARD,
                         fg=TEXT_SECOND, width=12, anchor="w").pack(side=tk.LEFT)
                tk.Label(row, text=value or "—", font=("Segoe UI", 9), bg=BG_CARD,
                         fg=TEXT_PRIMARY, anchor="w", wraplength=600, justify="left").pack(side=tk.LEFT)

            field("Tip defekta", p["defect_type"])
            field("Ozbiljnost", p["severity"])
            if p["description"]:
                field("Opis", p["description"])

        self.results_container.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.yview_moveto(1.0)

    def _clear_empty_hint(self):
        for w in self.results_container.winfo_children():
            if isinstance(w, tk.Label):
                w.destroy()

    # ---------- Model execution ----------
    def run_model(self, model_key):
        if not self.image_path:
            self.set_status("⚠ Prvo učitaj sliku!")
            return
        thread = threading.Thread(target=self._run_model_thread, args=(model_key,))
        thread.start()

    def _run_model_thread(self, model_key):
        self.set_status(f"⏳ Pozivam {model_key}...")
        self._clear_empty_hint()
        if model_key == "clip":
            result = classify_with_clip(self.image_path)
        else:
            result = query_ollama(self.image_path, model_key)
        self._add_card(result)
        self.set_status("✅ Gotovo")

    def run_all_models(self):
        if not self.image_path:
            self.set_status("⚠ Prvo učitaj sliku!")
            return
        thread = threading.Thread(target=self._run_all_thread)
        thread.start()

    def _run_all_thread(self):
        self._clear_empty_hint()
        models = ["clip", "moondream", "llava:7b-v1.6", "llava:13b"]
        for m in models:
            self.set_status(f"⏳ Pozivam {m}... (ovo može potrajati)")
            if m == "clip":
                result = classify_with_clip(self.image_path)
            else:
                result = query_ollama(self.image_path, m)
            self._add_card(result)
        self.set_status("✅ Svi modeli završeni")


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('default')
    style.configure("Vertical.TScrollbar", background=BG_PANEL, troughcolor=BG_DARK,
                     bordercolor=BG_DARK, arrowcolor=TEXT_SECOND)
    app = DefectDetectionApp(root)
    root.mainloop()
