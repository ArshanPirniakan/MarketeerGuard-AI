import sys
import json
import re
import threading
import urllib.request
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PLAYBOOK_PATH = Path(__file__).parent / "data" / "compliance_playbook.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"

SYSTEM_PROMPT = (
    "You are MarketeerGuard AI, a contract risk analyzer for freelance marketers and creators. "
    "You analyze contract clauses and explain risks in plain English. "
    "Always respond ONLY with a valid JSON object. No preamble, no markdown, no explanation outside JSON."
)

DARK_BG = "#0d0f14"
CARD_BG = "#13161e"
ACCENT = "#00e5a0"
ACCENT2 = "#ff4f6d"
TEXT_PRIMARY = "#e8eaf0"
TEXT_MUTED = "#5a6070"
BORDER = "#1e2230"
RISK_COLORS = {"HIGH": "#ff4f6d", "MEDIUM": "#ffb347", "LOW": "#00e5a0"}
RISK_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


class DocumentLoader:
    def load(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return self._load_pdf(path)
        elif ext in (".txt", ".md"):
            return Path(path).read_text(encoding="utf-8", errors="ignore")
        else:
            return self._load_pdf(path)

    def _load_pdf(self, path: str) -> str:
        text = self._try_pdfplumber(path)
        if not text.strip():
            text = self._try_pypdf2(path)
        return text

    def _try_pdfplumber(self, path: str) -> str:
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
            return "\n\n".join(pages)
        except Exception:
            return ""

    def _try_pypdf2(self, path: str) -> str:
        try:
            import PyPDF2
            pages = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
            return "\n\n".join(pages)
        except Exception:
            return ""


class ContractChunker:
    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)
        return self._build_chunks(sentences)

    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _build_chunks(self, sentences: List[str]) -> List[str]:
        chunks = []
        current = []
        current_len = 0
        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.chunk_size and current:
                chunks.append(" ".join(current))
                overlap_words = " ".join(current).split()[-self.overlap:]
                current = [" ".join(overlap_words)]
                current_len = len(current[0])
            current.append(sent)
            current_len += sent_len + 1
        if current:
            chunks.append(" ".join(current))
        return [c for c in chunks if len(c.strip()) > 20]


class RetrievalPipeline:
    def __init__(self):
        self.playbook = self._load_playbook()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=8000)
        self._sentence_model = None

    def _load_playbook(self) -> List[Dict]:
        with open(PLAYBOOK_PATH) as f:
            return json.load(f)

    def reload_playbook(self):
        self.playbook = self._load_playbook()

    def _get_sentence_model(self):
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._sentence_model = None
        return self._sentence_model

    def find_risky_chunks(self, chunks: List[str], threshold: str = "MEDIUM") -> List[Dict]:
        if not chunks:
            return []
        all_texts = [r["description"] for r in self.playbook] + chunks
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        except Exception:
            return []
        n_rules = len(self.playbook)
        rule_vecs = tfidf_matrix[:n_rules]
        chunk_vecs = tfidf_matrix[n_rules:]
        tfidf_scores = cosine_similarity(chunk_vecs, rule_vecs)
        semantic_scores = self._semantic_scores(chunks)
        results = []
        seen = set()
        for chunk_idx, chunk in enumerate(chunks):
            for rule_idx, rule in enumerate(self.playbook):
                if RISK_ORDER.get(rule["risk_level"], 0) < RISK_ORDER.get(threshold, 0):
                    continue
                t_score = tfidf_scores[chunk_idx, rule_idx]
                s_score = 0.0
                if semantic_scores is not None:
                    s_score = float(semantic_scores[chunk_idx, rule_idx])
                keyword_hit = self._keyword_hit(chunk, rule.get("keywords", []))
                combined = 0.6 * t_score + 0.4 * s_score + (0.25 if keyword_hit else 0.0)
                if combined > 0.18 or keyword_hit:
                    key = (chunk_idx, rule_idx)
                    if key not in seen:
                        seen.add(key)
                        results.append({"chunk": chunk, "rule": rule, "score": combined})
        results.sort(key=lambda x: (RISK_ORDER.get(x["rule"]["risk_level"], 0), x["score"]), reverse=True)
        return self._deduplicate(results)[:20]

    def _semantic_scores(self, chunks: List[str]):
        model = self._get_sentence_model()
        if model is None:
            return None
        try:
            rule_texts = [r["description"] for r in self.playbook]
            rule_emb = model.encode(rule_texts, normalize_embeddings=True)
            chunk_emb = model.encode(chunks, normalize_embeddings=True)
            return np.dot(chunk_emb, rule_emb.T)
        except Exception:
            return None

    def _keyword_hit(self, chunk: str, keywords: List[str]) -> bool:
        chunk_lower = chunk.lower()
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw.lower()) + r"\b", chunk_lower):
                return True
        return False

    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
        seen_rules = {}
        out = []
        for r in results:
            key = (r["rule"]["name"], r["chunk"][:80])
            if key not in seen_rules:
                seen_rules[key] = True
                out.append(r)
        return out


class PromptController:
    def explain(self, clause: str, rule: Dict) -> Optional[Dict]:
        prompt = self._build_prompt(clause, rule)
        response_text = self._call_ollama(prompt)
        if response_text:
            return self._parse_response(response_text, clause, rule)
        return None

    def _build_prompt(self, clause: str, rule: Dict) -> str:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Compliance Rule: {rule['name']}\n"
            f"Risk Level: {rule['risk_level']}\n"
            f"Rule Description: {rule['description']}\n\n"
            f"Contract Clause:\n\"{clause}\"\n\n"
            "Respond with ONLY this JSON:\n"
            '{"clause": "<exact clause>", "risk_level": "<HIGH|MEDIUM|LOW>", "explanation": "<plain English explanation, max 2 sentences>"}'
        )

    def _call_ollama(self, prompt: str) -> Optional[str]:
        try:
            payload = json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 300},
            }).encode("utf-8")
            req = urllib.request.Request(
                OLLAMA_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get("response", "")
        except Exception:
            return None

    def _parse_response(self, text: str, clause: str, rule: Dict) -> Optional[Dict]:
        text = text.strip()
        try:
            data = json.loads(text)
            return {
                "clause": data.get("clause", clause[:300]),
                "risk_level": data.get("risk_level", rule["risk_level"]),
                "explanation": data.get("explanation", ""),
                "rule": rule["name"],
            }
        except Exception:
            match = re.search(r'"explanation"\s*:\s*"([^"]+)"', text)
            if match:
                return {
                    "clause": clause[:300],
                    "risk_level": rule["risk_level"],
                    "explanation": match.group(1),
                    "rule": rule["name"],
                }
            return None

    def fallback_result(self, clause: str, rule: Dict) -> Dict:
        return {
            "clause": clause[:300],
            "risk_level": rule["risk_level"],
            "explanation": rule.get("description", "Potential compliance issue detected."),
            "rule": rule["name"],
        }


class AuditorEngine:
    def __init__(self, retrieval: RetrievalPipeline, prompt_ctrl: PromptController):
        self.retrieval = retrieval
        self.prompt_ctrl = prompt_ctrl

    def audit(self, chunks: List[str], use_llm: bool = True, threshold: str = "MEDIUM") -> List[Dict]:
        self.retrieval.reload_playbook()
        candidates = self.retrieval.find_risky_chunks(chunks, threshold=threshold)
        results = []
        for candidate in candidates:
            chunk = candidate["chunk"]
            rule = candidate["rule"]
            if use_llm:
                finding = self.prompt_ctrl.explain(chunk, rule)
                if finding is None:
                    finding = self.prompt_ctrl.fallback_result(chunk, rule)
            else:
                finding = self.prompt_ctrl.fallback_result(chunk, rule)
            results.append(finding)
        results.sort(key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x.get("risk_level", "LOW"), 3))
        return results


class MarketeerGuardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MarketeerGuard AI")
        self.root.geometry("1100x780")
        self.root.configure(bg=DARK_BG)
        self.root.minsize(900, 650)
        self.loader = DocumentLoader()
        self.chunker = ContractChunker()
        self.retrieval = RetrievalPipeline()
        self.prompt_ctrl = PromptController()
        self.engine = AuditorEngine(self.retrieval, self.prompt_ctrl)
        self.current_file = None
        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg=DARK_BG, pady=0)
        header.pack(fill="x", padx=0, pady=0)
        title_bar = tk.Frame(header, bg=DARK_BG)
        title_bar.pack(fill="x", padx=36, pady=(28, 0))
        tk.Label(title_bar, text="MarketeerGuard", font=("Courier New", 22, "bold"), fg=ACCENT, bg=DARK_BG).pack(side="left")
        tk.Label(title_bar, text=" AI  //  Contract Auditor", font=("Courier New", 14), fg=TEXT_MUTED, bg=DARK_BG).pack(side="left", padx=(4, 0), pady=(6, 0))
        tk.Label(title_bar, text="PRIVACY-FIRST · LOCAL · OPEN-SOURCE", font=("Courier New", 9), fg=TEXT_MUTED, bg=DARK_BG).pack(side="right", pady=(8, 0))
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=36, pady=(16, 0))
        body = tk.Frame(self.root, bg=DARK_BG)
        body.pack(fill="both", expand=True, padx=36, pady=20)
        left = tk.Frame(body, bg=DARK_BG, width=300)
        left.pack(side="left", fill="y", padx=(0, 16))
        left.pack_propagate(False)
        self._build_left_panel(left)
        right = tk.Frame(body, bg=DARK_BG)
        right.pack(side="left", fill="both", expand=True)
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        tk.Label(parent, text="UPLOAD CONTRACT", font=("Courier New", 9, "bold"), fg=TEXT_MUTED, bg=DARK_BG, anchor="w").pack(fill="x", pady=(0, 8))
        drop_frame = tk.Frame(parent, bg=CARD_BG, relief="flat", bd=0)
        drop_frame.pack(fill="x")
        inner = tk.Frame(drop_frame, bg=CARD_BG, padx=16, pady=24)
        inner.pack(fill="x")
        tk.Label(inner, text="⬆", font=("Courier New", 28), fg=ACCENT, bg=CARD_BG).pack()
        tk.Label(inner, text="Drop PDF here or", font=("Courier New", 10), fg=TEXT_MUTED, bg=CARD_BG).pack(pady=(8, 4))
        self.file_btn = tk.Button(inner, text="Browse File", font=("Courier New", 10, "bold"), fg=DARK_BG, bg=ACCENT, activebackground="#00c988", activeforeground=DARK_BG, relief="flat", padx=20, pady=8, cursor="hand2", command=self._browse_file)
        self.file_btn.pack(pady=(0, 4))
        self.file_label = tk.Label(inner, text="No file selected", font=("Courier New", 9), fg=TEXT_MUTED, bg=CARD_BG, wraplength=240)
        self.file_label.pack()
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=16)
        tk.Label(parent, text="AUDIT OPTIONS", font=("Courier New", 9, "bold"), fg=TEXT_MUTED, bg=DARK_BG, anchor="w").pack(fill="x", pady=(0, 8))
        opts_frame = tk.Frame(parent, bg=CARD_BG, padx=16, pady=16)
        opts_frame.pack(fill="x")
        self.use_llm_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opts_frame, text="Enable LLM Explanations", variable=self.use_llm_var, font=("Courier New", 9), fg=TEXT_PRIMARY, bg=CARD_BG, selectcolor=DARK_BG, activebackground=CARD_BG, activeforeground=TEXT_PRIMARY).pack(anchor="w")
        tk.Label(opts_frame, text="Requires Ollama running locally", font=("Courier New", 8), fg=TEXT_MUTED, bg=CARD_BG).pack(anchor="w", pady=(0, 8))
        tk.Label(opts_frame, text="Risk Threshold:", font=("Courier New", 9), fg=TEXT_PRIMARY, bg=CARD_BG, anchor="w").pack(fill="x")
        self.threshold_var = tk.StringVar(value="MEDIUM")
        for level in ["LOW", "MEDIUM", "HIGH"]:
            tk.Radiobutton(opts_frame, text=level, variable=self.threshold_var, value=level, font=("Courier New", 9), fg=RISK_COLORS[level], bg=CARD_BG, selectcolor=DARK_BG, activebackground=CARD_BG).pack(anchor="w")
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=16)
        self.audit_btn = tk.Button(parent, text="▶  RUN AUDIT", font=("Courier New", 12, "bold"), fg=DARK_BG, bg=ACCENT, activebackground="#00c988", activeforeground=DARK_BG, relief="flat", pady=12, cursor="hand2", state="disabled", command=self._run_audit)
        self.audit_btn.pack(fill="x")
        self.status_label = tk.Label(parent, text="", font=("Courier New", 9), fg=TEXT_MUTED, bg=DARK_BG, wraplength=260, justify="left")
        self.status_label.pack(fill="x", pady=(8, 0))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=16)
        tk.Label(parent, text="PLAYBOOK", font=("Courier New", 9, "bold"), fg=TEXT_MUTED, bg=DARK_BG, anchor="w").pack(fill="x", pady=(0, 8))
        tk.Button(parent, text="Edit Compliance Playbook", font=("Courier New", 9), fg=ACCENT, bg=CARD_BG, activebackground=BORDER, activeforeground=ACCENT, relief="flat", pady=8, cursor="hand2", command=self._open_playbook_editor).pack(fill="x")

    def _build_right_panel(self, parent):
        tabs = tk.Frame(parent, bg=DARK_BG)
        tabs.pack(fill="x")
        self.tab_var = tk.StringVar(value="results")
        for tab_id, tab_label in [("results", "RESULTS"), ("raw", "RAW TEXT"), ("log", "AUDIT LOG")]:
            btn = tk.Button(tabs, text=tab_label, font=("Courier New", 9, "bold"), fg=TEXT_PRIMARY if self.tab_var.get() == tab_id else TEXT_MUTED, bg=CARD_BG if self.tab_var.get() == tab_id else DARK_BG, relief="flat", padx=16, pady=8, cursor="hand2", command=lambda t=tab_id: self._switch_tab(t))
            btn.pack(side="left")
            btn._tab_id = tab_id
        self.tab_buttons = tabs.winfo_children()
        self.content_frame = tk.Frame(parent, bg=DARK_BG)
        self.content_frame.pack(fill="both", expand=True, pady=(4, 0))
        self.results_frame = tk.Frame(self.content_frame, bg=DARK_BG)
        self.raw_frame = tk.Frame(self.content_frame, bg=DARK_BG)
        self.log_frame = tk.Frame(self.content_frame, bg=DARK_BG)
        self.results_text = scrolledtext.ScrolledText(self.results_frame, font=("Courier New", 10), bg=CARD_BG, fg=TEXT_PRIMARY, insertbackground=ACCENT, relief="flat", padx=20, pady=20, state="disabled", cursor="arrow")
        self.results_text.pack(fill="both", expand=True)
        self.raw_text = scrolledtext.ScrolledText(self.raw_frame, font=("Courier New", 9), bg=CARD_BG, fg=TEXT_MUTED, insertbackground=ACCENT, relief="flat", padx=20, pady=20, state="disabled", cursor="arrow")
        self.raw_text.pack(fill="both", expand=True)
        self.log_text = scrolledtext.ScrolledText(self.log_frame, font=("Courier New", 9), bg=CARD_BG, fg=TEXT_MUTED, insertbackground=ACCENT, relief="flat", padx=20, pady=20, state="disabled", cursor="arrow")
        self.log_text.pack(fill="both", expand=True)
        self._configure_tags()
        self._switch_tab("results")
        self._show_welcome()

    def _configure_tags(self):
        self.results_text.tag_configure("heading", foreground=ACCENT, font=("Courier New", 13, "bold"))
        self.results_text.tag_configure("subheading", foreground=TEXT_PRIMARY, font=("Courier New", 11, "bold"))
        self.results_text.tag_configure("HIGH", foreground=RISK_COLORS["HIGH"], font=("Courier New", 10, "bold"))
        self.results_text.tag_configure("MEDIUM", foreground=RISK_COLORS["MEDIUM"], font=("Courier New", 10, "bold"))
        self.results_text.tag_configure("LOW", foreground=RISK_COLORS["LOW"], font=("Courier New", 10, "bold"))
        self.results_text.tag_configure("muted", foreground=TEXT_MUTED)
        self.results_text.tag_configure("normal", foreground=TEXT_PRIMARY)
        self.results_text.tag_configure("clause", foreground="#c8cad8", font=("Courier New", 9))
        self.results_text.tag_configure("separator", foreground=BORDER)
        self.log_text.tag_configure("info", foreground=ACCENT)
        self.log_text.tag_configure("warn", foreground=RISK_COLORS["MEDIUM"])
        self.log_text.tag_configure("error", foreground=RISK_COLORS["HIGH"])

    def _switch_tab(self, tab_id):
        self.tab_var.set(tab_id)
        for frame in [self.results_frame, self.raw_frame, self.log_frame]:
            frame.pack_forget()
        {"results": self.results_frame, "raw": self.raw_frame, "log": self.log_frame}[tab_id].pack(fill="both", expand=True)
        for btn in self.tab_buttons:
            if hasattr(btn, "_tab_id"):
                btn.configure(fg=TEXT_PRIMARY if btn._tab_id == tab_id else TEXT_MUTED, bg=CARD_BG if btn._tab_id == tab_id else DARK_BG)

    def _show_welcome(self):
        self._write_results([
            ("heading", "Welcome to MarketeerGuard AI\n\n"),
            ("muted", "Privacy-first contract auditing for freelancers, creators, and indie agencies.\n"),
            ("muted", "All analysis runs locally — your contracts never leave your machine.\n\n"),
            ("subheading", "How to use:\n"),
            ("normal", "  1. Upload a marketing contract PDF (NDA, Service Agreement, Influencer Brief)\n"),
            ("normal", "  2. Select your risk threshold and options\n"),
            ("normal", "  3. Click RUN AUDIT\n\n"),
            ("muted", "─" * 62 + "\n\n"),
            ("subheading", "What MarketeerGuard detects:\n"),
            ("HIGH", "  ● HIGH   "), ("normal", "Intellectual property transfers, exclusivity traps, indemnification clauses\n"),
            ("MEDIUM", "  ● MEDIUM "), ("normal", "Unlimited revisions, vague deliverables, auto-renewal terms\n"),
            ("LOW", "  ● LOW    "), ("normal", "Payment terms, late fee caps, termination notice periods\n"),
        ])

    def _browse_file(self):
        path = filedialog.askopenfilename(title="Select Marketing Contract PDF", filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if path:
            self.current_file = path
            self.file_label.configure(text=Path(path).name, fg=ACCENT)
            self.audit_btn.configure(state="normal")
            self._log(f"File loaded: {Path(path).name}", "info")

    def _run_audit(self):
        if not self.current_file:
            return
        self.audit_btn.configure(state="disabled", text="⏳ AUDITING...")
        self.status_label.configure(text="Extracting contract text...")
        self._clear_results()
        self._clear_log()
        threading.Thread(target=self._audit_thread, daemon=True).start()

    def _audit_thread(self):
        try:
            self._update_status("Extracting text from PDF...")
            self._log("Starting document extraction...", "info")
            text = self.loader.load(self.current_file)
            if not text.strip():
                self._update_status("Error: Could not extract text from PDF.")
                self._log("Extraction failed: empty text.", "error")
                self._reset_button()
                return
            self._update_raw(text)
            self._log(f"Extracted {len(text)} characters.", "info")
            self._update_status("Chunking contract sections...")
            self._log("Chunking document...", "info")
            chunks = self.chunker.chunk(text)
            self._log(f"Created {len(chunks)} chunks.", "info")
            self._update_status("Running hybrid retrieval pipeline...")
            self._log("Running TF-IDF + semantic retrieval...", "info")
            results = self.engine.audit(chunks, use_llm=self.use_llm_var.get(), threshold=self.threshold_var.get())
            self._log(f"Found {len(results)} flagged clauses.", "info")
            self._update_status(f"Audit complete. {len(results)} issues found.")
            self.root.after(0, lambda: self._render_results(results))
        except Exception as e:
            self._update_status(f"Error: {e}")
            self._log(f"Unhandled error: {e}", "error")
        finally:
            self._reset_button()

    def _render_results(self, results):
        self._clear_results()
        if not results:
            self._write_results([("heading", "✓  No Issues Found\n\n"), ("muted", "No compliance red flags detected at the selected threshold.\n"), ("muted", "This does not constitute legal advice.\n")])
            return
        risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for r in results:
            risk_counts[r.get("risk_level", "LOW")] += 1
        self._write_results([
            ("heading", f"Audit Report  —  {len(results)} Clause(s) Flagged\n\n"),
            ("HIGH", f"  ● {risk_counts['HIGH']} HIGH   "),
            ("MEDIUM", f"  ● {risk_counts['MEDIUM']} MEDIUM   "),
            ("LOW", f"  ● {risk_counts['LOW']} LOW\n\n"),
            ("muted", "─" * 62 + "\n\n"),
        ])
        for i, finding in enumerate(results, 1):
            risk = finding.get("risk_level", "LOW")
            clause = finding.get("clause", "")
            explanation = finding.get("explanation", "")
            rule = finding.get("rule", "")
            self._write_results([(risk, f"[{risk}]  "), ("subheading", f"Finding #{i}\n"), ("muted", f"Rule: {rule}\n"), ("clause", f'"{clause[:280]}{"..." if len(clause) > 280 else ""}"\n')])
            if explanation:
                self._write_results([("normal", f"  ↳ {explanation}\n")])
            self._write_results([("muted", "\n" + "─" * 62 + "\n\n")])

    def _write_results(self, segments):
        def _do():
            self.results_text.configure(state="normal")
            for tag, text in segments:
                self.results_text.insert("end", text, tag)
            self.results_text.configure(state="disabled")
            self.results_text.see("end")
        self.root.after(0, _do)

    def _clear_results(self):
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _update_raw(self, text):
        def _do():
            self.raw_text.configure(state="normal")
            self.raw_text.delete("1.0", "end")
            self.raw_text.insert("end", text)
            self.raw_text.configure(state="disabled")
        self.root.after(0, _do)

    def _log(self, msg, level="info"):
        def _do():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", f"[{level.upper()}] {msg}\n", level)
            self.log_text.configure(state="disabled")
            self.log_text.see("end")
        self.root.after(0, _do)

    def _update_status(self, msg):
        self.root.after(0, lambda: self.status_label.configure(text=msg))

    def _reset_button(self):
        self.root.after(0, lambda: self.audit_btn.configure(state="normal", text="▶  RUN AUDIT"))

    def _open_playbook_editor(self):
        win = tk.Toplevel(self.root)
        win.title("Compliance Playbook Editor")
        win.geometry("700x550")
        win.configure(bg=DARK_BG)
        tk.Label(win, text="COMPLIANCE PLAYBOOK", font=("Courier New", 11, "bold"), fg=ACCENT, bg=DARK_BG).pack(pady=(16, 4))
        tk.Label(win, text="Edit JSON to customize risk rules", font=("Courier New", 9), fg=TEXT_MUTED, bg=DARK_BG).pack()
        editor = scrolledtext.ScrolledText(win, font=("Courier New", 10), bg=CARD_BG, fg=TEXT_PRIMARY, insertbackground=ACCENT, relief="flat", padx=16, pady=16)
        editor.pack(fill="both", expand=True, padx=16, pady=12)
        try:
            with open(PLAYBOOK_PATH) as f:
                editor.insert("1.0", json.dumps(json.load(f), indent=2))
        except Exception as e:
            editor.insert("1.0", f"Error loading playbook: {e}")
        def save():
            try:
                data = json.loads(editor.get("1.0", "end"))
                with open(PLAYBOOK_PATH, "w") as f:
                    json.dump(data, f, indent=2)
                win.destroy()
                self._log("Playbook saved successfully.", "info")
            except Exception as ex:
                messagebox.showerror("Save Error", str(ex))
        tk.Button(win, text="Save & Close", font=("Courier New", 10, "bold"), fg=DARK_BG, bg=ACCENT, relief="flat", pady=8, cursor="hand2", command=save).pack(pady=(0, 16))


def main():
    root = tk.Tk()
    MarketeerGuardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
