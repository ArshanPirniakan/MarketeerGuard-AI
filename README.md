# MarketeerGuard AI

> Local, privacy-first contract auditor for freelancers, creators, and indie agencies.

MarketeerGuard reads your marketing contracts (NDAs, service agreements, influencer briefs) and flags risky clauses before you sign. Everything runs on your machine — your documents never touch a cloud API.

---

## What it detects

| Risk | Examples |
|------|----------|
| 🔴 HIGH | Perpetual IP transfers, exclusivity traps, broad indemnification, unlimited revisions |
| 🟡 MEDIUM | Auto-renewal clauses, vague deliverables, forced arbitration, no kill fee |
| 🟢 LOW | Extended payment terms (Net 60/90), short termination notice, portfolio restrictions |

---

## Setup

```bash
pip install -r requirements.txt
python marketeerguard.py
```

**Optional — LLM explanations via Ollama:**
```bash
ollama pull phi3
```
Ollama must be running locally at `http://localhost:11434`. If it's not running, the app falls back to rule-based descriptions automatically.

---

## How it works

1. Upload a contract PDF
2. The app chunks the text and runs a hybrid search pipeline (TF-IDF keyword matching + semantic sentence embeddings)
3. Flagged clauses are matched against a customizable compliance playbook
4. Optionally, a local Phi-3 model explains each finding in plain English
5. Results are displayed by risk level with clause excerpts and explanations

---

## Customizing rules

Click **Edit Compliance Playbook** in the app, or edit `data/compliance_playbook.json` directly. Each rule has:

```json
{
  "name": "Perpetual IP Transfer",
  "risk_level": "HIGH",
  "description": "...",
  "keywords": ["perpetuity", "irrevocable", "work for hire"]
}
```

---

## Stack

- `pdfplumber` / `PyPDF2` — PDF extraction
- `scikit-learn` — TF-IDF vectorization
- `sentence-transformers` — semantic embeddings (`all-MiniLM-L6-v2`)
- `Ollama` + Phi-3 — local LLM explanations
- `tkinter` — desktop UI

---

> **Disclaimer:** MarketeerGuard is not a substitute for legal advice. Always consult a qualified lawyer for binding contracts.
