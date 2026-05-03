from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pdfplumber, io, os, logging, re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USE_MOCK   = os.getenv("USE_MOCK", "true").lower() == "true"
# FIX #3: Default BASE_MODEL now matches adapter_config.json (was 1.5B, must be 3B)
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH", "/content/drive/MyDrive/finetuned_qwen")
OCR_LANG   = os.getenv("OCR_LANG", "eng+fra")


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class AnalyzeRequest(BaseModel):
    cv_text: str
    job_description: str

class AnalyzeResponse(BaseModel):
    score: str
    strengths: list[str]
    weaknesses: list[str]
    verdict: str
    extraction_method: str
    raw_output: str


# ═══════════════════════════════════════════════════════════════════════════════
# PDF EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_pdfplumber(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            # extract_text(layout=True) preserves word spacing.
            # x_tolerance/y_tolerance tune how aggressively chars are merged.
            t = page.extract_text(layout=True, x_tolerance=3, y_tolerance=3)
            if not t:
                # fallback: default extraction
                t = page.extract_text()
            if t:
                text += t + "\n"
    # Collapse runs of 3+ spaces (layout mode can leave wide gaps) to single space
    text = re.sub(r' {3,}', ' ', text)
    return text.strip()

def smart_extract(file_bytes: bytes):
    text = extract_text_pdfplumber(file_bytes)
    if len(text) >= 50:
        return text, "text"
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(file_bytes, dpi=300)
        ocr_text = "\n".join(
            pytesseract.image_to_string(img, lang=OCR_LANG) for img in images
        ).strip()
        return ocr_text, "ocr"
    except Exception as e:
        raise HTTPException(422, f"OCR failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

_JD_BOILERPLATE = [
    r"^job (description|title|summary|overview|posting)",
    r"^(position|role)\s*:",
    r"^we are (seeking|looking for|hiring|searching)",
    r"^about (us|the (role|position|company|team|opportunity))",
    r"^(the )?ideal candidate (will|should|must|would)",
    r"^(to )?join our (team|company|organization)",
    r"^(a )?(dedicated|passionate|motivated|talented) and",
    r"^(responsibilities|duties|tasks)\s*(include|:)?$",
    r"^(minimum |required? )?(qualifications?|requirements?)\s*:?$",
    r"^(preferred experience|nice to have)\s*:?$",
    r"^what (we offer|you('ll| will) (do|bring|get))",
    r"^(please )?(send|submit|apply|email|contact)",
    r"^(equal opportunity|eeo|diversity)",
    r"^(salary|compensation|benefits?|perks?)\s*:",
    r"^(location|remote|hybrid|on-?site)\s*:",
    r"^\d+\s*$",
    r"^[-–—=*#]{2,}$",
]

_CV_BOILERPLATE = [
    r"^(curriculum vitae|cv|resume|profile)$",
    r"^(dear|to whom it may concern|sincerely|regards|yours)",
    r"^(phone|email|address|linkedin|github|tel|mob|cell|fax)\s*:",
    r"^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$",
    r"^\+?\d[\d\s\(\)\-]{7,}$",
    r"^(january|february|march|april|may|june|july|august|"
     r"september|october|november|december)\s+\d{4}",
    r"^(jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}",
    r"^[-–—=*#]{2,}$",
    r"^\d+\s*$",
]

def _clean_text(text: str, patterns: list) -> str:
    cleaned = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 4:
            continue
        if any(re.search(p, line.lower()) for p in patterns):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def clean_job_description(text: str) -> str:
    return _clean_text(text, _JD_BOILERPLATE)

def clean_cv(text: str) -> str:
    return _clean_text(text, _CV_BOILERPLATE)


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

_DOMAIN_SIGNALS = {
    "ai_ml": {
        "machine learning", "deep learning", "neural network", "llm", "nlp",
        "pytorch", "tensorflow", "scikit-learn", "transformers", "fine-tuning",
        "qlora", "lora", "bert", "gpt", "cnn", "rnn", "lstm", "xgboost",
        "huggingface", "mlops", "airflow", "docker", "kubernetes", "fastapi",
        "big data", "data pipeline", "boto3", "minio", "postgresql", "rabbitmq",
        "computer vision", "object detection", "embedding", "vector database",
        "streamlit", "pyngrok", "fedora", "apache", "orchestration",
    },
    "healthcare": {
        "patient", "clinical", "nursing", "medical", "icu", "hospital",
        "diagnosis", "treatment", "medication", "surgery", "physician",
        "ehr", "epic", "hipaa", "vital signs", "triage", "radiology",
        "pharmacology", "anatomy", "physiology", "acls", "bls", "cpr",
        "wound care", "catheter", "ventilator", "infusion", "bedside",
        "registered nurse", "healthcare", "health care",
    },
    "software": {
        "javascript", "typescript", "react", "vue", "angular", "node",
        "java", "spring", "microservices", "rest api", "graphql",
        "ci/cd", "git", "agile", "scrum", "devops", "terraform",
    },
    "finance": {
        "accounting", "audit", "balance sheet", "financial statement",
        "ifrs", "gaap", "tax", "budget", "forecast", "equity", "portfolio",
        "risk management", "compliance", "regulatory",
    },
}

def detect_domain(text: str) -> str:
    text_lower = text.lower()
    scores = {
        domain: sum(1 for s in signals if s in text_lower)
        for domain, signals in _DOMAIN_SIGNALS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"


# ═══════════════════════════════════════════════════════════════════════════════
# STOPWORDS
# ═══════════════════════════════════════════════════════════════════════════════

STOPWORDS = {
    "the","and","or","of","to","a","an","in","for","on","with","at","by",
    "from","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall",
    "that","this","these","those","it","its","we","our","you","your","they",
    "their","he","she","his","her","i","my","me","us","as","if","but","not",
    "also","than","then","so","yet","both","either","each","all","any","more",
    "most","other","such","no","nor","too","very","just","over","under","out",
    "up","down","into","about","through","during","before","after","above",
    "between","own","same","can","must","need","use","using","used","including",
    "include","within","without","across","per","well","good","great","strong",
    "excellent","ability","experience","work","working","years","year","minimum",
    "required","requirements","preferred","knowledge","skills","skill","abilities",
    "role","position","job","candidate","team","company","organization","environment",
    "field","area","responsible","provide","support","ensure","maintain","manage",
    "develop","assist","perform","conduct","implement","demonstrate","proven",
    "proficiency","familiarity","understanding","exposure","background","related",
    "equivalent","relevant","various","multiple","different","following","including",
    "technical","innovative","scalable","robust","seamless","competitive",
    "python","english","french","arabic","communication","analytical",
    "problem","solving","detail","oriented","motivated","collaborative",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_bullet_requirements(text: str) -> list[str]:
    results = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(r'^[•·●▪▸▶\-\*\–\—\→✓►]', line) or re.match(r'^\d+[\.\)]\s', line):
            clean = re.sub(r'^[•·●▪▸▶\-\*\–\—\→✓►\d\.\)]\s*', '', line).strip()
            if len(clean) > 6:
                results.append(clean.lower())
    return results

def extract_noun_phrases(text: str, min_len: int = 3, max_words: int = 4) -> list[str]:
    text = re.sub(r"[•·●▪▸\-]", " ", text)
    text = re.sub(r"[^\w\s/+#.]", " ", text)
    words = text.split()
    candidates, i = [], 0
    while i < len(words):
        for n in range(max_words, 0, -1):
            chunk = words[i:i+n]
            phrase = " ".join(chunk).strip().lower()
            tokens = phrase.split()
            meaningful = [
                t for t in tokens
                if t not in STOPWORDS and len(t) >= min_len and not t.isdigit()
            ]
            if len(meaningful) >= max(1, n // 2) and len(phrase) >= min_len:
                candidates.append(phrase)
                i += n
                break
        else:
            i += 1
    seen, result = set(), []
    for c in candidates:
        if c not in seen and c not in STOPWORDS:
            seen.add(c)
            result.append(c)
    return result

def extract_skills(text: str) -> list[str]:
    bullets = extract_bullet_requirements(text)
    if len(bullets) >= 3:
        noun_phrases = extract_noun_phrases(text)
        bullet_blob = " ".join(bullets)
        extra = [p for p in noun_phrases if p not in bullet_blob and len(p.split()) <= 3]
        return bullets + extra[:10]
    return extract_noun_phrases(text)


# ═══════════════════════════════════════════════════════════════════════════════
# MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def terms_overlap(cv_terms: list, job_terms: list,
                  cv_domain: str, jd_domain: str):
    cross_domain = (
        cv_domain != jd_domain and
        cv_domain != "general" and
        jd_domain != "general"
    )
    cv_blob = " ".join(cv_terms).lower()
    matched, missing = [], []

    for jt in job_terms:
        jt_core = jt.lower().strip()
        if len(jt_core) < 4 or jt_core in STOPWORDS:
            continue

        found = (
            jt_core in cv_blob or
            any(jt_core in ct or ct in jt_core for ct in cv_terms)
        )

        if found:
            if cross_domain:
                jd_signals = _DOMAIN_SIGNALS.get(jd_domain, set())
                if any(sig in jt_core or jt_core in sig for sig in jd_signals):
                    matched.append(jt)
                else:
                    missing.append(jt)
            else:
                matched.append(jt)
        else:
            missing.append(jt)

    return matched, missing


# ═══════════════════════════════════════════════════════════════════════════════
# SMART MOCK
# ═══════════════════════════════════════════════════════════════════════════════

def smart_mock(cv_text: str, job_description: str) -> dict:
    cv_clean  = clean_cv(cv_text)
    jd_clean  = clean_job_description(job_description)

    cv_domain = detect_domain(cv_clean)
    jd_domain = detect_domain(jd_clean)
    logger.info(f"CV domain: {cv_domain} | JD domain: {jd_domain}")

    cv_terms  = extract_skills(cv_clean)
    job_terms = extract_skills(jd_clean)
    logger.info(f"CV terms ({len(cv_terms)}): {cv_terms[:6]}")
    logger.info(f"JD terms ({len(job_terms)}): {job_terms[:6]}")

    matched, missing = terms_overlap(cv_terms, job_terms, cv_domain, jd_domain)

    cross_domain_penalty = 0
    if cv_domain != jd_domain and cv_domain != "general" and jd_domain != "general":
        cross_domain_penalty = 30

    if job_terms:
        raw_score = (len(matched) / max(len(job_terms), 1)) * 100
        score = max(5, min(int(raw_score) - cross_domain_penalty, 95))
    else:
        score = max(5, min(40 + len(cv_terms) - cross_domain_penalty, 75))

    strengths = []
    if matched:
        for i in range(0, min(len(matched), 9), 3):
            chunk = matched[i:i+3]
            label = "Matches job requirement" if i == 0 else "Also matches"
            strengths.append(f"{label}: {', '.join(chunk)}")
    extra = [t for t in cv_terms
             if t not in " ".join(job_terms) and len(t.split()) <= 3][:3]
    if extra:
        strengths.append(f"Additional background (not required): {', '.join(extra)}")
    if not strengths:
        strengths = ["Candidate has a general professional background"]

    weaknesses = []
    if cross_domain_penalty:
        weaknesses.append(
            f"Domain mismatch: CV is {cv_domain.replace('_', '/')} — "
            f"role requires {jd_domain.replace('_', '/')} expertise"
        )
    if missing:
        for i in range(0, min(len(missing), 9), 3):
            chunk = missing[i:i+3]
            label = "Missing key requirement" if i == 0 else "Also missing"
            weaknesses.append(f"{label}: {', '.join(chunk)}")
    if not weaknesses:
        weaknesses = ["No significant gaps identified"]

    if score >= 70:
        verdict = "Candidate recommended for interview"
    elif score >= 45:
        verdict = "Candidate worth considering — some gaps present"
    else:
        verdict = "Candidate does not meet the key requirements for this role"

    raw = (
        f"SCORE: {score}%\n"
        f"CV DOMAIN: {cv_domain} | JD DOMAIN: {jd_domain}\n"
        f"MATCHED ({len(matched)}): {', '.join(matched) or 'none'}\n"
        f"MISSING ({len(missing)}): {', '.join(missing) or 'none'}\n\n"
        "STRENGTHS:\n" + "\n".join(f"- {s}" for s in strengths) +
        "\n\nWEAKNESSES:\n" + "\n".join(f"- {w}" for w in weaknesses) +
        f"\n\nVERDICT:\n{verdict}"
    )

    return {
        "score": f"{score}%",
        "strengths": strengths,
        "weaknesses": weaknesses,
        "verdict": verdict,
        "raw_output": raw,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# REAL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

tokenizer, model = None, None

def load_model():
    global tokenizer, model, USE_MOCK
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    use_gpu = torch.cuda.is_available()
    logger.info(f"Device: {'GPU' if use_gpu else 'CPU'}")

    logger.info(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Memory strategy ───────────────────────────────────────────────────────
    # GPU  → float16, full speed
    # CPU  → 4-bit quantization via bitsandbytes (cuts ~6 GB → ~2 GB RAM)
    #         bitsandbytes CPU support requires version >= 0.43.0
    if use_gpu:
        logger.info("Loading in float16 on GPU")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        logger.info("No GPU — loading in 4-bit quantization on CPU (needs ~2 GB RAM)")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32,   # CPU must use float32
                bnb_4bit_use_double_quant=False,         # saves a bit more RAM
                bnb_4bit_quant_type="nf4",
            )
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            # bitsandbytes sometimes doesn't support CPU quantization on older versions
            logger.warning(f"4-bit loading failed ({e}). Trying float32 with aggressive offloading...")
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                offload_folder="/tmp/offload",   # offload layers to disk if needed
                offload_state_dict=True,
            )

    logger.info(f"Attaching LoRA adapter from {MODEL_PATH}...")
    model = PeftModel.from_pretrained(base, MODEL_PATH)
    model.eval()
    logger.info("Model ready.")


# ── build_prompt ──────────────────────────────────────────────────────────────
# Uses Qwen's ChatML template (matches training) + a few-shot example so the
# model sees exactly what "correct" output looks like before generating.
def build_prompt(cv_text: str, job_description: str) -> str:
    cv_clean  = clean_cv(cv_text)
    jd_clean  = clean_job_description(job_description)
    cv_skills = extract_skills(cv_clean)
    jd_skills = extract_skills(jd_clean)
    cv_domain = detect_domain(cv_clean)
    jd_domain = detect_domain(jd_clean)
    matched, missing = terms_overlap(cv_skills, jd_skills, cv_domain, jd_domain)

    hint_lines = []
    if matched:
        hint_lines.append(f"Skills present in CV: {', '.join(matched[:6])}")
    if missing:
        hint_lines.append(f"Skills absent from CV: {', '.join(missing[:6])}")
    hint = ("\n" + "\n".join(hint_lines)) if hint_lines else ""

    system_msg = (
        "You are an HR recruiter. "
        "Output ONLY the structured block below — no extra prose, no explanations.\n\n"
        "### OUTPUT FORMAT (follow exactly)\n"
        "SCORE: <integer 0-100>\n"
        "STRENGTHS:\n"
        "- <one short phrase>\n"
        "- <one short phrase>\n"
        "- <one short phrase>\n"
        "WEAKNESSES:\n"
        "- <one short phrase>\n"
        "- <one short phrase>\n"
        "VERDICT: <one sentence max>\n\n"
        "### EXAMPLE\n"
        "SCORE: 72\n"
        "STRENGTHS:\n"
        "- 4 years Python and FastAPI\n"
        "- PostgreSQL experience matches requirement\n"
        "- REST API design background\n"
        "WEAKNESSES:\n"
        "- No Docker or CI/CD mentioned\n"
        "- Missing cloud platform experience\n"
        "VERDICT: Strong backend candidate, recommend interview to assess DevOps gaps."
    )

    user_msg = (
        f"JOB DESCRIPTION:\n{jd_clean[:600]}\n\n"
        f"CV:\n{cv_clean[:900]}\n"
        f"{hint}\n\n"
        "Now produce the structured output:"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ── generate_analysis ─────────────────────────────────────────────────────────
# Hard-stops generation as soon as VERDICT line ends — prevents the runaway
# prose the model produces when it doesn't see a natural stopping point.
def _make_stop_ids(words: list[str]) -> list[int]:
    """Return token IDs for a list of stop strings (best-effort)."""
    ids = []
    for w in words:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if enc:
            ids.append(enc[0])
    return list(set(ids))


def generate_analysis(cv_text: str, job_description: str) -> str:
    import torch
    prompt = build_prompt(cv_text, job_description)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Additional EOS: stop on <|im_end|> token AND on a blank line after VERDICT
    extra_eos = _make_stop_ids(["<|im_end|>", "<|endoftext|>"])
    all_eos = list({tokenizer.eos_token_id} | set(extra_eos))

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120 if not torch.cuda.is_available() else 180,
            do_sample=False,
            repetition_penalty=1.5,
            no_repeat_ngram_size=5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=all_eos,
        )

    raw = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    # Hard-cut: drop everything after the VERDICT line ends
    raw = _truncate_after_verdict(raw)
    logger.info(f"RAW MODEL OUTPUT:\n{raw}")
    return raw


def _truncate_after_verdict(text: str) -> str:
    """
    Cut the raw output right after the VERDICT sentence ends.
    Works on both multi-line and single-line model output.
    """
    m = re.search(r'VERD\w*\s*:', text, re.IGNORECASE)
    if not m:
        return text
    # Find the first sentence-ending punctuation after VERDICT:
    after_verdict = text[m.end():]
    stop = re.search(r'[.!?]', after_verdict)
    if stop:
        return text[:m.end() + stop.end()].strip()
    # No punctuation found — just take 200 chars after VERDICT:
    return text[:m.end() + 200].strip()


# ── parse_output ──────────────────────────────────────────────────────────────
# Strategy: find each keyword's position in the raw string, slice between them.
# This works even when the model outputs everything on a single line.
#
# The model consistently produces (with typos tolerated):
#   "<number> STRENGHTHS: - item ... WEAKNESSES: - item ... VERDICT: sentence"
#
def parse_output(raw: str, cv_text: str = "", job_description: str = "") -> dict:
    result = {
        "score": "N/A", "strengths": [], "weaknesses": [],
        "verdict": "N/A", "raw_output": raw,
    }

    # ── Step 0: collapse newlines so everything is one searchable string ──────
    # The model sometimes wraps mid-bullet onto the next line. Joining with a
    # space lets the keyword search and bullet splitter work on a flat string.
    flat = " ".join(raw.split())   # collapse ALL whitespace to single spaces

    # ── Locate each section keyword ───────────────────────────────────────────
    # Patterns are intentionally loose to absorb typos:
    #   STRENGHTHS / STRENGTHS / STRENGHTS  →  STREN\w+\s*:
    #   WEAKNESSES / WEAKERESONS / WEAKERES:S / WEAKERSS  →  WEAK\w*\s*:
    #   VERDICT / VEREDICIT / VERNEDICIT     →  VER\w*\s*:
    kw = {
        "score":      re.search(r'SCORE\s*:+\s*',    flat, re.IGNORECASE),
        "strengths":  re.search(r'STREN\w+\s*:+\s*', flat, re.IGNORECASE),
        "weaknesses": re.search(r'WEAK\w*\s*:+\s*',  flat, re.IGNORECASE),
        "verdict":    re.search(r'VER[EI]\w*\s*:+\s*|VERDICT\s*:+\s*', flat, re.IGNORECASE),
    }

    logger.info(f"Section positions — "
                f"score:{kw['score'] and kw['score'].start()} "
                f"strengths:{kw['strengths'] and kw['strengths'].start()} "
                f"weaknesses:{kw['weaknesses'] and kw['weaknesses'].start()} "
                f"verdict:{kw['verdict'] and kw['verdict'].start()}")

    def _slice(start_m, end_m) -> str:
        if start_m is None:
            return ""
        start = start_m.end()
        end   = end_m.start() if end_m else len(flat)
        return flat[start:end].strip()

    # ── SCORE ─────────────────────────────────────────────────────────────────
    score_text = _slice(kw["score"], kw["strengths"])
    if not score_text and kw["strengths"]:
        score_text = flat[: kw["strengths"].start()]
    m = re.search(r'(\d{1,3})', score_text)
    if m:
        result["score"] = f"{m.group(1)}%"

    # ── STRENGTHS ─────────────────────────────────────────────────────────────
    result["strengths"] = _extract_items(_slice(kw["strengths"], kw["weaknesses"]))

    # ── WEAKNESSES ────────────────────────────────────────────────────────────
    result["weaknesses"] = _extract_items(_slice(kw["weaknesses"], kw["verdict"]))

    # ── VERDICT ───────────────────────────────────────────────────────────────
    verdict_text = _slice(kw["verdict"], None)
    if verdict_text:
        first = re.split(r'(?<=[.!?])\s', verdict_text)[0].strip()
        result["verdict"] = first[:250]

    # ── Domain-awareness override ─────────────────────────────────────────────
    if cv_text and job_description:
        cv_domain = detect_domain(clean_cv(cv_text))
        jd_domain = detect_domain(clean_job_description(job_description))
        cross_domain = (
            cv_domain != jd_domain and
            cv_domain != "general" and
            jd_domain != "general"
        )
        if cross_domain:
            try:
                raw_pct = int(result["score"].replace("%", ""))
            except ValueError:
                raw_pct = 50
            penalised = max(5, raw_pct - 30)
            logger.info(f"Domain mismatch ({cv_domain}→{jd_domain}): {raw_pct}%→{penalised}%")
            result["score"] = f"{penalised}%"
            domain_msg = (
                f"Domain mismatch: CV is {cv_domain.replace('_','/')} background, "
                f"role requires {jd_domain.replace('_','/')} expertise."
            )
            result["weaknesses"] = [domain_msg] + result["weaknesses"]

    # ── Score fallback ────────────────────────────────────────────────────────
    if result["score"] == "N/A":
        logger.warning("No score — using rule-based fallback")
        mock = smart_mock(cv_text, job_description)
        result["score"] = mock["score"]
        if not result["strengths"]:  result["strengths"]  = mock["strengths"]
        if not result["weaknesses"]: result["weaknesses"] = mock["weaknesses"]

    # ── Verdict fallback ──────────────────────────────────────────────────────
    if result["verdict"] == "N/A":
        try:
            pct = int(result["score"].replace("%", ""))
        except ValueError:
            pct = 0
        if pct >= 70:
            result["verdict"] = "Candidate recommended for interview"
        elif pct >= 45:
            result["verdict"] = "Candidate worth considering — some gaps present"
        else:
            result["verdict"] = "Candidate does not meet the key requirements"

    # ── Guarantee non-empty lists ─────────────────────────────────────────────
    if not result["strengths"]:
        result["strengths"] = ["No specific strengths extracted"]
    if not result["weaknesses"]:
        result["weaknesses"] = ["No specific gaps extracted"]

    return result


def _extract_items(text: str) -> list[str]:
    if not text:
        return []

    items = []

    # On a flat (newline-collapsed) string, bullets appear as:
    #   "- item one - item two - item three"
    # Split on " - " or " • " preceded by at least one word character
    parts = re.split(r'(?<=\w)\s+[-•]\s+', text)

    # The first part may or may not start with a dash — strip it
    parts = [re.sub(r'^[-•]\s*', '', p).strip() for p in parts]

    for p in parts:
        clean = _trim_item(p)
        if clean:
            items.append(clean)

    # Fallback: no bullet separators found — treat whole text as one item
    if not items:
        clean = _trim_item(text)
        if clean:
            items.append(clean)

    return items[:4]


def _trim_item(text: str) -> str:
    """
    Trim a raw item string to a short, clean phrase:
    1. Strip leading/trailing whitespace and bullet characters
    2. Cut at the first sentence-ending punctuation
    3. Hard cap at 120 characters
    4. Return empty string if result is too short to be meaningful
    """
    text = text.strip().lstrip("-•· \t")
    # Strip lone letter + dash/space left by regex boundary artifacts e.g. "S -No mention"
    text = re.sub(r'^[A-Z]\s*[-–]\s*', '', text).strip()

    # Cut at first full stop, exclamation, or question mark
    m = re.search(r'[.!?]', text)
    if m:
        text = text[:m.start() + 1]   # include the punctuation

    text = text.strip()

    # Hard cap
    if len(text) > 120:
        # Try to cut at last space before 120 to avoid mid-word cuts
        cut = text[:120].rsplit(' ', 1)[0]
        text = cut.rstrip('.,;:') + '…'

    # Reject if too short or just punctuation
    if len(text) < 8 or re.fullmatch(r'[\W\d\s]+', text):
        return ""

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(cv_text: str, job_description: str, method: str = "text") -> dict:
    if USE_MOCK:
        result = smart_mock(cv_text, job_description)
    else:
        raw    = generate_analysis(cv_text, job_description)
        # Stash cv/jd so parse_output's score fallback can call smart_mock
        result = parse_output(raw, cv_text=cv_text, job_description=job_description)
    result["extraction_method"] = method
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app):
    if not USE_MOCK:
        load_model()   # may flip USE_MOCK=True if no GPU
    mode = "MOCK (rule-based)" if USE_MOCK else "REAL MODEL (GPU)"
    logger.info(f"Server starting in {mode} mode")
    yield

app = FastAPI(title="HR CV Analyzer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"], expose_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    analysis: dict


@app.post("/chat")
def chat(req: ChatRequest):
    """Answer a free-form HR question about a candidate using the analysis context."""
    a = req.analysis
    score_int = int(str(a.get("score", "0")).replace("%", "") or 0)

    context = (
        f"CV Analysis:\n"
        f"Score: {a.get('score', 'N/A')}\n"
        f"Verdict: {a.get('verdict', 'N/A')}\n"
        f"Strengths: {'; '.join(a.get('strengths', []))}\n"
        f"Weaknesses: {'; '.join(a.get('weaknesses', []))}\n"
        f"Raw model output: {str(a.get('raw_output', ''))[:600]}\n\n"
        f"HR Question: {req.question}"
    )

    if USE_MOCK:
        # Rule-based answers for mock mode — much richer than the frontend version
        q = req.question.lower()
        strengths = a.get("strengths", [])
        weaknesses = a.get("weaknesses", [])

        if any(w in q for w in ["strength", "good", "qualif", "bring", "has", "match"]):
            bullets = "\n".join(f"• {s}" for s in strengths) if strengths else "No specific strengths identified."
            return {"answer": f"Here is what the candidate brings to the role:\n\n{bullets}"}

        elif any(w in q for w in ["weak", "miss", "lack", "gap", "improv", "need"]):
            bullets = "\n".join(f"• {w}" for w in weaknesses) if weaknesses else "No significant gaps identified."
            return {"answer": f"Key gaps and areas for improvement:\n\n{bullets}"}

        elif any(w in q for w in ["hire", "interview", "invite", "recommend", "should we", "suitable"]):
            if score_int >= 70:
                rec = "Yes — recommend for interview."
                detail = f"Score of {a.get('score')} indicates strong alignment.\n\nTop reasons:\n" + "\n".join(f"• {s}" for s in strengths[:3])
            elif score_int >= 45:
                rec = "Conditional — consider a screening call first."
                detail = f"Score of {a.get('score')} shows partial fit.\n\nKey gaps to probe:\n" + "\n".join(f"• {w}" for w in weaknesses[:3])
            else:
                rec = "No — candidate does not meet core requirements."
                detail = f"Score of {a.get('score')} indicates significant gaps:\n" + "\n".join(f"• {w}" for w in weaknesses[:3])
            return {"answer": f"{rec}\n\n{detail}"}

        elif any(w in q for w in ["next", "step", "action", "now", "do", "plan"]):
            if score_int >= 70:
                steps = "1. Schedule a technical interview\n2. Prepare questions around: " + ", ".join(strengths[:2]) + "\n3. Check references"
            elif score_int >= 45:
                steps = "1. Schedule a 30-min screening call\n2. Ask candidate to address:\n" + "\n".join(f"   • {w}" for w in weaknesses[:2]) + "\n3. Reassess after call"
            else:
                steps = "1. Send a polite rejection\n2. Keep CV on file for future roles\n3. Key unmet requirements:\n" + "\n".join(f"   • {w}" for w in weaknesses[:2])
            return {"answer": f"Recommended next steps:\n\n{steps}"}

        elif any(w in q for w in ["score", "percent", "rate", "mark"]):
            return {"answer": f"The candidate scored {a.get('score')}.\n\n{a.get('verdict', '')}"}

        else:
            # Generic: summarise the full picture
            s_bullets = "\n".join(f"• {s}" for s in strengths)
            w_bullets = "\n".join(f"• {w}" for w in weaknesses)
            return {"answer": f"Overall assessment — Score: {a.get('score')}\n{a.get('verdict', '')}\n\nStrengths:\n{s_bullets}\n\nGaps:\n{w_bullets}"}
    else:
        # Real model — ask a focused single-turn question
        if tokenizer is None or model is None:
            return {"answer": "Model not loaded."}
        import torch
        system = (
            "You are an expert HR assistant. You have analysed a candidate's CV. "
            "Answer the recruiter's question in 3-5 sentences max. Be specific and direct."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": context},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                repetition_penalty=1.4,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        # Cut at first double-newline or after 3 sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        answer = " ".join(sentences[:4]).strip()
        return {"answer": answer or "I could not generate an answer for this question."}


@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    p = "/content/index.html"
    if not os.path.exists(p):
        raise HTTPException(404, "index.html not found at /content/index.html")
    with open(p, "r", encoding="utf-8") as f:
        html = f.read()

    # ── Patch sendChat to call /chat instead of the old /analyze fallback ─────
    # This runs every request so no separate file or cell is needed.
    import re as _re

    NEW_SEND_CHAT = """  async function sendChat() {
    const input = document.getElementById("chat-input");
    const q = input.value.trim();
    if (!q || !lastAnalysis) return;
    input.value = "";
    appendUserMsg(q);
    showTyping();
    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, analysis: lastAnalysis })
      });
      removeTyping();
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      appendBotText(data.answer || "No answer returned.");
    } catch(e) {
      removeTyping();
      appendBotText("Could not reach the server: " + e.message);
    }
  }"""

    # Replace the entire sendChat function in the HTML
    patched = _re.sub(
        r'async function sendChat\(\)\s*\{.*?\n  \}',
        NEW_SEND_CHAT,
        html,
        flags=_re.DOTALL
    )
    return patched


@app.get("/")
def root():
    return {"message": "HR CV Analyzer", "mode": "MOCK" if USE_MOCK else "REAL",
            "docs": "/docs", "ui": "/ui"}

@app.get("/health")
def health():
    return {"status": "ok", "mode": "mock" if USE_MOCK else "real"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(req: AnalyzeRequest):
    if not req.cv_text.strip():         raise HTTPException(400, "cv_text is empty")
    if not req.job_description.strip(): raise HTTPException(400, "job_description is empty")
    return AnalyzeResponse(**run_analysis(req.cv_text, req.job_description, "text"))

@app.post("/analyze/pdf", response_model=AnalyzeResponse)
async def analyze_pdf(
    cv_pdf: UploadFile = File(...),
    job_description: str = Form(default=""),
):
    if not cv_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF (.pdf)")
    if not job_description.strip():
        logger.warning("No job description provided — domain mismatch detection disabled")
        job_description = "Position not specified"
    pdf_bytes = await cv_pdf.read()
    try:
        cv_text, method = smart_extract(pdf_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"PDF extraction failed: {e}")
    if not cv_text or len(cv_text) < 20:
        raise HTTPException(422, "Could not extract readable text from this PDF.")
    return AnalyzeResponse(**run_analysis(cv_text, job_description, method))