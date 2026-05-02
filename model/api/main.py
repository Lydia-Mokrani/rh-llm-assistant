from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pdfplumber, io, os, logging, re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USE_MOCK   = os.getenv("USE_MOCK", "true").lower() == "true"
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
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
            t = page.extract_text()
            if t:
                text += t + "\n"
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
# TEXT CLEANING — strips boilerplate BEFORE any extraction
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
# DOMAIN DETECTION — prevents healthcare CV scoring 83% on an AI Engineer job
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
    # generic words that appear in EVERY doc and cause false matches
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
# MATCHING — domain-aware
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

    # Cross-domain penalty
    cross_domain_penalty = 0
    if cv_domain != jd_domain and cv_domain != "general" and jd_domain != "general":
        cross_domain_penalty = 30

    if job_terms:
        raw_score = (len(matched) / max(len(job_terms), 1)) * 100
        score = max(5, min(int(raw_score) - cross_domain_penalty, 95))
    else:
        score = max(5, min(40 + len(cv_terms) - cross_domain_penalty, 75))

    # Strengths
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

    # Weaknesses
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
# REAL MODEL — CPU + GPU support, repetition loop fixed
# ═══════════════════════════════════════════════════════════════════════════════

tokenizer, model = None, None

def load_model():
    global tokenizer, model
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    logger.info(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    use_gpu = torch.cuda.is_available()
    logger.info(f"Device: {'GPU' if use_gpu else 'CPU (slow but works)'}")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto" if use_gpu else None,
        dtype=torch.float16 if use_gpu else torch.float32,
    )
    if not use_gpu:
        base = base.to("cpu")

    logger.info(f"Attaching LoRA adapter from {MODEL_PATH}...")
    model = PeftModel.from_pretrained(base, MODEL_PATH)
    model.eval()
    logger.info("Model ready.")

def build_prompt(cv_text: str, job_description: str) -> str:
    cv_clean  = clean_cv(cv_text)
    jd_clean  = clean_job_description(job_description)
    cv_skills = extract_skills(cv_clean)
    jd_skills = extract_skills(jd_clean)
    cv_domain = detect_domain(cv_clean)
    jd_domain = detect_domain(jd_clean)
    matched, missing = terms_overlap(cv_skills, jd_skills, cv_domain, jd_domain)

    hint = ""
    if matched:
        hint += f"\n[Skills found in CV: {', '.join(matched[:6])}]"
    if missing:
        hint += f"\n[Skills missing from CV: {', '.join(missing[:6])}]"

    return (
        "<s>[INST] You are a senior HR recruiter. "
        "Analyze the CV against the job description. "
        "Be specific — reference actual skills, tools, and experience. "
        "Do NOT repeat any word more than twice in your entire response.\n\n"
        f"JOB DESCRIPTION:\n{jd_clean[:800]}\n\n"
        f"CV:\n{cv_clean[:1200]}\n"
        f"{hint}\n\n"
        "Reply in EXACTLY this format:\n"
        "SCORE: <0-100>\n"
        "STRENGTHS:\n- <strength 1>\n- <strength 2>\n- <strength 3>\n"
        "WEAKNESSES:\n- <gap 1>\n- <gap 2>\n"
        "VERDICT: <one sentence> [/INST]"
    )

def _kill_repetition(text: str) -> str:
    """Cut output the moment a word repeats 4+ times in a row."""
    words = text.split()
    result, streak = [], 1
    for i, w in enumerate(words):
        if i > 0 and w.lower() == words[i-1].lower():
            streak += 1
            if streak >= 4:
                break
        else:
            streak = 1
        result.append(w)
    return " ".join(result)

def generate_analysis(cv_text: str, job_description: str) -> str:
    import torch
    prompt = build_prompt(cv_text, job_description)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,          # greedy decoding — no randomness, no loops
            repetition_penalty=1.4,   # strong penalty for repeating tokens
            no_repeat_ngram_size=4,   # never repeat any 4-word sequence
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return _kill_repetition(raw)

def parse_output(raw: str) -> dict:
    result = {
        "score": "N/A", "strengths": [], "weaknesses": [],
        "verdict": "N/A", "raw_output": raw,
    }
    section = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        up = line.upper()
        if "SCORE" in up:
            result["score"] = line.split(":", 1)[-1].strip()
        elif "STRENGTH" in up:
            section = "strengths"
        elif "WEAKNESS" in up:
            section = "weaknesses"
        elif "VERDICT" in up:
            section = "verdict"
            parts = line.split(":", 1)
            if len(parts) > 1 and parts[1].strip():
                result["verdict"] = parts[1].strip()
        elif line.startswith("-"):
            item = line.lstrip("- ").strip()
            if section == "strengths":   result["strengths"].append(item)
            elif section == "weaknesses": result["weaknesses"].append(item)
        elif section == "verdict" and result["verdict"] == "N/A":
            result["verdict"] = line

    s = result["score"].replace("%", "").strip()
    if s.isdigit():
        result["score"] = f"{s}%"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(cv_text: str, job_description: str, method: str = "text") -> dict:
    if USE_MOCK:
        result = smart_mock(cv_text, job_description)
    else:
        raw    = generate_analysis(cv_text, job_description)
        result = parse_output(raw)
    result["extraction_method"] = method
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app):
    if not USE_MOCK:
        load_model()
    else:
        logger.info("MOCK MODE — rule-based analysis active")
    yield

app = FastAPI(title="HR CV Analyzer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"], expose_headers=["*"],
)

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    p = "/content/index.html"
    if not os.path.exists(p):
        raise HTTPException(404, "index.html not found at /content/index.html")
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/")
def root():
    return {"message": "HR CV Analyzer", "mode": "MOCK" if USE_MOCK else "REAL",
            "docs": "/docs", "ui": "/ui"}

@app.get("/health")
def health():
    return {"status": "ok", "mode": "mock" if USE_MOCK else "real"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(req: AnalyzeRequest):
    if not req.cv_text.strip():       raise HTTPException(400, "cv_text is empty")
    if not req.job_description.strip(): raise HTTPException(400, "job_description is empty")
    return AnalyzeResponse(**run_analysis(req.cv_text, req.job_description, "text"))

@app.post("/analyze/pdf", response_model=AnalyzeResponse)
async def analyze_pdf(
    cv_pdf: UploadFile = File(...),
    job_description: str = Form(default="Position not specified"),
):
    if not cv_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF (.pdf)")
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