# ── CELL 2c : Patch index.html to use the /chat endpoint ─────────────────────
# Run this AFTER Cell 2 (which writes main.py) and BEFORE Cell 4 (server start)

with open("/content/index.html", "r") as f:
    html = f.read()

# ── New sendChat: calls /chat with full analysis object ───────────────────────
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

# ── Replace everything from sendChat declaration to the tryLocalAnswer block ──
import re

# Match the full sendChat function (from async function sendChat to its closing })
# then also remove tryLocalAnswer since it's no longer needed
pattern = re.compile(
    r'// ── Chat ─+\n\s*async function sendChat\(\).*?'
    r'(?=\s*document\.getElementById\("chat-input"\)\.addEventListener)',
    re.DOTALL
)

patched, n = pattern.subn(NEW_SEND_CHAT + "\n\n  ", html)

if n == 0:
    # Fallback: find the block between sendChat opening and the addEventListener
    start_marker = "async function sendChat() {"
    end_marker = 'document.getElementById("chat-input").addEventListener'
    s = html.find(start_marker)
    e = html.find(end_marker)
    if s != -1 and e != -1:
        # Walk back to include the comment line before sendChat
        comment_start = html.rfind("  // ── Chat", 0, s)
        if comment_start == -1:
            comment_start = s
        patched = html[:comment_start] + NEW_SEND_CHAT + "\n\n  " + html[e:]
        print("✅ Patched via fallback string splice")
    else:
        print("❌ Could not locate sendChat in index.html — skipping patch")
        patched = html
else:
    print(f"✅ Patched via regex ({n} replacement)")

# Also remove tryLocalAnswer function entirely (no longer called)
patched = re.sub(
    r'\n  // Answer simple questions.*?return null;\n  \}',
    '',
    patched,
    flags=re.DOTALL
)

with open("/content/index.html", "w") as f:
    f.write(patched)

# Verify
with open("/content/index.html") as f:
    check = f.read()
ok = '"/chat"' in check
print(f"{'✅' if ok else '❌'} /chat endpoint {'found' if ok else 'NOT found'} in index.html")
print(f"{'✅' if 'tryLocalAnswer' not in check else '⚠️  tryLocalAnswer still present'}")