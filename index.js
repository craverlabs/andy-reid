// Multi-client chatbot backend with grounded AI answers (no hallucinations).
// - Reads /configs/<clientId>.json (hot-reloaded)
// - Semantic FAQ match first (q/a + keywords supported); pricing guardrail
// - Conversational model fallback with rolling session history
// - Model constrained to ONLY use provided context/FAQs and admit uncertainty

import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import session from "express-session";
import OpenAI from "openai";
import { GoogleSpreadsheet } from "google-spreadsheet";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const app = express();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.use(cors());
app.use(bodyParser.json());
app.use(session({
  secret: process.env.SESSION_SECRET || "craverlabs",
  resave: false,
  saveUninitialized: true
}));

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.static(path.join(__dirname, "public")));
app.use("/configs", express.static(path.join(__dirname, "configs")));

const CONFIGS_DIR = path.join(__dirname, "configs");
const clientCache = Object.create(null);

// ---------- Config loading / hot-reload ----------
function readClientConfig(clientId) {
  const file = path.join(CONFIGS_DIR, `${clientId}.json`);
  if (!fs.existsSync(file)) return null;
  try { return JSON.parse(fs.readFileSync(file, "utf-8")); }
  catch (e) { console.error("Config parse error:", clientId, e); return null; }
}
function getClientConfig(clientId) {
  if (!clientId) return null;
  if (clientCache[clientId]) return clientCache[clientId];
  const cfg = readClientConfig(clientId);
  if (cfg) clientCache[clientId] = cfg;
  return cfg;
}
if (fs.existsSync(CONFIGS_DIR)) {
  fs.watch(CONFIGS_DIR, { persistent: false }, (evt, filename) => {
    if (filename && filename.endsWith(".json")) {
      const id = filename.replace(".json", "");
      delete clientCache[id];
      const fresh = readClientConfig(id);
      if (fresh) clientCache[id] = fresh;
      console.log(`Config reloaded: ${id}`);
    }
  });
}

// ---------- Helpers: contact/pricing lines ----------
function wordBoundaryIncludes(message, keyword) {
  if (!keyword) return false;
  const kw = String(keyword).replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&");
  const re = new RegExp(`\\b${kw}\\b`, "i");
  return re.test(message);
}
function getContactLine(cfg) {
  // Prefer explicit 'support' FAQ item, else empty
  const faqs = cfg?.faqs || [];
  for (const f of faqs) {
    const keys = (f.keywords || []).map(k => String(k).toLowerCase());
    if (keys.some(k => ["contact","support","email","phone","reach","number","hours"].includes(k))) {
      return f.a;
    }
  }
  // Also check q-text FAQs
  for (const f of faqs) {
    if ((f.q || "").toLowerCase().includes("contact")) return f.a;
    if ((f.q || "").toLowerCase().includes("support")) return f.a;
  }
  return "";
}
function getPricingLine(cfg) {
  // Explicitly scan FAQs first
  const faqs = cfg?.faqs || [];
  for (const f of faqs) {
    const keys = (f.keywords || []).map(k => String(k).toLowerCase());
    if (keys.some(k => ["pricing","cost","monthly","fee","charge","subscription","price","money"].includes(k))) {
      return f.a;
    }
  }
  // Or rely on hard-coded policy text in config if present
  if (cfg?.knowledge?.pricing_policy) return cfg.knowledge.pricing_policy;
  // Hard fallback
  return "Our service costs $35 per month with an initial setup fee of $75.";
}

// ---------- Grounding context ----------
function buildGroundingContext(cfg) {
  const brand = cfg.brandName || "Craver Labs AI";
  const contact = getContactLine(cfg);
  const pricing = getPricingLine(cfg);

  const faqLines = (cfg.faqs || []).slice(0, 24).map((f, i) => {
    const header = f.q ? `Q: ${f.q}` : `FAQ ${i+1}`;
    const keys = (f.keywords || []).slice(0, 8).join(", ");
    const keyStr = keys ? ` [${keys}]` : "";
    return `${header}${keyStr}\nA: ${f.a}`;
  }).join("\n\n");

  // Try to find install/how-to answer from either q/a or keywords
  const aboutInstall =
    (cfg.faqs || []).find(f =>
      (f.q && /install|setup|integration|code|embed/i.test(f.q)) ||
      (f.keywords||[]).some(k => /install|installation|setup|integration|code|how to add|apply/i.test(String(k)))
    )?.a ||
    "Install is quick: add ~10–15 lines of HTML; your unique ID applies your customizations.";

  const style = cfg.systemPrompt
    ? `STYLE GUIDANCE:\n${cfg.systemPrompt}\n\n`
    : "";

  return `${style}BRAND: ${brand}

AUTHORITATIVE PRICING: ${pricing}

CONTACT: ${contact || "(Contact details not provided.)"}

INSTALLATION: ${aboutInstall}

POLICIES:
- Use ONLY information in this CONTEXT and FAQs. Do NOT invent features/services.
- If the user asks about pricing or money, answer with the AUTHORITATIVE PRICING line verbatim.
- If the exact info is not present, say you're not certain and offer CONTACT.
- Keep replies natural and concise (<=120 words). Avoid repeating the same greeting.

FAQs:
${faqLines || "(none provided)"}`;
}

// ---------- Simple semantic FAQ selection (no embeddings) ----------
const STOPWORDS = new Set([
  "the","a","an","and","or","but","of","to","in","on","for","with","about","is","it","this","that","i","you","we",
  "me","my","your","our","at","as","by","be","are","was","were","from","can","how","what","when","where","why",
  "do","did","does","please","could","would","should","tell","say","more","expand","details","detail","info"
]);
function tokenize(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(w => w && !STOPWORDS.has(w));
}
function overlapScore(aTokens, bTokens) {
  if (!aTokens.length || !bTokens.length) return 0;
  const aSet = new Set(aTokens);
  const bSet = new Set(bTokens);
  let inter = 0;
  for (const t of aSet) if (bSet.has(t)) inter++;
  const denom = Math.sqrt(aSet.size * bSet.size); // cosine-like
  return denom ? inter / denom : 0;
}
function bestFaqMatch(cfg, userMsg) {
  const faqs = cfg?.faqs || [];
  if (!faqs.length) return null;

  const uTok = tokenize(userMsg);
  let best = null;

  faqs.forEach((f, idx) => {
    const qText = [f.q, ...(f.keywords || [])].filter(Boolean).join(" ");
    const aText = f.a || "";
    const candTok = tokenize(`${qText} ${aText}`);
    const score = overlapScore(uTok, candTok);
    if (!best || score > best.score) best = { idx, item: f, score };
  });

  // Threshold—tuneable. 0.18 works decently for short queries.
  return best && best.score >= 0.18 ? best.item : null;
}

// ---------- Google Sheets ----------
async function appendToSheet(lead, sheetId) {
  const creds = JSON.parse(process.env.GOOGLE_CREDS);
  const doc = new GoogleSpreadsheet(sheetId);
  await doc.useServiceAccountAuth(creds);
  await doc.loadInfo();
  const sheet = doc.sheetsByIndex[0];
  try { await sheet.setHeaderRow(["Name","Email","Phone","Message","Timestamp","PagePath"]); } catch {}
  await sheet.addRow({
    Name: lead.name,
    Email: lead.email,
    Phone: lead.phone,
    Message: lead.message || "",
    Timestamp: new Date().toISOString(),
    PagePath: lead.pagePath || ""
  });
}

// ---------- Health/version ----------
app.get("/healthz", (_, res) => res.send("ok"));
app.get("/version", (_, res) => {
  const ids = fs.existsSync(CONFIGS_DIR)
    ? fs.readdirSync(CONFIGS_DIR).filter(f => f.endsWith(".json")).map(f => f.replace(".json",""))
    : [];
  res.json({ status: "ok", clients: ids });
});

// ---------- Chat ----------
app.post("/chat", async (req, res) => {
  const clientId = String(req.query.client || "").trim();
  const cfg = getClientConfig(clientId);
  if (!cfg) return res.status(400).json({ error: "Invalid client" });

  if (!req.session[clientId]) req.session[clientId] = { askedExtra: false, history: [] };
  const s = req.session[clientId];

  const messageRaw = (req.body.message || "").toString();
  const lower = messageRaw.trim().toLowerCase();

  // 0) Special: "what was your last message"
  if (/\b(last message|what did you just say|what was your previous reply)\b/i.test(lower)) {
    const last = [...(s.history || [])].reverse().find(m => m.role === "assistant");
    const reply = last?.content
      ? `My last message was: "${last.content}"`
      : "I haven’t sent a prior message yet in this session.";
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: reply });
    s.history = s.history.slice(-20);
    return res.json({ reply });
  }

  // 1) Semantic FAQ pass (uses q/a and keywords; replaces brittle keyword-only match)
  const sem = bestFaqMatch(cfg, lower);
  if (sem?.a) {
    const reply = sem.a;
    s.askedExtra = true;
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: reply });
    s.history = s.history.slice(-20);
    return res.json({ reply });
  }

  // 2) Pricing guardrail
  if (/\b(price|pricing|cost|fee|charge|subscription|money)\b/i.test(lower)) {
    const reply = getPricingLine(cfg);
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: reply });
    s.history = s.history.slice(-20);
    return res.json({ reply });
  }

  // 3) Closing pass (after at least one helpful answer)
  if (s.askedExtra && /\b(no|nope|that's all|nothing|i'm good|im good|all set)\b/i.test(lower)) {
    const closing = cfg.closingMessage || "";
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: closing });
    s.history = s.history.slice(-20);
    return res.json({ reply: closing });
  }

  // 4) Grounded model fallback (short rolling history maintained)
  try {
    const modelName = (cfg.model && cfg.model.name) || "gpt-5-mini";
    const grounding = buildGroundingContext(cfg);
    const prior = (s.history || []).slice(-8); // short rolling history
    const messages = [
      {
        role: "system",
        content:
`You are a helpful assistant for ${cfg.brandName || "Craver Labs AI"}.
CONTEXT (authoritative):
${grounding}

INSTRUCTIONS:
- Use ONLY the above CONTEXT and FAQs.
- If unsure or not present in CONTEXT, say you're not certain and provide CONTACT info.
- If asked about pricing/money, reply with the AUTHORITATIVE PRICING line exactly.
- Keep responses natural and concise (<=120 words).`
      },
      ...prior,
      { role: "user", content: messageRaw }
    ];

    const rsp = await openai.chat.completions.create({
      model: modelName,
      temperature: 0.2,
      max_tokens: 300,
      messages
    });

    const text = rsp.choices?.[0]?.message?.content?.trim();
    const reply = text && text.length
      ? text
      : (cfg.fallback?.answer || "I’m not certain based on what I have. " + getContactLine(cfg));

    s.askedExtra = true;
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: reply });
    s.history = s.history.slice(-20);
    return res.json({ reply });
  } catch (err) {
    console.error("OpenAI error:", err);
    const safe = (cfg.fallback && cfg.fallback.answer) || "I’m not certain based on what I have.";
    const contact = getContactLine(cfg);
    return res.json({ reply: contact ? `${safe} ${contact}` : safe });
  }
});

// ---------- Lead ----------
app.post("/lead", async (req, res) => {
  const { name, email, phone, message, client, pagePath } = req.body || {};
  if (!name || !email || !phone) return res.status(400).json({ error: "Missing fields" });

  const cfg = getClientConfig(String(client || "").trim());
  if (!cfg?.sheetId) return res.status(400).json({ error: "Invalid client (missing sheetId)" });

  try {
    await appendToSheet({ name, email, phone, message, pagePath }, cfg.sheetId);
    return res.json({ success: true });
  } catch (err) {
    console.error("Sheet error:", err);
    return res.status(500).json({ error: "Sheet error" });
  }
});

// ---------- Common questions ----------
app.get("/common-questions", (req, res) => {
  const clientId = String(req.query.client || "").trim();
  const cfg = getClientConfig(clientId);
  if (!cfg) return res.status(400).json({ error: "Invalid client" });
  res.json({ questions: cfg.commonQuestions || [] });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server on :${PORT}`));



