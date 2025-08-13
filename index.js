// Multi-client chatbot backend with grounded AI answers (no hallucinations).
// - Reads /configs/<clientId>.json (hot-reloaded)
// - Semantic match across FAQs + Knowledge (q/a + keywords supported)
// - Conversational model fallback with rolling session history + few-shot examples
// - Behavior (greeter, low-info handling, fallback style, thresholds, memory) driven by client JSON
// - Pricing guardrail; "last message" special
// - Google Sheet lead capture

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
  const faqs = cfg?.faqs || [];
  for (const f of faqs) {
    const keys = (f.keywords || []).map(k => String(k).toLowerCase());
    if (keys.some(k => ["contact","support","email","phone","reach","number","hours"].includes(k))) return f.a;
  }
  for (const f of faqs) {
    const q = (f.q || "").toLowerCase();
    if (q.includes("contact") || q.includes("support")) return f.a;
  }
  // fallback: knowledge.support if present
  if (cfg?.knowledge?.support) return cfg.knowledge.support;
  return "";
}
function getPricingLine(cfg) {
  const faqs = cfg?.faqs || [];
  for (const f of faqs) {
    const keys = (f.keywords || []).map(k => String(k).toLowerCase());
    if (keys.some(k => ["pricing","cost","monthly","fee","charge","subscription","price","money"].includes(k))) {
      return f.a;
    }
  }
  if (cfg?.knowledge?.pricing_policy) return cfg.knowledge.pricing_policy;
  return "Our service costs $35 per month with an initial setup fee of $75.";
}

// ---------- Behavior from JSON with sane defaults ----------
function bval(cfg, path, fallback) {
  const parts = path.split(".");
  let cur = cfg?.behavior;
  for (const p of parts) {
    if (cur && Object.prototype.hasOwnProperty.call(cur, p)) cur = cur[p];
    else return fallback;
  }
  return cur ?? fallback;
}
function safeRegexUnion(patterns, fallbackRe) {
  try {
    const joined = (patterns || []).map(s => `(?:${s})`).join("|");
    return joined ? new RegExp(joined, "i") : fallbackRe;
  } catch {
    return fallbackRe;
  }
}

// ---------- Grounding context (includes FAQs + Knowledge + styling) ----------
function buildGroundingContext(cfg) {
  const brand = cfg.brandName || "Craver Labs AI";
  const contact = getContactLine(cfg);
  const pricing = getPricingLine(cfg);

  const faqLines = (cfg.faqs || []).slice(0, 48).map((f, i) => {
    const header = f.q ? `Q: ${f.q}` : `FAQ ${i+1}`;
    const keys = (f.keywords || []).slice(0, 10).join(", ");
    const keyStr = keys ? ` [${keys}]` : "";
    return `${header}${keyStr}\nA: ${f.a}`;
  }).join("\n\n");

  // Knowledge blocks flattened
  const knowledgeBlocks = [];
  if (cfg.knowledge) {
    for (const [k, v] of Object.entries(cfg.knowledge)) {
      if (Array.isArray(v)) {
        knowledgeBlocks.push(`${k}:\n- ${v.join("\n- ")}`);
      } else if (typeof v === "string") {
        knowledgeBlocks.push(`${k}: ${v}`);
      }
    }
  }
  const knowledgeText = knowledgeBlocks.length ? `\n\nKNOWLEDGE:\n${knowledgeBlocks.join("\n\n")}` : "";

  // Find install summary
  const aboutInstall =
    (cfg.faqs || []).find(f =>
      (f.q && /install|setup|integration|code|embed/i.test(f.q)) ||
      (f.keywords||[]).some(k => /install|installation|setup|integration|code|how to add|apply/i.test(String(k)))
    )?.a ||
    "Install is quick: add ~10–15 lines of HTML; your unique ID applies your customizations.";

  const style = cfg.systemPrompt ? `STYLE GUIDANCE:\n${cfg.systemPrompt}\n\n` : "";

  return `${style}BRAND: ${brand}

AUTHORITATIVE PRICING: ${pricing}

CONTACT: ${contact || "(Contact details not provided.)"}

INSTALLATION: ${aboutInstall}

POLICIES:
- Use ONLY information in this CONTEXT, FAQs, and KNOWLEDGE. Do NOT invent features/services.
- If the user asks about pricing or money, answer with the AUTHORITATIVE PRICING line verbatim.
- If the exact info is not present, say you're not certain and offer CONTACT.
- Keep replies natural and concise (<=120 words). Avoid repeating the same greeting.

FAQs:
${faqLines || "(none provided)"}${knowledgeText}`;
}

// ---------- Simple semantic selection across FAQs + Knowledge (no embeddings) ----------
const DEFAULT_STOPWORDS = new Set([
  "the","a","an","and","or","but","of","to","in","on","for","with","about","is","it","this","that","i","you","we",
  "me","my","your","our","at","as","by","be","are","was","were","from","can","how","what","when","where","why",
  "do","did","does","please","could","would","should","tell","say","more","expand","details","detail","info"
]);
function tokenize(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(w => w && !DEFAULT_STOPWORDS.has(w));
}
function overlapScore(aTokens, bTokens) {
  if (!aTokens.length || !bTokens.length) return 0;
  const aSet = new Set(aTokens);
  const bSet = new Set(bTokens);
  let inter = 0;
  for (const t of aSet) if (bSet.has(t)) inter++;
  const denom = Math.sqrt(aSet.size * bSet.size);
  return denom ? inter / denom : 0;
}
function collectKnowledgeSnippets(cfg) {
  const out = [];
  if (!cfg?.knowledge) return out;
  for (const [k, v] of Object.entries(cfg.knowledge)) {
    if (Array.isArray(v)) {
      v.forEach((line, i) => {
        out.push({ q: `${k} ${i+1}`, a: String(line) });
      });
    } else if (typeof v === "string") {
      out.push({ q: k, a: v });
    }
  }
  return out;
}
function bestSemanticMatch(cfg, userMsg, minScore = 0.18) {
  const items = [];
  const faqs = cfg?.faqs || [];
  faqs.forEach(f => items.push({ q: [f.q, ...(f.keywords||[])].filter(Boolean).join(" "), a: f.a }));

  const kn = collectKnowledgeSnippets(cfg);
  kn.forEach(s => items.push({ q: s.q, a: s.a }));

  if (!items.length) return null;

  const uTok = tokenize(userMsg);
  let best = null;

  items.forEach((itm, idx) => {
    const candTok = tokenize(`${itm.q || ""} ${itm.a || ""}`);
    const score = overlapScore(uTok, candTok);
    if (!best || score > best.score) best = { idx, item: itm, score };
  });

  return best && best.score >= minScore ? best.item : null;
}

// ---------- Menus / greetings ----------
function menuFromCommonQuestions(cfg, count = 4) {
  const qs = (cfg.commonQuestions || []).slice(0, count);
  if (!qs.length) return "";
  return "\n\nHere are a few things I can help with:\n– " + qs.join("\n– ");
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

  // Behavior config
  const enableGreeter = bval(cfg, "enableGreeter", true);
  const greeterMessage = bval(cfg, "greeterMessage", cfg.welcomeMessage || "Hi! I’m here to help with setup, support, or pricing—what do you need?");
  const clarifyMessage = bval(cfg, "clarifyMessage", "Got it—what do you need help with (installation steps, pricing, or lead setup)?");
  const menuItemCount = bval(cfg, "menuItemCount", 4);
  const minSemantic = bval(cfg, "semantic.minScore", 0.18);
  const historyTurns = bval(cfg, "memory.historyTurns", 8);
  const lastTpl = bval(cfg, "memory.lastMessageTemplate", 'My last message was: "{{text}}"');
  const lastNoneTpl = bval(cfg, "memory.noLastMessageTemplate", "I haven’t sent a prior message yet in this session.");
  const enableNRFallback = bval(cfg, "enableNonRepeatingFallback", true);
  const attachMenuOnFallback = cfg?.fallback?.attachMenuOnFallback ?? true;
  const fallback1 = cfg?.fallback?.answer || "I’m not certain yet—could you clarify what you need (installation steps, pricing, or lead setup)?";
  const fallback2 = cfg?.fallback?.secondAnswer || "I can help with installation, pricing, or lead setup.";
  const errorNoteOnce = bval(cfg, "errorNoteOnce", "(temporary issue, trying again shortly)");

  // Regex from JSON (safe-compiled)
  const GREETING_RE = safeRegexUnion(bval(cfg, "greetingPatterns", [
    "\\b(hi|hello|hey|yo|howdy|good (morning|afternoon|evening))\\b"
  ]), /\b(hi|hello|hey|yo|howdy|good (morning|afternoon|evening))\b/i);

  const LOWINFO_RE = safeRegexUnion(bval(cfg, "lowInfoPatterns", [
    "^(\\s*[\\?\\.!]*\\s*|what[?!]*|are you[?!]*|huh[?!]*|idk|ok|k|sure|yeah|yep|nope|who(dis)?|sup)\\s*$"
  ]), /^(\s*[\?\.!]*\s*|what[?!]*|are you[?!]*|huh[?!]*|idk|ok|k|sure|yeah|yep|nope|who(dis)?|sup)\s*$/i);

  // Session init
  if (!req.session[clientId]) req.session[clientId] = { askedExtra: false, history: [], usedFallbackOnce: false, errorNotified: false };
  const s = req.session[clientId];

  const messageRaw = (req.body.message || "").toString();
  const lower = messageRaw.trim().toLowerCase();

  // 0) "what was your last message?"
  if (/\b(last message|what did you just say|what was your previous reply)\b/i.test(lower)) {
    const last = [...(s.history || [])].reverse().find(m => m.role === "assistant");
    const reply = last?.content ? lastTpl.replace("{{text}}", last.content) : lastNoneTpl;
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: reply });
    s.history = s.history.slice(-2 * historyTurns);
    return res.json({ reply });
  }

  // 0.5) Greeter / low-info handler (prevents fallback loops)
  if (enableGreeter && (GREETING_RE.test(lower) || LOWINFO_RE.test(lower))) {
    const base = GREETING_RE.test(lower) ? greeterMessage : clarifyMessage;
    let reply = base;
    if (menuItemCount > 0) reply += menuFromCommonQuestions(cfg, menuItemCount);
    s.askedExtra = true;
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: reply });
    s.history = s.history.slice(-2 * historyTurns);
    return res.json({ reply });
  }

  // 1) Semantic pass over FAQs + Knowledge
  const sem = bestSemanticMatch(cfg, lower, Number(minSemantic) || 0.18);
  if (sem?.a) {
    const reply = sem.a;
    s.askedExtra = true;
    s.usedFallbackOnce = false;
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: reply });
    s.history = s.history.slice(-2 * historyTurns);
    return res.json({ reply });
  }

  // 2) Pricing guardrail
  if (/\b(price|pricing|cost|fee|charge|subscription|money)\b/i.test(lower)) {
    const reply = getPricingLine(cfg);
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: reply });
    s.history = s.history.slice(-2 * historyTurns);
    return res.json({ reply });
  }

  // 3) Closing pass
  if (s.askedExtra && /\b(no|nope|that's all|nothing|i'm good|im good|all set)\b/i.test(lower)) {
    const closing = cfg.closingMessage || "";
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: closing });
    s.history = s.history.slice(-2 * historyTurns);
    return res.json({ reply: closing });
  }

  // 4) Grounded model fallback (reason from CONTEXT/FAQs/KNOWLEDGE + short history + examples)
  try {
    const modelName = (cfg.model && cfg.model.name) || "gpt-5-mini";
    const grounding = buildGroundingContext(cfg);
    const prior = (s.history || []).slice(-historyTurns);
    const messages = [
      {
        role: "system",
        content:
`You are a helpful assistant for ${cfg.brandName || "Craver Labs AI"}.
You are stateless; use only the provided conversation messages and the CONTEXT below.
If specific details are missing, ask one brief clarifying question or say you're not certain and offer CONTACT.

CONTEXT (authoritative):
${grounding}

INSTRUCTIONS:
- Use ONLY the above CONTEXT, FAQs, and KNOWLEDGE.
- If asked about pricing/money, reply with the AUTHORITATIVE PRICING line exactly.
- Keep responses natural and concise (<=120 words).`
      },
      // Few-shot examples (optional, from config)
      ...((cfg.semanticExamples || []).slice(0, 4).flatMap(ex => ([
        { role: "user", content: ex.user },
        { role: "assistant", content: ex.assistant }
      ]))),
      ...prior,
      { role: "user", content: messageRaw }
    ];

    const rsp = await openai.chat.completions.create({
      model: modelName,
      temperature: 0.2,
      max_tokens: 300,
      messages
    });

    let text = rsp.choices?.[0]?.message?.content?.trim();
    // Safety net: if the model ignored the pricing rule, force-correct
    if (/\b(price|pricing|cost|fee|charge|subscription|money)\b/i.test(lower)) {
      text = getPricingLine(cfg);
    }

    let reply = text && text.length ? text : fallback1;
    if ((!text || !text.length) && attachMenuOnFallback && menuItemCount > 0) {
      reply += menuFromCommonQuestions(cfg, menuItemCount);
    }

    s.askedExtra = true;
    s.usedFallbackOnce = !text || !text.length;
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: reply });
    s.history = s.history.slice(-2 * historyTurns);
    return res.json({ reply });
  } catch (err) {
    console.error("OpenAI error:", err);

    // Non-repeating fallback
    let reply;
    if (enableNRFallback && !s.usedFallbackOnce) {
      reply = fallback1;
      if (attachMenuOnFallback && menuItemCount > 0) reply += menuFromCommonQuestions(cfg, menuItemCount);
      if (!s.errorNotified && errorNoteOnce) reply += ` ${errorNoteOnce}`;
      s.usedFallbackOnce = true;
      s.errorNotified = true;
    } else {
      reply = fallback2;
      if (attachMenuOnFallback && menuItemCount > 0) reply += menuFromCommonQuestions(cfg, menuItemCount);
    }

    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: reply });
    s.history = s.history.slice(-2 * historyTurns);
    return res.json({ reply });
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




