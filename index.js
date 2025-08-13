// Multi-client chatbot backend with grounded AI answers (no hallucinations).
// - Reads /configs/<clientId>.json (hot-reloaded)
// - FAQ match first; pricing guardrail
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
    if (keys.some(k => ["contact","support","email","phone","reach","number","hours"].includes(k))) {
      return f.a;
    }
  }
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
  return "Our service costs $35 per month with an initial setup fee of $75.";
}
function buildGroundingContext(cfg) {
  const brand = cfg.brandName || "Craver Labs AI";
  const contact = getContactLine(cfg);
  const pricing = getPricingLine(cfg);

  const faqLines = (cfg.faqs || []).slice(0, 16).map((f, i) => {
    const keys = (f.keywords || []).slice(0, 8).join(", ");
    return `FAQ ${i+1} [${keys}]\n${f.a}`;
  }).join("\n\n");

  const aboutInstall = (cfg.faqs || []).find(f =>
    (f.keywords||[]).some(k => /install|installation|setup|integration|code|how to add|apply/i.test(k))
  )?.a || "Install is quick: add ~10 lines of HTML; your unique ID applies your customizations.";

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

async function appendToSheet(lead, sheetId) {
  const creds = JSON.parse(process.env.GOOGLE_CREDS);
  const doc = new GoogleSpreadsheet(sheetId);
  await doc.useServiceAccountAuth(creds);
  await doc.loadInfo();
  const sheet = doc.sheetsByIndex[0];
  try { await sheet.setHeaderRow(["Name","Email","Phone","Message","Timestamp"]); } catch {}
  await sheet.addRow({
    Name: lead.name,
    Email: lead.email,
    Phone: lead.phone,
    Message: lead.message || "",
    Timestamp: new Date().toISOString(),
  });
}

// health/version
app.get("/healthz", (_, res) => res.send("ok"));
app.get("/version", (_, res) => {
  const ids = fs.existsSync(CONFIGS_DIR)
    ? fs.readdirSync(CONFIGS_DIR).filter(f => f.endsWith(".json")).map(f => f.replace(".json",""))
    : [];
  res.json({ status: "ok", clients: ids });
});

// chat
app.post("/chat", async (req, res) => {
  const clientId = String(req.query.client || "").trim();
  const cfg = getClientConfig(clientId);
  if (!cfg) return res.status(400).json({ error: "Invalid client" });

  if (!req.session[clientId]) req.session[clientId] = { askedExtra: false, history: [] };
  const s = req.session[clientId];

  const messageRaw = (req.body.message || "").toString();
  const lower = messageRaw.trim().toLowerCase();

  // 1) FAQ pass
  for (const f of (cfg.faqs || [])) {
    const keys = Array.isArray(f.keywords) ? f.keywords : [];
    if (keys.some(k => wordBoundaryIncludes(lower, String(k).toLowerCase()))) {
      const reply = f.a;
      s.askedExtra = true;
      s.history.push({ role: "user", content: messageRaw });
      s.history.push({ role: "assistant", content: reply });
      s.history = s.history.slice(-20);
      return res.json({ reply });
    }
  }

  // 2) Pricing guardrail
  if (/\b(price|pricing|cost|fee|charge|subscription|money)\b/i.test(lower)) {
    const reply = getPricingLine(cfg);
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: reply });
    s.history = s.history.slice(-20);
    return res.json({ reply });
  }

  // 3) Closing pass
  if (s.askedExtra && /\b(no|nope|that's all|nothing|i'm good|im good|all set)\b/i.test(lower)) {
    const closing = cfg.closingMessage || "";
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: closing });
    s.history = s.history.slice(-20);
    return res.json({ reply: closing });
  }

  // 4) Grounded model fallback
  try {
    const modelName = (cfg.model && cfg.model.name) || "gpt-3.5-turbo";
    const grounding = buildGroundingContext(cfg);
    const prior = (s.history || []).slice(-8); // short rolling history
    const messages = [
      { role: "system",
        content:
`You are a helpful assistant for ${cfg.brandName || "Craver Labs AI"}.
CONTEXT (authoritative):
${grounding}

INSTRUCTIONS:
- Use ONLY the above CONTEXT and FAQs.
- If unsure or not present in CONTEXT, say you're not certain and provide CONTACT info.
- If asked about pricing/money, reply with the AUTHORITATIVE PRICING line exactly.
- Keep responses natural and concise (<=120 words).` },
      ...prior,
      { role: "user", content: messageRaw }
    ];

    const rsp = await openai.chat.completions.create({
      model: modelName,
      temperature: 0.2,
      max_tokens: 450,
      messages
    });

    const text = rsp.choices?.[0]?.message?.content?.trim();
    const reply = text && text.length ? text : (cfg.fallback?.answer || "I’m not certain based on what I have. " + getContactLine(cfg));
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

// lead
app.post("/lead", async (req, res) => {
  const { name, email, phone, message, client } = req.body || {};
  if (!name || !email || !phone) return res.status(400).json({ error: "Missing fields" });

  const cfg = getClientConfig(String(client || "").trim());
  if (!cfg?.sheetId) return res.status(400).json({ error: "Invalid client (missing sheetId)" });

  try {
    await appendToSheet({ name, email, phone, message }, cfg.sheetId);
    return res.json({ success: true });
  } catch (err) {
    console.error("Sheet error:", err);
    return res.status(500).json({ error: "Sheet error" });
  }
});

app.get("/common-questions", (req, res) => {
  const clientId = String(req.query.client || "").trim();
  const cfg = getClientConfig(clientId);
  if (!cfg) return res.status(400).json({ error: "Invalid client" });
  res.json({ questions: cfg.commonQuestions || [] });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server on :${PORT}`));


