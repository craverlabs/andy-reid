// index.js — Multi-client chatbot with real conversation memory
// - Per-client config: /configs/<clientId>.json (hot-reloaded)
// - Keeps rolling chat history in express-session per visitor + clientId
// - FAQ match first; pricing guardrail
// - Grounded AI fallback that uses PRIOR TURNS (no “fresh start” each time)

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

// --- Middlewares ---
app.use(cors({ origin: true, credentials: true }));
app.use(bodyParser.json());

// If you’re embedding the widget from another domain, set CROSS_SITE=1 in env
const crossSite = process.env.CROSS_SITE === "1";
app.use(
  session({
    secret: process.env.SESSION_SECRET || "craverlabs",
    resave: false,
    saveUninitialized: true,
    cookie: {
      httpOnly: true,
      secure: crossSite || process.env.NODE_ENV === "production",
      sameSite: crossSite ? "none" : "lax",
      maxAge: 7 * 24 * 60 * 60 * 1000,
    },
  })
);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Static assets
app.use(express.static(path.join(__dirname, "public")));
app.use("/configs", express.static(path.join(__dirname, "configs")));

// --- Load & hot-reload client configs ---
const CONFIGS_DIR = path.join(__dirname, "configs");
const clientCache = Object.create(null);

function readClientConfig(clientId) {
  const file = path.join(CONFIGS_DIR, `${clientId}.json`);
  if (!fs.existsSync(file)) return null;
  try {
    return JSON.parse(fs.readFileSync(file, "utf-8"));
  } catch (e) {
    console.error("Config parse error:", clientId, e);
    return null;
  }
}
function getClientConfig(clientId) {
  if (!clientId) return null;
  if (clientCache[clientId]) return clientCache[clientId];
  const cfg = readClientConfig(clientId);
  if (cfg) clientCache[clientId] = cfg;
  return cfg;
}
if (fs.existsSync(CONFIGS_DIR)) {
  fs.watch(CONFIGS_DIR, { persistent: false }, (_evt, filename) => {
    if (filename && filename.endsWith(".json")) {
      const id = filename.replace(".json", "");
      delete clientCache[id];
      const fresh = readClientConfig(id);
      if (fresh) clientCache[id] = fresh;
      console.log(`Config reloaded: ${id}`);
    }
  });
}

// --- Helpers ---
function wordBoundaryIncludes(message, keyword) {
  if (!keyword) return false;
  const kw = String(keyword).replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&");
  const re = new RegExp(`\\b${kw}\\b`, "i");
  return re.test(message);
}
function getContactLine(cfg) {
  const faqs = cfg?.faqs || [];
  for (const f of faqs) {
    const keys = (f.keywords || []).map((k) => String(k).toLowerCase());
    if (
      keys.some((k) =>
        ["contact", "support", "email", "phone", "reach", "number", "hours"].includes(k)
      )
    ) {
      return f.a;
    }
  }
  return "";
}
function getPricingLine(cfg) {
  const faqs = cfg?.faqs || [];
  for (const f of faqs) {
    const keys = (f.keywords || []).map((k) => String(k).toLowerCase());
    if (
      keys.some((k) =>
        ["pricing", "cost", "monthly", "fee", "charge", "subscription", "price", "money"].includes(
          k
        )
      )
    ) {
      return f.a;
    }
  }
  // Authoritative fallback if not found in JSON:
  return "Our service costs $35 per month with an initial setup fee of $75.";
}
function buildGroundingContext(cfg) {
  const brand = cfg.brandName || "Craver Labs AI";
  const contact = getContactLine(cfg);
  const pricing = getPricingLine(cfg);

  const faqLines = (cfg.faqs || [])
    .slice(0, 16)
    .map((f, i) => {
      const keys = (f.keywords || []).slice(0, 8).join(", ");
      return `FAQ ${i + 1} [${keys}]\n${f.a}`;
    })
    .join("\n\n");

  const installLine =
    (cfg.faqs || []).find((f) =>
      (f.keywords || []).some((k) => /install|installation|setup|integration|code|how to add|apply/i.test(k))
    )?.a ||
    "Install is quick: add ~10 lines of HTML; your unique ID applies your customizations.";

  const style = cfg.systemPrompt ? `STYLE:\n${cfg.systemPrompt}\n\n` : "";

  return `${style}BRAND: ${brand}

AUTHORITATIVE PRICING: ${pricing}

CONTACT: ${contact || "(Contact details not provided.)"}

INSTALLATION: ${installLine}

POLICIES:
- Use ONLY information in this CONTEXT and FAQs. Do NOT invent features/services.
- If the user asks about pricing/money, answer with the AUTHORITATIVE PRICING line verbatim.
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
  try {
    await sheet.setHeaderRow(["Name", "Email", "Phone", "Message", "Timestamp"]);
  } catch {}
  await sheet.addRow({
    Name: lead.name,
    Email: lead.email,
    Phone: lead.phone,
    Message: lead.message || "",
    Timestamp: new Date().toISOString(),
  });
}

// --- Health ---
app.get("/healthz", (_req, res) => res.send("ok"));
app.get("/version", (_req, res) => {
  const ids = fs.existsSync(CONFIGS_DIR)
    ? fs.readdirSync(CONFIGS_DIR).filter((f) => f.endsWith(".json")).map((f) => f.replace(".json", ""))
    : [];
  res.json({ status: "ok", clients: ids });
});

// --- Chat (with conversation memory) ---
app.post("/chat", async (req, res) => {
  const clientId = String(req.query.client || "").trim();
  const cfg = getClientConfig(clientId);
  if (!cfg) return res.status(400).json({ error: "Invalid client" });

  // Per-visitor, per-client session bucket
  if (!req.session[clientId]) req.session[clientId] = { askedExtra: false, history: [] };
  const s = req.session[clientId];

  const messageRaw = (req.body.message || "").toString().trim();
  const lower = messageRaw.toLowerCase();

  // 1) FAQ match
  for (const f of cfg.faqs || []) {
    const keys = Array.isArray(f.keywords) ? f.keywords : [];
    if (keys.some((k) => wordBoundaryIncludes(lower, String(k).toLowerCase()))) {
      const reply = f.a;
      // Append BOTH turns to history so future messages have context
      s.askedExtra = true;
      s.history.push({ role: "user", content: messageRaw });
      s.history.push({ role: "assistant", content: reply });
      s.history = s.history.slice(-24); // cap
      return res.json({ reply });
    }
  }

  // 2) Pricing guardrail (authoritative)
  if (/\b(price|pricing|cost|fee|charge|subscription|money)\b/i.test(lower)) {
    const reply = getPricingLine(cfg);
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: reply });
    s.history = s.history.slice(-24);
    return res.json({ reply });
  }

  // 3) Closing / end-conversation intents
  if (s.askedExtra && /\b(no|nope|that's all|nothing|i'm good|im good|all set)\b/i.test(lower)) {
    const closing = cfg.closingMessage || "";
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: closing });
    s.history = s.history.slice(-24);
    return res.json({ reply: closing });
  }

  // 4) Grounded AI fallback with PRIOR TURNS
  try {
    const modelName = (cfg.model && cfg.model.name) || "gpt-3.5-turbo";
    const grounding = buildGroundingContext(cfg);

    // Include prior turns (user+assistant) so it remembers the conversation
    const prior = (s.history || []).slice(-12); // keep it lean but contextual
    const messages = [
      {
        role: "system",
        content: `You are a helpful assistant for ${cfg.brandName || "Craver Labs AI"}.\n` +
          `CONTEXT (authoritative):\n${grounding}\n\n` +
          `INSTRUCTIONS:\n` +
          `- Use ONLY the above CONTEXT and FAQs.\n` +
          `- If unsure or not present, say you're not certain and provide CONTACT info.\n` +
          `- If asked about pricing/money, reply with the AUTHORITATIVE PRICING line exactly.\n` +
          `- Keep responses natural and concise (<=120 words).`
      },
      ...prior,
      { role: "user", content: messageRaw }
    ];

    const rsp = await openai.chat.completions.create({
      model: modelName,
      temperature: 0.2,
      max_tokens: 450,
      messages
    });

    const ai = rsp.choices?.[0]?.message?.content?.trim();
    const reply =
      ai && ai.length
        ? ai
        : (cfg.fallback?.answer || `I’m not certain based on what I have. ${getContactLine(cfg)}`);

    s.askedExtra = true;
    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: reply });
    s.history = s.history.slice(-24);

    return res.json({ reply });
  } catch (err) {
    console.error("OpenAI error:", err);
    const safe = (cfg.fallback && cfg.fallback.answer) || "I’m not certain based on what I have.";
    const contact = getContactLine(cfg);
    const reply = contact ? `${safe} ${contact}` : safe;

    s.history.push({ role: "user", content: messageRaw });
    s.history.push({ role: "assistant", content: reply });
    s.history = s.history.slice(-24);

    return res.json({ reply });
  }
});

// --- Lead capture ---
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

// --- Suggested questions passthrough ---
app.get("/common-questions", (req, res) => {
  const clientId = String(req.query.client || "").trim();
  const cfg = getClientConfig(clientId);
  if (!cfg) return res.status(400).json({ error: "Invalid client" });
  res.json({ questions: cfg.commonQuestions || [] });
});

// --- Start ---
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Server on :${PORT}`));
