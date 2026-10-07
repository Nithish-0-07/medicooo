const USE_BACKEND = false;                       // true -> call FastAPI model
const API_URL = "http://127.0.0.1:8000/predict";
const CLASSES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"];
const STAGES = ["Upload","Quality","Preprocess","EfficientNet-B0","Classify","Safety"];
const $ = (s) => document.querySelector(s);

/* ---------- navigation ---------- */
function show(view) {
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("on", v.id === view));
  document.querySelectorAll("#nav button").forEach(b => b.classList.toggle("on", b.dataset.view === view));
  if (view === "history") renderHistory();
  if (view === "dashboard") renderStats();
}
document.querySelectorAll("[data-view]").forEach(b => b.onclick = () => show(b.dataset.view));
document.querySelectorAll("[data-go]").forEach(b => b.onclick = () => show(b.dataset.go));

/* ---------- history storage ---------- */
const load = () => { try { return JSON.parse(localStorage.getItem("rc_hist") || "[]"); } catch { return []; } };
const save = (h) => { try { localStorage.setItem("rc_hist", JSON.stringify(h)); } catch {} };
function addHistory(rec) { const h = load(); h.unshift(rec); save(h.slice(0, 200)); }

function renderStats() {
  const h = load();
  $("#s-total").textContent = h.length;
  $("#s-refer").textContent = h.filter(r => r.status === "Refer to specialist").length;
  $("#s-poor").textContent = h.filter(r => r.status === "Poor quality").length;
}
function renderHistory() {
  const h = load();
  $("#hist-empty").style.display = h.length ? "none" : "block";
  $("#hist-body").innerHTML = h.map(r => `<tr>
    <td>${new Date(r.time).toLocaleString()}</td><td>${esc(r.patient)}</td><td>${esc(r.file)}</td>
    <td>${esc(r.label)}</td><td>${r.conf == null ? "–" : Math.round(r.conf * 100) + "%"}</td><td>${esc(r.status)}</td></tr>`).join("");
}
$("#clear-hist").onclick = () => { save([]); renderHistory(); };
const esc = (s) => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

/* ---------- image helpers ---------- */
function loadImage(file) {
  return new Promise((res, rej) => {
    const img = new Image(), url = URL.createObjectURL(file);
    img.onload = () => res({ img, url });
    img.onerror = () => rej(new Error("Not a readable image"));
    img.src = url;
  });
}
// Quality assessment: brightness + sharpness (variance of Laplacian) on a 224px copy
function assessQuality(img) {
  const c = document.createElement("canvas"); c.width = c.height = 224;
  const x = c.getContext("2d", { willReadFrequently: true });
  x.drawImage(img, 0, 0, 224, 224);
  const d = x.getImageData(0, 0, 224, 224).data, g = new Float32Array(224 * 224);
  let sum = 0;
  for (let i = 0; i < g.length; i++) { g[i] = .299 * d[i*4] + .587 * d[i*4+1] + .114 * d[i*4+2]; sum += g[i]; }
  const brightness = sum / g.length;
  let m = 0, m2 = 0, n = 0;
  for (let y = 1; y < 223; y++) for (let xx = 1; xx < 223; xx++) {
    const i = y * 224 + xx, l = 4 * g[i] - g[i-1] - g[i+1] - g[i-224] - g[i+224];
    m += l; m2 += l * l; n++;
  }
  const sharpness = m2 / n - (m / n) ** 2;
  const issues = [];
  if (brightness < 25) issues.push("too dark");
  if (brightness > 220) issues.push("overexposed");
  if (sharpness < 15) issues.push("blurry");
  return { ok: !issues.length, issues, brightness, sharpness };
}
// Preprocess to 224x224 tensor-like array (values 0..1, ImageNet normalised)
function preprocess(img) {
  const c = document.createElement("canvas"); c.width = c.height = 224;
  c.getContext("2d").drawImage(img, 0, 0, 224, 224);
  return c;
}

/* ---------- model ---------- */
// DEMO ONLY: deterministic pseudo-prediction seeded by the image pixels.
async function mockModel(canvas) {
  const d = canvas.getContext("2d").getImageData(0, 0, 224, 224).data;
  let seed = 7; for (let i = 0; i < d.length; i += 97) seed = (seed * 31 + d[i]) >>> 0;
  const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
  const raw = CLASSES.map(() => rnd() ** 2 + .05);
  raw[Math.floor(rnd() * 5)] += 1.2;
  const t = raw.reduce((a, b) => a + b, 0);
  await new Promise(r => setTimeout(r, 500));
  return raw.map(v => v / t);
}
async function realModel(file) {
  const fd = new FormData(); fd.append("file", file);
  const r = await fetch(API_URL, { method: "POST", body: fd });
  if (!r.ok) throw new Error("Backend error " + r.status);
  return (await r.json()).probabilities;           // array of 5 numbers
}
function safety(probs) {
  const top = probs.indexOf(Math.max(...probs)), conf = probs[top];
  let status, tone;
  if (top >= 2) { status = "Refer to specialist"; tone = "bad"; }
  else if (conf < .6 || top === 1) { status = "Specialist review advised"; tone = "warn"; }
  else { status = "Routine follow-up"; tone = "ok"; }
  return { top, conf, status, tone };
}

/* ---------- screening flow ---------- */
const drop = $("#drop"), input = $("#files");
input.onchange = () => run([...input.files]);
["dragover","dragenter"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave","drop"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", ev => run([...ev.dataTransfer.files]));

async function run(files) {
  files = files.filter(f => f.type.startsWith("image/"));
  if (!files.length) return;
  const patient = $("#p-name").value.trim() || "Unnamed";
  const id = $("#p-id").value.trim();
  for (const f of files) await processOne(f, id ? `${patient} (${id})` : patient);
  input.value = "";
}

async function processOne(file, patient) {
  const card = document.createElement("div"); card.className = "card";
  card.innerHTML = `<img alt=""><div><h3>${esc(file.name)}</h3><ul class="steps">${STAGES.map(s => `<li>${s}</li>`).join("")}</ul><div class="out">Processing…</div></div>`;
  $("#results").prepend(card);
  const steps = [...card.querySelectorAll(".steps li")], out = card.querySelector(".out");
  const mark = (i, cls = "done") => steps[i].className = cls;
  const record = (label, conf, status) => addHistory({ time: Date.now(), patient, file: file.name, label, conf, status });

  try {
    const { img, url } = await loadImage(file); card.querySelector("img").src = url; mark(0);
    const q = assessQuality(img);
    if (!q.ok) {                                   // poor quality stops THIS image only
      mark(1, "fail");
      out.innerHTML = `<span class="badge b-bad">Poor quality</span><p>Image is ${q.issues.join(" and ")}. Retake the photo and upload again.</p>`;
      record("Not analysed", null, "Poor quality"); return;
    }
    mark(1);
    const canvas = preprocess(img); mark(2);
    const probs = USE_BACKEND ? await realModel(file) : await mockModel(canvas); mark(3);
    const s = safety(probs); mark(4); mark(5);
    out.innerHTML = `<span class="badge b-${s.tone}">${s.status}</span>
      <p><b>${CLASSES[s.top]}</b> · confidence ${Math.round(s.conf * 100)}%</p>
      <div class="bar"><i style="width:${s.conf * 100}%"></i></div>
      <p class="muted">${USE_BACKEND ? "" : "Demo result from a mock model. "}A specialist must review every result.</p>`;
    record(CLASSES[s.top], s.conf, s.status);
  } catch (e) {
    out.innerHTML = `<span class="badge b-bad">Error</span><p>${esc(e.message)}. Check the file or the backend and try again.</p>`;
  }
}
renderStats();
