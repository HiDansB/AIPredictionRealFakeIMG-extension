importScripts("ort.min.js");

const CONFIG = {
  INPUT_SIZE: 200,
  MEAN: [0.485, 0.456, 0.406],
  STD: [0.229, 0.224, 0.225],

  // ✅ calibrated after portrait + AI fine-tune
  FAKE_THRESHOLD: 0.5,
  SUSPICIOUS_THRESHOLD: 0.5,
};

let session = null;
let modelLoaded = false;

// ── Context menu ─────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyzeImage",
    title: "🤖 Phân tích ảnh AI",
    contexts: ["image"],
  });
  chrome.contextMenus.create({
    id: "analyzeVideo",
    title: "🎬 Phân tích video AI",
    contexts: ["video"],
  });
});

// ── Load SINGLE model ────────────────────────────────────────
async function loadModel() {
  if (modelLoaded) return true;

  try {
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/";
    ort.env.wasm.numThreads = 1;

    const buf = await (
      await fetch(chrome.runtime.getURL("model/model.onnx"))
    ).arrayBuffer();

    session = await ort.InferenceSession.create(buf, {
      executionProviders: ["wasm"],
    });

    modelLoaded = true;
    console.log("[BG] ✅ Fine-tuned model loaded");
    return true;
  } catch (e) {
    console.error("[BG] ❌ Model load error:", e.message);
    return false;
  }
}

// ── Preprocess ───────────────────────────────────────────────
function preprocess(px, size) {
  const [mR, mG, mB] = CONFIG.MEAN;
  const [sR, sG, sB] = CONFIG.STD;

  const buf = new Float32Array(3 * size * size);

  for (let i = 0; i < size * size; i++) {
    buf[i] = (px[i * 4] / 255 - mR) / sR;
    buf[size * size + i] = (px[i * 4 + 1] / 255 - mG) / sG;
    buf[2 * size * size + i] = (px[i * 4 + 2] / 255 - mB) / sB;
  }

  return buf;
}

// ── Stable softmax ───────────────────────────────────────────
function softmax(logits) {
  const m = Math.max(logits[0], logits[1]);
  const e0 = Math.exp(logits[0] - m);
  const e1 = Math.exp(logits[1] - m);

  return {
    pReal: e0 / (e0 + e1),
    pFake: e1 / (e0 + e1),
  };
}

// ── Fetch pixels ─────────────────────────────────────────────
async function fetchPixels(url, size) {
  const response = await fetch(url, {
    mode: "cors",
    cache: "no-cache",
  });

  const blob = await response.blob();

  const bitmap = await createImageBitmap(blob, {
    resizeWidth: size,
    resizeHeight: size,
    resizeQuality: "high",
    imageOrientation: "none",
  });

  const cvs = new OffscreenCanvas(size, size);
  const ctx = cvs.getContext("2d", { colorSpace: "srgb" });

  ctx.drawImage(bitmap, 0, 0, size, size);
  bitmap.close();

  const imageData = ctx.getImageData(0, 0, size, size, {
    colorSpace: "srgb",
  });

  return imageData.data;
}

// ── Single model inference ───────────────────────────────────
async function runInference(imageUrl) {
  const px = await fetchPixels(imageUrl, CONFIG.INPUT_SIZE);

  const input = new ort.Tensor("float32", preprocess(px, CONFIG.INPUT_SIZE), [
    1,
    3,
    CONFIG.INPUT_SIZE,
    CONFIG.INPUT_SIZE,
  ]);

  const out = await session.run({
    [session.inputNames[0]]: input,
  });

  const logits = out[session.outputNames[0]].data;
  const prob = softmax(logits);

  console.log("[MODEL]", prob);

  return prob;
}

// ── UI helpers ───────────────────────────────────────────────
async function injectUI(tabId) {
  await chrome.scripting.executeScript({
    target: { tabId },
    func: () => {
      if (document.getElementById("ai-det-style")) return;

      const s = document.createElement("style");
      s.id = "ai-det-style";
      s.textContent = `
        @keyframes aiSpin { to{transform:rotate(360deg)} }
        @keyframes aiSlide { from{opacity:0;transform:translateX(16px)} to{opacity:1;transform:translateX(0)} }
        #ai-det-box {
          position:fixed!important;
          top:20px!important;
          right:20px!important;
          z-index:2147483647!important;
          width:300px!important;
          background:#13151c!important;
          border:1px solid #2d3148!important;
          border-radius:16px!important;
          padding:18px!important;
          box-shadow:0 20px 60px rgba(0,0,0,.7)!important;
          font-family:-apple-system,BlinkMacSystemFont,sans-serif!important;
          color:#e8eaf0!important;
          animation:aiSlide .3s ease!important;
        }
      `;
      document.head.appendChild(s);
    },
  });
}

async function showLoading(tabId) {
  await injectUI(tabId);

  await chrome.scripting.executeScript({
    target: { tabId },
    func: () => {
      let box = document.getElementById("ai-det-box");
      if (!box) {
        box = document.createElement("div");
        box.id = "ai-det-box";
        document.body.appendChild(box);
      }

      box.innerHTML = `
        <div style="display:flex;align-items:center;gap:12px">
          <div style="width:26px;height:26px;border:3px solid #2d3148;border-top-color:#4f8ef7;border-radius:50%;animation:aiSpin .8s linear infinite"></div>
          <div>
            <div style="font-size:14px;font-weight:600">Đang phân tích...</div>
            <div style="font-size:12px;color:#6b7280">Fine-tuned CvT đang xử lý</div>
          </div>
        </div>
      `;
    },
  });
}

async function showResult(tabId, result, imageUrl) {
  await chrome.scripting.executeScript({
    target: { tabId },
    func: (r, url, fakeThr, suspiciousThr) => {
      const box = document.getElementById("ai-det-box");
      if (!box) return;

      const fakePct = Math.round(r.pFake * 100);
      const realPct = Math.round(r.pReal * 100);

      let title = "✅ Real Image";
      let color = "#22c55e";
      let confidence = realPct;

      if (r.pFake >= fakeThr) {
        title = "🤖 AI-Generated";
        color = "#ef4444";
        confidence = fakePct;
      } else if (r.pFake >= suspiciousThr) {
        title = "⚠️ Suspicious";
        color = "#f59e0b";
        confidence = fakePct;
      }

      box.innerHTML = `
        <div style="font-size:14px;font-weight:700;margin-bottom:12px">
          🤖 AI Detector (Fine-tuned)
        </div>

        <img src="${url}" 
             style="width:100%;height:100px;object-fit:cover;border-radius:8px;margin-bottom:12px"/>

        <div style="font-size:16px;font-weight:700;color:${color};margin-bottom:10px">
          ${title}
        </div>

        <div style="font-size:13px;margin-bottom:10px">
          Độ tin cậy: <strong>${confidence}%</strong>
        </div>

        <div style="margin-bottom:6px">✅ Real: ${realPct}%</div>
        <div style="margin-bottom:12px">🤖 Fake: ${fakePct}%</div>

        <div style="font-size:11px;color:#6b7280;border-top:1px solid #2d3148;padding-top:8px">
          CvT-13 Fine-tuned · ONNX Runtime · 100% local
        </div>
      `;
    },
    args: [
      result,
      imageUrl,
      CONFIG.FAKE_THRESHOLD,
      CONFIG.SUSPICIOUS_THRESHOLD,
    ],
  });
}

async function showError(tabId, message) {
  await chrome.scripting.executeScript({
    target: { tabId },
    func: (msg) => {
      const box = document.getElementById("ai-det-box");
      if (!box) return;

      box.innerHTML = `
        <div style="color:#ef4444;font-size:13px">
          ⚠️ ${msg}
        </div>
      `;
    },
    args: [message],
  });
}

// ── Main handler ─────────────────────────────────────────────
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "analyzeImage" || !info.srcUrl) return;

  await showLoading(tab.id);

  try {
    const ok = await loadModel();
    if (!ok) {
      await showError(tab.id, "Không load được model ONNX");
      return;
    }

    const result = await runInference(info.srcUrl);

    console.log("[BG] Result:", result);

    await showResult(tab.id, result, info.srcUrl);
  } catch (e) {
    console.error("[BG] Error:", e.message);
    await showError(tab.id, e.message || "Có lỗi xảy ra");
  }
});
