"""
FastAPI servis za predikciju kretanja pešaka — sa vizuelnim UI.

Pokretanje:
    uv run uvicorn deployment.api:app --reload

Interfejs (crtanje putanje):  http://127.0.0.1:8000/ui
Swagger dokumentacija:        http://127.0.0.1:8000/docs
"""

import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator

# =============================================================================
# Učitavanje modela
# =============================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model nije pronađen: {MODEL_PATH}. Prvo pokreni: uv run src/export_model.py"
    )

_bundle = joblib.load(MODEL_PATH)
MODEL = _bundle['model']
N_FEATURES = _bundle['n_features']
PRED_STEPS = _bundle['pred_steps']
WINDOW_SIZE = _bundle['window_size']
FEATURE_NAMES = _bundle['feature_names']

app = FastAPI(
    title="Predikcija kretanja pešaka",
    description="Predviđa narednih N pozicija pešaka na osnovu istorije kretanja.",
    version="1.0",
)

# =============================================================================
# Šeme
# =============================================================================
class ZahtevPredikcije(BaseModel):
    features: list[float]

    @field_validator('features')
    @classmethod
    def proveri_duzinu(cls, v):
        if len(v) != N_FEATURES:
            raise ValueError(f"Očekuje se {N_FEATURES} featura, dobijeno {len(v)}.")
        return v


class Pozicija(BaseModel):
    korak: int
    delta_x: float
    delta_y: float


class OdgovorPredikcije(BaseModel):
    predikcije: list[Pozicija]


# =============================================================================
# Endpointi
# =============================================================================
@app.get("/")
def info():
    return {
        "servis": "Predikcija kretanja pešaka",
        "model": "XGBoost (samo rel koordinate)",
        "window_size": WINDOW_SIZE,
        "ui": "/ui",
        "dokumentacija": "/docs",
    }


@app.post("/predict", response_model=OdgovorPredikcije)
def predict(zahtev: ZahtevPredikcije):
    try:
        x = np.array(zahtev.features, dtype=float).reshape(1, -1)
        y = MODEL.predict(x)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Greška u predikciji: {e}")

    predikcije = [
        Pozicija(korak=s + 1, delta_x=float(y[s]), delta_y=float(y[PRED_STEPS + s]))
        for s in range(PRED_STEPS)
    ]
    return OdgovorPredikcije(predikcije=predikcije)


# =============================================================================
# Vizuelni UI — crtanje putanje klikom
# =============================================================================
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return f"""
<!DOCTYPE html>
<html lang="sr">
<head>
<meta charset="utf-8">
<title>Predikcija kretanja pešaka</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 760px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }}
  h1 {{ font-size: 20px; }}
  p  {{ color: #555; font-size: 14px; line-height: 1.5; }}
  canvas {{ border: 1px solid #ccc; border-radius: 8px; cursor: crosshair; background: #fafafa; display: block; }}
  .row {{ display: flex; gap: 8px; margin: 12px 0; align-items: center; flex-wrap: wrap; }}
  button {{ font-size: 14px; padding: 8px 14px; border: none; border-radius: 6px; cursor: pointer; }}
  #predict {{ background: #2563eb; color: white; }}
  #predict:disabled {{ background: #9db8e8; cursor: default; }}
  #reset {{ background: #e5e7eb; }}
  .status {{ font-size: 13px; color: #666; }}
  .legend {{ font-size: 13px; margin-top: 8px; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }}
</style>
</head>
<body>
  <h1>Predikcija kretanja pešaka</h1>
  <p>Klikni <b>{WINDOW_SIZE}</b> tačaka na grafiku da nacrtaš istoriju kretanja pešaka
     (redom, od prve do poslednje pozicije). Kad uneseš sve tačke, klikni <b>Predvidi</b>
     da vidiš predviđenih narednih {PRED_STEPS} koraka.</p>

  <canvas id="board" width="720" height="480"></canvas>

  <div class="row">
    <button id="predict" disabled>Predvidi</button>
    <button id="reset">Obriši</button>
    <span class="status" id="status">Uneto 0 / {WINDOW_SIZE} tačaka</span>
  </div>

  <div class="legend">
    <span class="dot" style="background:#2563eb"></span> istorija (uneseno)
    &nbsp;&nbsp;
    <span class="dot" style="background:#dc2626"></span> predikcija (model)
  </div>

<script>
const WINDOW = {WINDOW_SIZE};
const PRED_STEPS = {PRED_STEPS};
const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const predictBtn = document.getElementById('predict');

// Skala: piksela po metru (za prikaz). Origin je centar platna.
const SCALE = 45;
const W = canvas.width, H = canvas.height;
let points = [];   // klik pozicije u pikselima

function toMeters(px, py) {{
  // piksel -> metri (y obrnut da gore bude +)
  return [ (px - W/2) / SCALE, (H/2 - py) / SCALE ];
}}
function toPixels(mx, my) {{
  return [ W/2 + mx * SCALE, H/2 - my * SCALE ];
}}

function drawGrid() {{
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#eee';
  ctx.lineWidth = 1;
  for (let x = 0; x <= W; x += SCALE) {{ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }}
  for (let y = 0; y <= H; y += SCALE) {{ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }}
  // ose
  ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(0,H/2); ctx.lineTo(W,H/2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(W/2,0); ctx.lineTo(W/2,H); ctx.stroke();
}}

function drawPath(pts, color, dashed) {{
  if (pts.length === 0) return;
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.5;
  ctx.setLineDash(dashed ? [6,5] : []);
  ctx.beginPath();
  pts.forEach((p, i) => i === 0 ? ctx.moveTo(p[0],p[1]) : ctx.lineTo(p[0],p[1]));
  ctx.stroke();
  ctx.setLineDash([]);
  pts.forEach(p => {{ ctx.beginPath(); ctx.arc(p[0],p[1],4,0,2*Math.PI); ctx.fill(); }});
}}

function redraw(predPixels) {{
  drawGrid();
  drawPath(points, '#2563eb', false);
  if (predPixels && points.length > 0) {{
    // spoji poslednju tačku istorije sa predikcijom
    drawPath([points[points.length-1], ...predPixels], '#dc2626', true);
  }}
}}

canvas.addEventListener('click', (e) => {{
  if (points.length >= WINDOW) return;
  const r = canvas.getBoundingClientRect();
  points.push([e.clientX - r.left, e.clientY - r.top]);
  statusEl.textContent = `Uneto ${{points.length}} / ${{WINDOW}} tačaka`;
  predictBtn.disabled = (points.length !== WINDOW);
  redraw(null);
}});

document.getElementById('reset').addEventListener('click', () => {{
  points = [];
  statusEl.textContent = `Uneto 0 / ${{WINDOW}} tačaka`;
  predictBtn.disabled = true;
  redraw(null);
}});

predictBtn.addEventListener('click', async () => {{
  // Konvertuj klikove u metre
  const metri = points.map(p => toMeters(p[0], p[1]));
  // Poslednja tačka = T (trenutna pozicija). Relativne koord = svaka - T.
  const T = metri[metri.length - 1];
  const relX = [], relY = [];
  for (let i = 0; i < WINDOW - 1; i++) {{
    relX.push(metri[i][0] - T[0]);
    relY.push(metri[i][1] - T[1]);
  }}
  const features = [...relX, ...relY];

  statusEl.textContent = 'Računam predikciju...';
  try {{
    const resp = await fetch('/predict', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ features }})
    }});
    if (!resp.ok) {{ const err = await resp.json(); throw new Error(JSON.stringify(err.detail)); }}
    const data = await resp.json();

    // Predikcije su delte u odnosu na T. Kumulativno ih pretvaramo u apsolutne metre.
    const predPixels = data.predikcije.map(p => {{
      const mx = T[0] + p.delta_x;
      const my = T[1] + p.delta_y;
      return toPixels(mx, my);
    }});
    redraw(predPixels);
    statusEl.textContent = 'Gotovo — crveno je predikcija.';
  }} catch (e) {{
    statusEl.textContent = 'Greška: ' + e.message;
  }}
}});

drawGrid();
</script>
</body>
</html>
"""