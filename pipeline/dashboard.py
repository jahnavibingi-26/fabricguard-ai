"""FabricGuard AI - operator dashboard (Python standard library only, runs on the laptop or the board).

It reads the logs written by infer.py and shows, for the selected inspection:
  - the annotated image and a big PASS / WARNING / REJECT banner
  - the fixed advice (instant) and the Llama advice (appears when the background job finishes,
    tagged with its source: llama or fallback, plus seconds and tokens per second)
  - Confirm defect / False alarm buttons, saved to feedback.jsonl

Usage (two terminals, from ~/fabric_project/pipeline):
    python dashboard.py --logs ../logs --port 8000
    python infer.py --source <folder> --out ../logs --llama --delay 3
Then open http://localhost:8000   (from another computer: http://<board-ip>:8000)
"""
import argparse
import json
import os
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

LOCK = threading.Lock()


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


READ_ONLY = False


def build_state(logs, n=25):
    insp = [e for e in read_jsonl(os.path.join(logs, "inspections.jsonl")) if e.get("type") == "inspection"]
    llama = {e["image"]: e for e in read_jsonl(os.path.join(logs, "llama.jsonl"))}
    fb_all = read_jsonl(os.path.join(logs, "feedback.jsonl"))
    fb = {e["image"]: e for e in fb_all}
    counts = {"PASS": 0, "WARNING": 0, "REJECT": 0}
    lat = []
    for e in insp:
        counts[e["verdict"]] = counts.get(e["verdict"], 0) + 1
        lat.append(e["latency_ms"]["total_ms"])
    pool = [e for e in insp if not (e.get("live") and e["verdict"] == "PASS")]   # live PASS frames: counted, not listed
    live_file = os.path.join(logs, "latest.jpg")
    live_on = os.path.exists(live_file) and (datetime.now().timestamp() - os.path.getmtime(live_file)) < 5
    recent = []
    for e in pool[-n:][::-1]:
        item = dict(e)
        item["llama"] = llama.get(e["image"])
        item["feedback"] = fb.get(e["image"], {}).get("label")
        recent.append(item)
    real = [v for v in llama.values() if not v.get("cached")]
    return {
        "counts": counts, "total": len(insp),
        "mean_ms": round(sum(lat) / len(lat), 1) if lat else None,
        "fps": round(1000 / (sum(lat) / len(lat)), 1) if lat else None,
        "llama_seen": bool(llama), "live": live_on, "readonly": READ_ONLY,
        "llama_jobs": len(llama), "llama_fallbacks": sum(1 for v in llama.values() if v.get("source") == "fallback"),
        "llama_mean_s": round(sum(v["seconds"] for v in real) / len(real), 1) if real else None,
        "confirmed": sum(1 for e in fb.values() if e["label"] == "confirmed"),
        "false_alarm": sum(1 for e in fb.values() if e["label"] == "false_alarm"),
        "recent": recent,
    }


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FabricGuard AI - operator dashboard</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--ink:#1c2430;--mute:#6b7685;--line:#dde1e7;--pass:#2e9e55;--warn:#e0a000;--rej:#d63a2f}
@media (prefers-color-scheme:dark){:root{--bg:#14181f;--card:#1d232c;--ink:#e8ecf2;--mute:#97a1b0;--line:#2c3441}}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;padding:12px 18px;background:var(--card);border-bottom:1px solid var(--line)}
header h1{font-size:18px;margin:0 8px 0 0}.stat{font-size:13px;color:var(--mute)}.stat b{color:var(--ink);font-size:16px}
main{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:16px;padding:16px;max-width:1300px;margin:0 auto}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:16px}
.card h2{font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--mute);margin:0 0 8px}
#img{width:100%;border-radius:8px;display:block;background:#0003;min-height:120px}
#banner{font-size:30px;font-weight:700;color:#fff;border-radius:8px;padding:12px 16px;text-align:center}
.PASS{background:var(--pass)}.WARNING{background:var(--warn)}.REJECT{background:var(--rej)}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;color:#fff;font-size:12px;font-weight:600}
.tag{font-size:12px;color:var(--mute);margin-top:6px}.advice{margin:0 0 4px;white-space:pre-wrap}
button{font:inherit;padding:9px 16px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer;margin-right:8px}
button.ok{border-color:var(--pass)}button.bad{border-color:var(--rej)}button:disabled{opacity:.45;cursor:default}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:5px 6px;border-bottom:1px solid var(--line);text-align:left}
tr.row{cursor:pointer}tr.sel{background:#7a8aa022}.scroll{overflow-x:auto}
ul{margin:4px 0 0;padding-left:18px}
</style></head><body>
<header><h1>FabricGuard AI</h1>
<span class="stat">Inspected <b id="s_total">0</b></span>
<span class="stat">PASS <b id="s_pass">0</b></span><span class="stat">WARNING <b id="s_warn">0</b></span><span class="stat">REJECT <b id="s_rej">0</b></span>
<span class="stat">Vision <b id="s_ms">-</b> ms (<b id="s_fps">-</b> FPS)</span>
<span class="stat">Llama jobs <b id="s_llama">0</b> (fallbacks <b id="s_fb">0</b>, mean <b id="s_ls">-</b> s)</span>
<span class="stat">Operator: confirmed <b id="s_conf">0</b>, false alarms <b id="s_fa">0</b></span></header>
<main>
<section><div class="card" id="livecard" style="display:none"><h2>Live camera (annotated)</h2><img id="live" alt="live feed" style="width:100%;border-radius:8px;display:block"></div>
<div class="card"><h2 id="imgtitle">Latest alert</h2><img id="img" alt="annotated fabric image"></div>
<div class="card"><h2>History (click a row)</h2><div class="scroll"><table><thead><tr><th>Time</th><th>Image</th><th>Verdict</th><th>Top defect</th><th>Operator</th></tr></thead><tbody id="hist"></tbody></table></div>
<div class="tag"><a href="#" id="live">Follow latest</a></div></div></section>
<section><div id="banner" class="PASS">Waiting for data</div>
<div class="card" style="margin-top:16px"><h2>Defects found</h2><div id="dets">-</div></div>
<div class="card"><h2>Fixed advice (instant, expert-style table, not validated by an expert)</h2><p class="advice" id="fixed">-</p><div class="tag" id="disp"></div></div>
<div class="card"><h2 id="lh">Llama advice (on-device, ExecuTorch)</h2><p class="advice" id="llama">-</p><div class="tag" id="ltag"></div></div>
<div class="card"><h2>Operator decision</h2><button class="ok" id="b_ok">Confirm defect</button><button class="bad" id="b_fa">False alarm</button><div class="tag" id="fbmsg"></div></div>
</section></main>
<script>
let state=null, pinned=null;
const $=id=>document.getElementById(id);
const col={PASS:'var(--pass)',WARNING:'var(--warn)',REJECT:'var(--rej)'};
function cur(){ if(!state||!state.recent.length) return null; if(pinned){const f=state.recent.find(e=>e.image===pinned); if(f) return f;} return state.recent[0]; }
function render(){
  if(!state) return; const s=state;
  $('s_total').textContent=s.total; $('s_pass').textContent=s.counts.PASS; $('s_warn').textContent=s.counts.WARNING; $('s_rej').textContent=s.counts.REJECT;
  $('s_ms').textContent=s.mean_ms??'-'; $('s_fps').textContent=s.fps??'-'; $('s_llama').textContent=s.llama_jobs; $('s_fb').textContent=s.llama_fallbacks;
  $('s_ls').textContent=s.llama_mean_s??'-'; $('s_conf').textContent=s.confirmed; $('s_fa').textContent=s.false_alarm;
  const h=$('hist'); h.textContent='';
  const c=cur();
  s.recent.forEach(e=>{const tr=document.createElement('tr'); tr.className='row'+(c&&e.image===c.image?' sel':'');
    const top=e.detections&&e.detections.length?e.detections.slice().sort((a,b)=>b.conf-a.conf)[0].class:'-';
    [e.time.slice(11),e.image,null,top,e.feedback||'-'].forEach((v,i)=>{const td=document.createElement('td');
      if(i===2){const p=document.createElement('span');p.className='pill';p.style.background=col[e.verdict];p.textContent=e.verdict;td.appendChild(p);} else td.textContent=v; tr.appendChild(td);});
    tr.onclick=()=>{pinned=e.image;render();}; h.appendChild(tr);});
  if(!c) return;
  $('imgtitle').textContent=(pinned?'Selected: ':'Latest alert: ')+c.image;
  const src='/img/'+encodeURIComponent(c.image); if($('img').dataset.src!==src){$('img').dataset.src=src;$('img').src=src;}
  const b=$('banner'); b.className=c.verdict; b.textContent=c.verdict+(c.n_defects?'  ('+c.n_defects+' defect'+(c.n_defects>1?'s':'')+')':'');
  const d=$('dets'); d.textContent='';
  if(!c.detections.length) d.textContent='No defects above the confidence threshold.';
  else {const ul=document.createElement('ul'); c.detections.forEach(x=>{const li=document.createElement('li');
    li.textContent=x.class+' - '+x.severity+', confidence '+x.conf+', '+x.area_pct+'% of frame, '+x.location; ul.appendChild(li);}); d.appendChild(ul);}
  const a=c.advice_fixed;
  $('fixed').textContent=a?('Cause: '+a.cause+'. Action: '+a.action+'.'):'No action needed.'; $('disp').textContent=a?('Disposition: '+a.disposition):'';
  const L=c.llama, lt=$('ltag');
  if(!a){$('llama').textContent='-'; lt.textContent='';}
  else if(L){$('llama').textContent=L.text;
    $('lh').textContent=(L.source==='llama'?'Llama advice (on-device, ExecuTorch)':'Llama advice unavailable (showing fixed text)'); lt.textContent='Source: '+(L.source==='llama'?'Llama 3.2 1B (ExecuTorch)':'FALLBACK to fixed text (Llama output unavailable or failed the format check)')+
      ' | '+(L.cached?'reused from an identical earlier alert':L.seconds+' s')+(L.decode_token_per_sec?' | decode '+Math.round(L.decode_token_per_sec)+' tok/s':'')+(L.generated_tokens?' | '+L.generated_tokens+' tokens':'');}
  else if(s.live && (Date.now()-Date.parse(c.time))<45000){$('llama').textContent='Generating on-device...'; lt.textContent='The fixed advice above is valid now; the Llama text replaces it when ready. Frames that are not alert events get no Llama job.';}
  else if(s.llama_seen){$('llama').textContent='No Llama job for this frame.'; lt.textContent='Llama runs only for alert events (new defect, worse verdict or after the cooldown). The fixed advice above applies.';}
  else {$('llama').textContent='Not run for this inspection.'; lt.textContent='Start infer.py with --llama to enable it.';}
  const done=!!c.feedback, none=c.verdict==='PASS', ro=!!s.readonly;
  $('b_ok').disabled=done||none||ro; $('b_fa').disabled=done||none||ro;
  $('fbmsg').textContent=ro?'View only (shared link): operator buttons are disabled.':(none?'No action needed for PASS.':(done?'Recorded: '+c.feedback:''));
}
async function send(label){const c=cur(); if(!c) return; await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:c.image,label})}); poll();}
$('b_ok').onclick=()=>send('confirmed'); $('b_fa').onclick=()=>send('false_alarm'); $('live').onclick=e=>{e.preventDefault();pinned=null;render();};
async function poll(){try{const r=await fetch('/api/state'); state=await r.json(); render();}catch(e){}}
poll(); setInterval(poll,1000);
function refreshLive(){ if(state&&state.live){ $('livecard').style.display=''; const im=new Image(); im.onload=()=>{$('live').src=im.src;}; im.src='/live.jpg?'+Date.now(); } else { $('livecard').style.display='none'; } }
setInterval(refreshLive,350);
</script></body></html>"""


def make_handler(logs):
    annotated = os.path.join(logs, "annotated")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(200, json.dumps(build_state(logs)).encode(), "application/json")
            elif path == "/live.jpg":
                f = os.path.join(logs, "latest.jpg")
                if os.path.isfile(f):
                    self._send(200, open(f, "rb").read(), "image/jpeg")
                else:
                    self._send(404, b"no live feed", "text/plain")
            elif path.startswith("/img/"):
                name = os.path.splitext(os.path.basename(unquote(path[5:])))[0]   # basename blocks path tricks
                f = os.path.join(annotated, name + ".jpg")
                if os.path.isfile(f):
                    self._send(200, open(f, "rb").read(), "image/jpeg")
                else:
                    self._send(404, b"not found", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/feedback":
                return self._send(404, b"not found", "text/plain")
            if READ_ONLY:
                return self._send(403, b"view only", "text/plain")
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                image, label = str(data["image"]), data["label"]
                assert label in ("confirmed", "false_alarm")
            except Exception:
                return self._send(400, b"bad request", "text/plain")
            verdict = next((e["verdict"] for e in reversed(read_jsonl(os.path.join(logs, "inspections.jsonl")))
                            if e.get("image") == image), None)
            line = {"type": "feedback", "time": datetime.now().isoformat(timespec="seconds"),
                    "image": image, "system_verdict": verdict, "label": label}
            with LOCK, open(os.path.join(logs, "feedback.jsonl"), "a") as f:
                f.write(json.dumps(line) + "\n")
            self._send(200, b'{"ok": true}', "application/json")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from the local network; 127.0.0.1 = this machine only")
    ap.add_argument("--read-only", action="store_true", help="viewers cannot press Confirm / False alarm (use when sharing a link)")
    a = ap.parse_args()
    global READ_ONLY
    READ_ONLY = a.read_only
    os.makedirs(os.path.join(a.logs, "annotated"), exist_ok=True)
    print(f"Dashboard on http://localhost:{a.port}  (logs: {os.path.abspath(a.logs)})  host {a.host}"
          + ("  [VIEW ONLY]" if READ_ONLY else "") + "  Ctrl+C to stop")
    ThreadingHTTPServer((a.host, a.port), make_handler(a.logs)).serve_forever()


if __name__ == "__main__":
    main()
