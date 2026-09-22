"""Launch the local browser UI for reviewing endpoint feedback."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlparse

from aqa3d.feedback_review import FeedbackReviewStore


PROJECT = Path(__file__).resolve().parent


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>定势反馈人工复核</title>
<style>
:root{--bg:#f6f7f8;--paper:#fff;--ink:#1c2329;--muted:#65717b;--line:#d9dee2;--accent:#176b5b;--accent2:#d9eee8;--warn:#aa5a16;--danger:#a33a32;--shadow:0 2px 12px rgba(21,31,38,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;letter-spacing:0}
button,select,input,textarea{font:inherit;letter-spacing:0}button{cursor:pointer}.topbar{position:sticky;top:0;z-index:10;background:#182126;color:#fff;border-bottom:1px solid #29343a}
.topbar-inner{max-width:1500px;margin:auto;min-height:64px;padding:10px 24px;display:flex;align-items:center;gap:18px}.brand{font-size:18px;font-weight:700;white-space:nowrap}.progress-wrap{flex:1;min-width:180px}.progress-text{display:flex;justify-content:space-between;font-size:13px;color:#dce4e7}.progress{height:7px;margin-top:5px;background:#39454b}.progress>div{height:100%;background:#54b49f;transition:width .2s}.export{border:1px solid #6e7a80;background:transparent;color:white;padding:8px 13px;border-radius:4px}
.layout{max-width:1500px;margin:0 auto;display:grid;grid-template-columns:230px minmax(0,1fr);min-height:calc(100vh - 65px)}aside{border-right:1px solid var(--line);padding:22px 18px;background:#eef1f2}aside label{display:block;color:var(--muted);font-size:12px;margin:0 0 5px}select,input,textarea{width:100%;border:1px solid #bdc7cc;background:white;color:var(--ink);border-radius:4px;padding:8px}aside select{margin-bottom:15px}.queue{margin-top:18px;border-top:1px solid var(--line);padding-top:14px}.queue-row{display:flex;justify-content:space-between;padding:5px 0;color:var(--muted)}.queue-row strong{color:var(--ink)}
main{padding:24px 30px 60px;min-width:0}.context{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;border-bottom:1px solid var(--line);padding-bottom:16px}.eyebrow{color:var(--accent);font-weight:700;font-size:13px}.context h1{font-size:25px;margin:2px 0 8px}.technique{max-width:1050px;margin:0;color:#354149}.record-count{font-variant-numeric:tabular-nums;color:var(--muted);white-space:nowrap}.summary{padding:16px 0;border-bottom:1px solid var(--line)}.summary h2,.visuals h2,.metric h2{font-size:16px;margin:0 0 7px}.summary p{margin:0;max-width:1200px}
.visuals{padding:20px 0;border-bottom:1px solid var(--line)}.image-pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}.figure{margin:0;min-width:0}.figure figcaption{font-weight:650;margin-bottom:7px}.figure img{display:block;width:100%;height:310px;object-fit:contain;object-position:center;background:#111923;border:1px solid #bcc6cb}.teacher-figure{margin-top:18px}.teacher-figure img{height:auto;max-height:500px}
.metric{padding:22px 0}.metric-head{display:flex;align-items:flex-start;justify-content:space-between;gap:18px}.status{padding:4px 8px;border-radius:3px;background:#e9edef;color:#49545a;font-size:13px;white-space:nowrap}.status.candidate{background:#e3f2ed;color:#145b4d}.status.review{background:#fff0dc;color:#86460f}.aspect{color:#37444b;margin:0 0 18px}.numbers{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.number{padding:13px 15px;border-right:1px solid var(--line)}.number:last-child{border-right:0}.number span{display:block;color:var(--muted);font-size:12px}.number strong{display:block;font-size:17px;margin-top:2px;font-variant-numeric:tabular-nums}.evidence{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:16px 0}.evidence section{border-left:3px solid #9da9ae;padding-left:12px}.evidence h3{font-size:13px;margin:0 0 5px}.evidence p{margin:0;color:#455159}.suggestion{border-left-color:var(--accent)!important}.review-reason{border-left-color:var(--warn)!important}
.review-panel{background:var(--paper);box-shadow:var(--shadow);border:1px solid var(--line);padding:18px;margin-top:6px}.review-panel h2{font-size:17px;margin:0 0 4px}.review-note{color:var(--muted);margin:0 0 14px}.verdicts{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.verdict{border:1px solid #aeb9be;background:#fff;padding:11px;border-radius:4px;font-weight:650}.verdict[data-value=correct].active{background:#dcefe8;border-color:#358a75;color:#125847}.verdict[data-value=false_positive].active{background:#f5dfdc;border-color:#b4544c;color:#7c2822}.verdict[data-value=uncertain].active{background:#f8e9d4;border-color:#b87a37;color:#754511}.form-grid{display:grid;grid-template-columns:220px 220px 1fr;gap:12px;margin-top:14px}.form-grid label{font-size:12px;color:var(--muted)}.form-grid input,.form-grid select,.form-grid textarea{display:block;margin-top:5px}.form-grid textarea{height:72px;resize:vertical}.save-state{height:20px;color:var(--muted);font-size:12px;margin-top:8px}.navigation{display:flex;justify-content:space-between;margin-top:18px}.nav{border:1px solid #9ba8ae;background:#fff;padding:9px 16px;border-radius:4px}.nav.primary{background:var(--accent);color:#fff;border-color:var(--accent)}.nav:disabled{opacity:.4;cursor:not-allowed}.empty{padding:60px;text-align:center;color:var(--muted)}
@media(max-width:900px){.layout{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid var(--line)}main{padding:20px 16px}.image-pair,.evidence,.form-grid{grid-template-columns:1fr}.figure img{height:auto;max-height:420px}.numbers{grid-template-columns:1fr 1fr}.number:nth-child(2){border-right:0}.topbar-inner{flex-wrap:wrap}.brand{width:100%}}
</style>
</head>
<body>
<header class="topbar"><div class="topbar-inner"><div class="brand">定势反馈人工复核</div><div class="progress-wrap"><div class="progress-text"><span id="progressLabel">载入中</span><span id="progressDetail"></span></div><div class="progress"><div id="progressBar"></div></div></div><button class="export" id="exportButton">导出 CSV</button></div></header>
<div class="layout">
<aside><label for="studentFilter">学生</label><select id="studentFilter"></select><label for="moveFilter">招式</label><select id="moveFilter"></select><label for="statusFilter">复核状态</label><select id="statusFilter"><option value="all">全部</option><option value="pending">未复核</option><option value="completed">已复核</option></select><div class="queue"><div class="queue-row"><span>候选建议</span><strong id="candidateCount">0</strong></div><div class="queue-row"><span>需复核</span><strong id="reviewCount">0</strong></div><div class="queue-row"><span>当前筛选</span><strong id="filteredCount">0</strong></div></div></aside>
<main id="content"><div class="empty">正在载入复核数据...</div></main>
</div>
<script>
const state={data:null,filtered:[],index:0,saving:false};
const $=id=>document.getElementById(id);
const escapeHtml=s=>String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const n=(v,d=3)=>v===null||v===undefined?'NA':Number(v).toFixed(d);
const verdictZh={pending:'未复核',correct:'正确',false_positive:'误判',uncertain:'无法判断'};
const causeZh={'':'未选择',reconstruction:'3D 重建',alignment:'时序/边界对齐',reference_range:'参考范围',rule:'反馈规则',temporal_window:'定势窗口',other:'其他'};
const directionZh={above:'偏高',below:'偏低',within:'区间内'};
async function load(){const response=await fetch('/api/data');if(!response.ok)throw new Error(await response.text());state.data=await response.json();buildFilters();applyFilters();updateProgress();}
function buildFilters(){const students=[...new Set(state.data.records.map(r=>r.student_id))];$('studentFilter').innerHTML='<option value="all">全部学生</option>'+students.map(x=>`<option value="${escapeHtml(x)}">Student ${escapeHtml(x)}</option>`).join('');const moves=[...new Map(state.data.records.map(r=>[r.move_id,r.move_name_zh])).entries()];$('moveFilter').innerHTML='<option value="all">全部招式</option>'+moves.map(([id,name])=>`<option value="${id}">${id}. ${escapeHtml(name)}</option>`).join('');}
function applyFilters(){const previous=state.filtered[state.index]?.record_id;const sid=$('studentFilter').value,mid=$('moveFilter').value,status=$('statusFilter').value;state.filtered=state.data.records.filter(r=>(sid==='all'||r.student_id===sid)&&(mid==='all'||String(r.move_id)===mid)&&(status==='all'||(status==='pending'?(r.annotation.manual_verdict==='pending'):(r.annotation.manual_verdict!=='pending'))));let found=state.filtered.findIndex(r=>r.record_id===previous);state.index=found>=0?found:0;$('filteredCount').textContent=state.filtered.length;render();}
function updateProgress(){const p=state.data.progress,percent=p.total?100*p.completed/p.total:0;$('progressLabel').textContent=`已完成 ${p.completed} / ${p.total}`;$('progressDetail').textContent=`正确 ${p.counts.correct} · 误判 ${p.counts.false_positive} · 无法判断 ${p.counts.uncertain}`;$('progressBar').style.width=percent+'%';$('candidateCount').textContent=state.data.records.filter(r=>r.decision==='feedback_candidate').length;$('reviewCount').textContent=state.data.records.filter(r=>r.decision==='needs_review').length;}
function render(){const root=$('content');if(!state.filtered.length){root.innerHTML='<div class="empty">当前筛选条件下没有待显示的指标。</div>';return;}const r=state.filtered[state.index],ref=r.teacher_reference,a=r.annotation;const reasons=r.review_reasons_zh.length?r.review_reasons_zh.join('；'):'无';const suggestion=r.feedback||'该项暂不生成纠正建议。';root.innerHTML=`
<section class="context"><div><div class="eyebrow">Student ${escapeHtml(r.student_id)} · ${r.move_id}. ${escapeHtml(r.move_name_zh)}</div><h1>结束定势人工复核</h1><p class="technique">${escapeHtml(r.final_technique_step)}</p></div><div class="record-count">${state.index+1} / ${state.filtered.length}</div></section>
<section class="summary"><h2>简短训练提示</h2><p>${escapeHtml(r.coach_summary)}</p></section>
<section class="visuals"><div class="image-pair"><figure class="figure"><figcaption>学生结束定势 · ${n(r.boundary_time_seconds,3)}s</figcaption><img src="${r.images.student}" alt="学生结束定势"></figure><figure class="figure"><figcaption>标准参考锚点 · BV1WE411W7JB</figcaption><img src="${r.images.reference}" alt="标准参考锚点"></figure></div><figure class="figure teacher-figure"><figcaption>十位教师结束定势对照</figcaption><img src="${r.images.teachers}" alt="十位教师结束定势对照"></figure></section>
<section class="metric"><div class="metric-head"><div><h2>${escapeHtml(r.metric_label_zh)}</h2><p class="aspect">${escapeHtml(r.technique_aspect)}</p></div><span class="status ${r.decision==='feedback_candidate'?'candidate':'review'}">${escapeHtml(r.decision_zh)}</span></div><div class="numbers"><div class="number"><span>学生窗口中位数</span><strong>${n(r.value)} ${escapeHtml(r.unit)}</strong></div><div class="number"><span>中心帧值</span><strong>${n(r.center_value)} ${escapeHtml(r.unit)}</strong></div><div class="number"><span>教师 median ± 2MAD</span><strong>${n(ref.median_minus_2mad)} ~ ${n(ref.median_plus_2mad)}</strong></div><div class="number"><span>教师 P10–P90</span><strong>${n(ref.p10)} ~ ${n(ref.p90)}</strong></div></div><div class="evidence"><section class="review-reason"><h3>系统判断与复核原因</h3><p>${escapeHtml(directionZh[r.direction]||r.direction)}；${escapeHtml(reasons)}</p></section><section class="suggestion"><h3>候选训练建议</h3><p>${escapeHtml(suggestion)}</p></section></div></section>
<section class="review-panel"><h2>人工结论</h2><p class="review-note">“正确”表示当前候选建议或暂缓判断合理，不代表整套动作合格。</p><div class="verdicts">${['correct','false_positive','uncertain'].map(v=>`<button class="verdict ${a.manual_verdict===v?'active':''}" data-value="${v}">${verdictZh[v]}</button>`).join('')}</div><div class="form-grid"><label>疑似原因<select id="causeInput">${Object.entries(causeZh).map(([v,label])=>`<option value="${v}" ${a.suspected_cause===v?'selected':''}>${label}</option>`).join('')}</select></label><label>复核人<input id="reviewerInput" maxlength="100" value="${escapeHtml(a.reviewer)}"></label><label>备注<textarea id="notesInput" maxlength="2000">${escapeHtml(a.notes)}</textarea></label></div><div class="save-state" id="saveState">${a.manual_verdict==='pending'?'尚未复核':'已保存：'+verdictZh[a.manual_verdict]}</div></section>
<div class="navigation"><button class="nav" id="previousButton" ${state.index===0?'disabled':''}>上一项</button><button class="nav primary" id="nextButton" ${state.index===state.filtered.length-1?'disabled':''}>下一项</button></div>`;
root.querySelectorAll('.verdict').forEach(button=>button.addEventListener('click',()=>save(button.dataset.value,true)));$('causeInput').addEventListener('change',()=>save(a.manual_verdict,false));$('reviewerInput').addEventListener('change',()=>save(a.manual_verdict,false));$('notesInput').addEventListener('change',()=>save(a.manual_verdict,false));$('previousButton').onclick=()=>navigate(-1);$('nextButton').onclick=()=>navigate(1);
}
async function save(verdict,advance){if(state.saving)return;const r=state.filtered[state.index];state.saving=true;$('saveState').textContent='保存中...';try{const payload={record_id:r.record_id,manual_verdict:verdict,suspected_cause:$('causeInput').value,reviewer:$('reviewerInput').value,notes:$('notesInput').value};const response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!response.ok)throw new Error(await response.text());const result=await response.json();r.annotation=result.annotation;state.data.progress=result.progress;updateProgress();if(advance&&state.index<state.filtered.length-1){state.index++;render();}else{render();}}catch(error){$('saveState').textContent='保存失败：'+error.message;}finally{state.saving=false;}}
function navigate(delta){state.index=Math.max(0,Math.min(state.filtered.length-1,state.index+delta));render();window.scrollTo({top:0,behavior:'smooth'});}
['studentFilter','moveFilter','statusFilter'].forEach(id=>$(id).addEventListener('change',applyFilters));$('exportButton').addEventListener('click',()=>{window.location='/api/export';});document.addEventListener('keydown',event=>{if(event.target.matches('input,textarea,select'))return;if(event.key==='ArrowLeft')navigate(-1);if(event.key==='ArrowRight')navigate(1);});
load().catch(error=>{$('content').innerHTML=`<div class="empty">载入失败：${escapeHtml(error.message)}</div>`});
</script>
</body></html>'''


class ReviewRequestHandler(BaseHTTPRequestHandler):
    store: FeedbackReviewStore

    def _headers(self, status, content_type, length=None, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if length is not None:
            self.send_header("Content-Length", str(length))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _send_bytes(self, data, content_type, status=HTTPStatus.OK, extra=None):
        self._headers(status, content_type, len(data), extra)
        self.wfile.write(data)

    def _send_json(self, data, status=HTTPStatus.OK):
        self._send_bytes(json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                         "application/json; charset=utf-8", status)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/":
                self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/data":
                self._send_json(self.store.client_payload())
            elif path == "/api/export":
                self._send_bytes(self.store.export_csv(), "text/csv; charset=utf-8",
                    extra={"Content-Disposition": 'attachment; filename="manual_review_labeled.csv"'})
            elif path.startswith("/api/image/"):
                token = path.removeprefix("/api/image/")
                image = self.store.image_paths.get(token)
                if image is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                data = image.read_bytes()
                self._send_bytes(data, mimetypes.guess_type(image.name)[0] or "application/octet-stream",
                                 extra={"Cache-Control": "private, max-age=300"})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (OSError, ValueError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        if path != "/api/review":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16384:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(length))
            annotation = self.store.update(data.get("record_id", ""), data.get("manual_verdict", ""),
                data.get("suspected_cause", ""), data.get("reviewer", ""), data.get("notes", ""))
            self._send_json({"annotation": annotation, "progress": self.store.progress()})
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format, *args):
        if args and str(args[1]).startswith("4"):
            super().log_message(format, *args)


def create_server(result_root: Path, host: str, port: int):
    store = FeedbackReviewStore(result_root)
    handler = type("BoundReviewRequestHandler", (ReviewRequestHandler,), {"store": store})
    return ThreadingHTTPServer((host, port), handler), store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path,
        default=PROJECT/"endpoint_feedback_results/first3_five_students")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    server, store = create_server(args.result_root, args.host, args.port)
    print(f"Reviewing {len(store.review_records)} records at http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
