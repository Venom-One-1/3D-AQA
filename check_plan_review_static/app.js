"use strict";
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const verdicts = {pending:"未复核",correct:"对应",false_positive:"不对应",uncertain:"无法判断"};
const supportLabels = {direct:"直接测量",proxy:"间接代理",unsupported:"无法测量"};
const operations = {joint_angle:"三点夹角",vector_angle:"两向量夹角",relative_position:"归一化相对位置",distance:"归一化距离"};
const joints = {Pelvis:"骨盆",L_Hip:"左髋",R_Hip:"右髋",Spine1:"下段脊柱",Spine2:"中段脊柱",Spine3:"上段脊柱",L_Knee:"左膝",R_Knee:"右膝",L_Ankle:"左踝",R_Ankle:"右踝",L_Foot:"左足",R_Foot:"右足",Neck:"颈部",L_Collar:"左锁骨",R_Collar:"右锁骨",Head:"头部关节点",L_Shoulder:"左肩",R_Shoulder:"右肩",L_Elbow:"左肘",R_Elbow:"右肘",L_Wrist:"左腕",R_Wrist:"右腕",L_Hand:"左手关节点",R_Hand:"右手关节点",HipMidpoint:"双髋中点",ShoulderMidpoint:"双肩中点",AnkleMidpoint:"双踝中点",WristMidpoint:"双腕中点"};
const normalizers = {torso_length:"躯干长：骨盆至颈部",shoulder_width:"肩宽：左肩至右肩",average_leg_length:"平均腿长：左右髋—膝—踝长度的均值"};
const components = {right:"right：正值向身体右侧，负值向身体左侧（left = -right）",up:"up：正值沿躯干向上",forward:"forward：正值向身体前方",value:"value：无符号夹角或距离"};
const state = {data:null,move:1,repeat:1,current:null,visible:[],drafts:{},conflicts:[],saving:false,inflight:{}};
const icon = name => `<i data-lucide="${name}"></i>`;
const refreshIcons = () => window.lucide?.createIcons();
const annotation = rid => state.drafts[rid]?.value || state.data.annotations[rid];
const comparable = a => JSON.stringify([a.manual_verdict,a.notes,a.reviewer]);
const same = (a,b) => comparable(a) === comparable(b);
function message(text, error=false) { $("message").hidden=!text; $("message").textContent=text; $("message").className="message"+(error?" error":""); }
function persistDrafts() {
  try { localStorage.setItem(state.storageKey, JSON.stringify(state.drafts)); }
  catch { message("浏览器无法保存本地草稿；当前未保存标注仍在页面中，请保存到服务器或导出 CSV。",true); }
}
function updateProgress() {
  const ids=Object.keys(state.data.records), done=ids.filter(id=>annotation(id).manual_verdict!=="pending").length;
  const dirty=Object.keys(state.drafts).length;
  $("progress").textContent=`已复核 ${done} / ${ids.length} · ${dirty ? dirty+" 项未保存" : "已同步"}`;
  $("save").disabled=state.saving || !dirty || !!state.conflicts.length;
  $("export").disabled=false;
}
function edit(patch) {
  const rid=state.current, base=state.drafts[rid]?.base || state.data.annotations[rid];
  const value={...annotation(rid),...patch,updated_at:new Date().toISOString()};
  if(same(value,state.data.annotations[rid]) && !state.inflight[rid]) delete state.drafts[rid];
  else state.drafts[rid]={base,value};
  persistDrafts(); updateProgress(); renderQueue(); renderVerdicts();
}
function activeCase() { return state.data.cases.find(c=>c.move_id===state.move && c.repeat===state.repeat); }
function renderQueue() {
  const scroll=$("queue").scrollTop;
  const c=activeCase();
  $("queueCount").textContent=`${c.record_ids.filter(id=>annotation(id).manual_verdict!=="pending").length}/${c.record_ids.length}`;
  $("queue").innerHTML=state.visible.map(id=>{
    const r=state.data.records[id],v=annotation(id).manual_verdict;
    const name={pending:"circle",correct:"check",false_positive:"x",uncertain:"circle-help"}[v];
    return `<button class="queue-item ${id===state.current?"active":""}" data-record="${esc(id)}" aria-current="${id===state.current?"true":"false"}" title="${esc(r.source_text)} · ${verdicts[v]}"><span class="cid">${esc(r.clause_id.slice(1))}</span><span class="text">${esc(r.source_text)}</span><span class="${v}">${icon(name)}</span></button>`;
  }).join("");
  $("queue").querySelectorAll("[data-record]").forEach(b=>b.onclick=()=>{state.current=b.dataset.record;render();});
  $("queue").scrollTop=scroll;
  refreshIcons();
}
function applyFilters(preferredClause) {
  const c=activeCase();
  state.visible=c.record_ids.filter(id=>{
    const r=state.data.records[id];
    const mode=$("triage").value,j=r.judge;
    const selected=!state.data.judge || mode==="all" || (mode==="human_queue"?j.human_queue:mode==="audit_sample"?j.audit_sample:j.triage===mode);
    return selected && ($("support").value==="all" || r.support===$("support").value) && (!$("pending").checked || annotation(id).manual_verdict==="pending");
  });
  state.current=state.visible.find(id=>state.data.records[id].clause_id===preferredClause) || (state.visible.includes(state.current)?state.current:state.visible[0]);
  $("repeats").innerHTML=state.data.cases.filter(x=>x.move_id===state.move).map(x=>`<button data-repeat="${x.repeat}" class="${x.repeat===state.repeat?"active":""}" aria-pressed="${x.repeat===state.repeat}">第 ${x.repeat} 次</button>`).join("");
  $("repeats").querySelectorAll("button").forEach(b=>b.onclick=()=>{const clause=state.data.records[state.current]?.clause_id;state.repeat=Number(b.dataset.repeat);applyFilters(clause);});
  render();
}
function geometry(q,e) {
  const p=q.points.map(x=>joints[x]||x);
  let rows;
  if(q.operation==="joint_angle") rows=[["顶点",p[1]],["两条边",`${p[1]} → ${p[0]}；${p[1]} → ${p[2]}`],["含义","0–180°，180° 表示三点共线、两边反向"]];
  else if(q.operation==="vector_angle") rows=[["向量 A",`${p[0]} → ${p[1]}`],["向量 B",`${p[2]} → ${p[3]}`],["含义","0–180° 无符号夹角；0° 同向，180° 反向"]];
  else if(q.operation==="relative_position") rows=[["目标点",p[0]],["原点",p[1]],["计算",`${p[0]} − ${p[1]}，投影到身体坐标轴后归一化`],["分量",e.components.map(x=>components[x]).join("；")]];
  else rows=[["两点",`${p[0]} 与 ${p[1]}`],["计算","三维欧氏距离，除以归一化长度"]];
  if(q.normalizer) rows.push(["归一化",normalizers[q.normalizer]||q.normalizer]);
  if(q.coordinate_frame) rows.push(["坐标系","torso_v1：随身体转动；向上轴沿骨盆至颈部，不是重力竖直方向"]);
  rows.push(["单位",q.unit==="degree"?"度（degree）":"无量纲比例（ratio）"]);
  return `<dl>${rows.map(([k,v])=>`<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;
}
function queryHtml(record) {
  if(!record.evidence.length) return '<p class="muted">无测量查询：模型判定本条无法由当前结束定势几何信息验证。</p>';
  return record.evidence.map(e=>{
    const q=record.query_definitions[e.query_id];
    return `<section class="query"><div class="query-header"><h3>${esc(operations[q.operation])}</h3><code>${esc(e.query_id)}</code></div><div class="query-grid"><div class="geometry">${geometry(q,e)}<small>${q.points.map(esc).join(" · ")}</small></div><div class="query-analysis"><span class="label">模型选取理由</span><p>${esc(e.purpose)}</p><div class="relation"><span class="label">模型预期关系 · 待复核</span><p>${esc(e.expected_relation)}</p></div></div></div></section>`;
  }).join("");
}
function renderVerdicts() {
  if(!state.current || !$("verdicts"))return;
  const v=annotation(state.current).manual_verdict;
  $("verdicts").innerHTML=Object.entries(verdicts).map(([key,label])=>`<button data-verdict="${key}" class="${key} ${key===v?"active":""}" aria-pressed="${key===v}">${label}</button>`).join("");
  $("verdicts").querySelectorAll("button").forEach(b=>b.onclick=()=>edit({manual_verdict:b.dataset.verdict}));
}
function judgeHtml(record) {
  if(!record.judge)return "";
  const j=record.judge,result=j.judgment;
  const names={relevance:"测量关联",direction:"方向与符号",coverage:"覆盖程度",scope:"推断边界"};
  const labels={pass:"通过",fail:"发现问题",uncertain:"不确定"};
  const route={needs_review:"需要人工裁决",auto_pass_provisional:"预审通过 · 未校准",auto_return:"程序退回",pending_judge:"缺少合法评审"};
  return `<section class="judge-section"><h2>独立预审 · ${esc(state.data.judge.requested_model)}</h2><p><strong>${esc(route[j.triage])}</strong>${j.audit_sample?" · 随机抽检":""}${j.human_model_disagreement?" · 与旧人工标注存在分歧":""}</p><p>${esc(result?.summary||j.triage_reason)}</p><div class="judge-grid">${result?Object.entries(result.dimensions).map(([key,d])=>`<div><h3>${names[key]} · ${labels[d.verdict]}</h3><p>${esc(d.reason)}</p></div>`).join(""):""}</div>${result?.issues.length?`<details open><summary>具体问题依据（${result.issues.length}）</summary>${result.issues.map(i=>`<div class="judge-issue"><h3>${names[i.dimension]} · ${esc(i.code)}</h3><p>${esc(i.reason)}</p><dl><dt>要领原文</dt><dd>${esc(i.source_quote)}</dd><dt>计划原文</dt><dd>${esc(i.plan_quote)}</dd></dl></div>`).join("")}</details>`:""}<p class="footnote">数值探针 ${esc(JSON.stringify(j.probe_counts))}；仅验证几何接口，不证明文本关系正确。预审未经独立校准，不替代人工结论。</p></section>`;
}
function render() {
  renderQueue();updateProgress();
  if(!state.current){$("content").innerHTML='<p class="empty">当前筛选下没有条目。</p>';return;}
  const r=state.data.records[state.current],c=activeCase(),a=annotation(state.current),index=state.visible.indexOf(state.current);
  const technique=esc(c.technique).replace(esc(r.source_text),`<mark>${esc(r.source_text)}</mark>`);
  const others=state.data.cases.filter(x=>x.move_id===c.move_id && x.repeat!==c.repeat).map(x=>{
    const other=x.record_ids.map(id=>state.data.records[id]).find(x=>x.clause_id===r.clause_id);
    return `<section class="other-plan"><h3>第 ${x.repeat} 次 · ${supportLabels[other.support]}</h3><p>${esc(other.endpoint_interpretation)}</p>${queryHtml(other)}<div class="limits"><ul>${other.limitations.map(x=>`<li>${esc(x)}</li>`).join("")}</ul></div><button data-switch="${x.repeat}">复核第 ${x.repeat} 次的本条要领</button></section>`;
  }).join("");
  $("content").innerHTML=`<div class="context-head"><div><small>${esc(c.case_id)} · ${esc(r.clause_id)} · 修复版</small><h1>${c.move_id}. ${esc(c.move_name)}</h1></div><div class="navigation"><button id="previous" class="icon-button" title="上一条要领" aria-label="上一条要领" ${index<=0?"disabled":""}>${icon("chevron-left")}</button><span>${index+1}/${state.visible.length}</span><button id="next" class="icon-button" title="下一条要领" aria-label="下一条要领" ${index===state.visible.length-1?"disabled":""}>${icon("chevron-right")}</button></div></div>
    <section class="context"><div><h2>动作要领原文</h2><p class="technique">${technique}</p></div><figure style="margin:0"><figcaption class="caption"><span>参考锚点 · BV1WE411W7JB</span><span>${Number(c.reference_time).toFixed(3)} s</span></figcaption><button class="image-open" id="openImage" title="放大标准参考结束定势" aria-label="放大标准参考结束定势"><img src="${esc(c.image)}" alt="${esc(c.move_name)}参考结束定势"></button><p class="footnote">该图仅用于人工复核，未输入计划生成模型。</p></figure></section>
    <section class="clause"><div class="clause-head"><h2>${esc(r.clause_id)} · ${esc(r.source_text)}</h2><span class="badge ${esc(r.support)}">模型声明：${supportLabels[r.support]}</span></div><p>${esc(r.endpoint_interpretation)}</p>${queryHtml(r)}<div class="limits"><h3>模型声明的测量局限</h3><ul>${r.limitations.map(x=>`<li>${esc(x)}</li>`).join("")}</ul></div></section>
    ${judgeHtml(r)}
    <section class="review-form"><div><h2>方案是否对应本条动作要领？</h2><div class="verdicts" id="verdicts" role="group" aria-label="对应关系复核"></div><p class="footnote">仅复核方案关联及解释的合理性，不评价图中动作质量；合理的“无法测量”也可标为“对应”。</p></div><div class="annotation-fields"><label>复核人<input id="reviewer" maxlength="100" value="${esc(a.reviewer)}"></label><label>备注<textarea id="notes" maxlength="4000" placeholder="测量部位、方向、关联或局限的问题">${esc(a.notes)}</textarea></label></div></section>
    <details><summary>同一要领的其他生成轮次</summary>${others}</details>
    <details><summary>本计划的修复记录与既有审查提示</summary><p class="footnote">程序修复不代表语义验证通过；以下为此前记录，不是本次人工结论。</p><ul>${c.semantic_warnings.map(x=>`<li>${esc(x)}</li>`).join("")}</ul><pre>${esc(JSON.stringify(c.repair_changes,null,2))}</pre></details>
    <details><summary>本条原始结构化内容（修复版）</summary><pre>${esc(JSON.stringify({check:{clause_id:r.clause_id,support:r.support,endpoint_interpretation:r.endpoint_interpretation,evidence:r.evidence,limitations:r.limitations},queries:r.query_definitions},null,2))}</pre></details>`;
  renderVerdicts();refreshIcons();
  $("previous").onclick=()=>navigate(-1);$("next").onclick=()=>navigate(1);
  $("notes").oninput=e=>edit({notes:e.target.value});$("reviewer").oninput=e=>edit({reviewer:e.target.value});
  $("openImage").onclick=()=>{$("largeImage").src=c.image;$("imageDialog").showModal();};
  $("content").querySelectorAll("[data-switch]").forEach(b=>b.onclick=()=>{state.repeat=Number(b.dataset.switch);applyFilters(r.clause_id);});
}
function navigate(delta) {
  if(!state.current)return;
  const i=state.visible.indexOf(state.current),next=state.visible[i+delta];
  if(next){state.current=next;render();}
}
async function request(url, options={}) {
  const control=new AbortController(),timeout=setTimeout(()=>control.abort(),15000);
  try {
    const response=await fetch(url,{...options,signal:control.signal});
    if(!response.ok){const text=await response.text();let error=text;try{error=JSON.parse(text).error||text;}catch{} throw new Error(error);}
    return await response.json();
  } finally {clearTimeout(timeout);}
}
async function save() {
  if(state.saving || state.conflicts.length || !Object.keys(state.drafts).length)return;
  const snapshot=JSON.parse(JSON.stringify(state.drafts));
  state.inflight=snapshot;
  state.saving=true;updateProgress();message("正在保存…");
  try {
    const result=await request("/api/review",{method:"POST",headers:{"Content-Type":"application/json","X-Review-Token":state.data.token},body:JSON.stringify({revision:state.data.revision,updates:Object.entries(snapshot).map(([record_id,d])=>({record_id,...d.value}))})});
    state.data.annotations=result.annotations;state.data.revision=result.revision;
    for(const [rid,d] of Object.entries(state.drafts)) {
      if(snapshot[rid] && same(d.value,snapshot[rid].value)) delete state.drafts[rid];
      else d.base=result.annotations[rid];
    }
    persistDrafts();message("复核进度已保存。");
  } catch(error) {message(`保存失败：${error.message}。本地草稿仍保留，可导出 CSV 备份；连接恢复后重试。若服务器版本冲突，请刷新页面核对。`,true);}
  finally {state.inflight={};state.saving=false;updateProgress();}
}
function exportCsv() {
  const cell=value=>{let text=String(value??"");if(/^[\s]*[=+@-]/.test(text))text="'"+text;return '"'+text.replace(/"/g,'""')+'"';};
  const lines=[state.data.export_fields.map(cell).join(",")];
  for(const [rid,row] of Object.entries(state.data.export_rows)) {
    const record={...row,...annotation(rid),save_status:state.drafts[rid]?"local_draft":"saved"};
    lines.push(state.data.export_fields.map(f=>cell(record[f])).join(","));
  }
  const url=URL.createObjectURL(new Blob(["\ufeff"+lines.join("\r\n")+"\r\n"],{type:"text/csv;charset=utf-8"}));
  const link=document.createElement("a");link.href=url;link.download="check_plan_review"+(Object.keys(state.drafts).length?"_with_drafts":"")+".csv";document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  message(Object.keys(state.drafts).length?"已导出全部条目（包含未保存草稿，save_status=local_draft）；服务器进度未因此改变。":"已导出全部复核条目。");
}
async function load() {
  state.data=await request("/api/data");
  state.storageKey="check-plan-review:"+state.data.dataset_id+":"+state.data.state_path;
  try {
    const stored=JSON.parse(localStorage.getItem(state.storageKey)||"{}");
    for(const [rid,d] of Object.entries(stored)) {
      if(!state.data.records[rid] || !d?.value || !d?.base || !(d.value.manual_verdict in verdicts))continue;
      const saved=state.data.annotations[rid];
      if(same(d.value,saved))continue;
      state.drafts[rid]=d;
      if(!same(d.base,saved))state.conflicts.push(rid);
    }
  } catch {message("无法读取浏览器草稿。服务器已保存的标注不受影响。",true);}
  if(state.conflicts.length) $("conflict").hidden=false;
  else if(Object.keys(state.drafts).length)message(`已恢复 ${Object.keys(state.drafts).length} 项本地未保存草稿。`);
  const moves=[...new Map(state.data.cases.map(c=>[c.move_id,c.move_name])).entries()];
  state.move=moves[0][0];state.repeat=state.data.cases[0].repeat;
  $("model").textContent=state.data.model+" · 结束定势几何";
  $("triageFilter").hidden=!state.data.judge;
  if(state.data.judge)$("model").textContent+=` · 预审人工队列 ${state.data.judge.human_queue_count}/${state.data.judge.total}`;
  $("move").innerHTML=moves.map(([id,name])=>`<option value="${id}">${id}. ${esc(name)}</option>`).join("");$("move").disabled=false;
  applyFilters();
}
function resolveConflict(keepLocal) {
  for(const id of state.conflicts) {
    if(keepLocal)state.drafts[id].base=state.data.annotations[id];else delete state.drafts[id];
  }
  state.conflicts=[];$("conflict").hidden=true;persistDrafts();render();
}
$("move").onchange=()=>{state.move=Number($("move").value);applyFilters();};
$("support").onchange=()=>applyFilters();$("pending").onchange=()=>applyFilters();
$("triage").onchange=()=>applyFilters();
$("save").onclick=save;$("export").onclick=exportCsv;$("closeImage").onclick=()=>$("imageDialog").close();
$("keepDraft").onclick=()=>{if(confirm("以本地草稿保留冲突项？下一次保存将覆盖服务器上这些项的标注。"))resolveConflict(true);};
$("useServer").onclick=()=>{if(confirm("放弃冲突项的本地草稿，采用服务器已保存标注？"))resolveConflict(false);};
window.addEventListener("beforeunload",e=>{if(Object.keys(state.drafts).length){e.preventDefault();e.returnValue="";}});
load().catch(error=>{message("载入失败："+error.message,true);$("content").innerHTML='<p class="empty">无法读取实验数据。</p>';});
