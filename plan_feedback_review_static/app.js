'use strict';
const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = v => typeof v === 'number' ? v.toFixed(3) : '—';
const verdicts = {pending:'未标注',correct:'正确',false_positive:'误判',uncertain:'无法判断'};
const state = {data:null,current:null,filtered:[],drafts:{},conflicts:[],saving:false,inflight:{}};
const icons = () => window.lucide?.createIcons();
const annotation = id => state.drafts[id]?.value || state.data.annotations[id];
const same = (a,b) => JSON.stringify([a.manual_verdict,a.notes,a.reviewer]) === JSON.stringify([b.manual_verdict,b.notes,b.reviewer]);
const rid = (source,scope,id) => `${state.current.case_id}:${source}:${scope}:${id}`;
const complete = c => c.record_ids.every(id => annotation(id).manual_verdict !== 'pending');
function message(text, bad=false) { $('message').hidden=!text; $('message').textContent=text; $('message').className=bad?'error':''; }
function persist() {
  try { localStorage.setItem(state.key, JSON.stringify(state.drafts)); }
  catch { message('本地草稿写入失败；当前修改尚在页面中，未保存。',true); }
}
function progress() {
  const ids=Object.keys(state.data.records),dirty=Object.keys(state.drafts).length;
  $('progress').textContent=`已标注 ${ids.filter(id=>annotation(id).manual_verdict!=='pending').length} / ${ids.length}`;
  $('saveState').textContent=state.saving?'保存中…':dirty?`${dirty} 项未保存`:'已同步';
  $('save').disabled=state.saving || !dirty || !!state.conflicts.length;
  $('export').disabled=false;
}
function queue() {
  const status=$('status').value;
  state.filtered=state.data.cases.filter(c=>(!$('student').value || c.student_id===$('student').value)
    && (!$('move').value || String(c.move_id)===$('move').value)
    && (status==='all' || (status==='pending'&&!complete(c)) || (status==='completed'&&complete(c)) || (status==='flagged'&&c.validation_errors.length)));
  $('queue').innerHTML=state.filtered.map(c=>`<button class="case ${c.case_id===state.current?.case_id?'active':''}" data-case="${esc(c.case_id)}" aria-current="${c.case_id===state.current?.case_id}"><span>${esc(c.student_id)} · ${esc(c.move_name)}</span><small>${c.record_ids.filter(id=>annotation(id).manual_verdict!=='pending').length}/${c.record_ids.length}</small></button>`).join('');
}
function review(id) {
  const a=annotation(id);
  return `<div class="review" data-record="${esc(id)}"><div class="verdicts" role="radiogroup" aria-label="${esc(id)} 复核结论">${Object.entries(verdicts).map(([v,label])=>`<label class="verdict"><input type="radio" name="${esc(id)}" value="${v}" ${a.manual_verdict===v?'checked':''}>${label}</label>`).join('')}</div><textarea aria-label="${esc(id)} 备注" placeholder="备注" maxlength="4000">${esc(a.notes)}</textarea><div class="saved-meta">${esc(a.reviewer)} ${a.updated_at?esc(new Date(a.updated_at).toLocaleString()):''}</div></div>`;
}
function rangesRow(label,unit,student,ref) {
  return `<tr><td>${esc(label)}<br><small>${esc(unit)}</small></td><td>${fmt(student.value)} / ${fmt(student.center_value)}</td><td>${fmt(ref.median_minus_2mad)} – ${fmt(ref.median_plus_2mad)}</td><td>${fmt(ref.p10)} – ${fmt(ref.p90)}</td></tr>`;
}
function table(rows) { return `<div class="table-scroll"><table><thead><tr><th>指标</th><th>学生值 / 中心帧</th><th>教师 median ± 2MAD</th><th>教师 P10–P90</th></tr></thead><tbody>${rows}</tbody></table></div>`; }
const components={right:'身体右向（负值向左）',up:'躯干向上（非重力方向）',forward:'身体前向',value:'夹角 / 距离'};
function newEvidence(f) {
  const selected=state.current.new_evidence.filter(e=>f.evidence.some(c=>c.evidence_id===e.evidence_id));
  return table(selected.map(e=>rangesRow(e.evidence_id,e.definition.unit,e.student,e.reference)).join(''))
    +selected.map(e=>`<details><summary>${esc(e.evidence_id)} · 查询定义与十位教师</summary><p class="definition">${esc(e.definition.operation)} · ${esc(e.definition.points.join(' → '))}<br>${esc(components[e.component]||e.component)} · ${esc(e.definition.normalizer||'不归一化')}<br>${esc(e.definition.formula)}</p><p class="definition">${esc((e.definition.limitations||[]).join(' '))}</p><p class="definition">警告：${esc(e.warnings.join('；')||'无')}</p><div class="table-scroll"><table><thead><tr><th>教师</th><th>窗口中位数</th><th>中心帧</th><th>警告</th></tr></thead><tbody>${e.teacher_values.map(t=>`<tr><td>${esc(t.subject_id)}</td><td>${fmt(t.value)}</td><td>${fmt(t.center_value)}</td><td>${esc((t.warnings||[]).join('；')||'无')}</td></tr>`).join('')}</tbody></table></div></details>`).join('');
}
function oldEvidence(f) {
  return table(state.current.f_metrics.filter(m=>(f.evidence_metric_ids||[]).includes(m.metric_id))
    .map(m=>rangesRow(m.label,m.unit,m.student,m.reference)).join(''));
}
function finding(f,source) {
  const isNew=source==='plan_feedback';
  const label=isNew?(f.certainty==='supported'?'证据支持 · 模型声明':'谨慎判断 · 模型声明'):(f.assessment==='insufficient_evidence'?'证据不足':'可能的问题');
  const clauses=isNew?state.current.checks.filter(c=>(f.clause_ids||[]).includes(c.clause_id)).map(c=>`${c.clause_id} · ${c.text}`).join('；'):'';
  return `<article class="finding"><span class="badge ${isNew&&f.certainty==='tentative'?'warn':''}">${esc(f.finding_id)} · ${esc(label)}</span><h3>${esc(isNew?f.observation:f.problem)}</h3>${clauses?`<p class="definition">${esc(clauses)}</p>`:''}<div class="caption">判断依据</div><p>${esc(isNew?f.synthesis:f.explanation)}</p>${isNew?`<div class="caption">优先理由</div><p>${esc(f.why_priority)}</p>`:''}<div class="caption">训练指导</div><p class="guidance">${esc((isNew?f.training_guidance:f.guidance)||'暂无纠正建议')}</p><div class="caption">局限与不确定性</div><p>${esc(isNew?f.caveat:f.uncertainty)}</p><details><summary>指标值与教师参考范围</summary>${isNew?newEvidence(f):oldEvidence(f)}</details>${review(rid(source,'finding',f.finding_id))}</article>`;
}
function panel(source,title) {
  const c=state.current,r=c.responses[source],isNew=source==='plan_feedback';
  return `<div class="method ${isNew?'f':'rule'}" data-source="${source}"><h2>${esc(title)}<small>${r.findings.length} 条 findings</small></h2><div class="model-label">${esc(state.data.models[source])}</div>${isNew&&c.validation_errors.length?`<div class="validation-note"><strong>未通过自动可靠性约束 · 非人工误判</strong><ul>${c.validation_errors.map(x=>`<li>${esc(x)}</li>`).join('')}</ul></div>`:''}<section class="summary"><h3>教练式总结 · coach_summary</h3><p>${esc(r.coach_summary)}</p>${review(rid(source,'coach_summary','summary'))}</section>${r.findings.map(f=>finding(f,source)).join('')||'<p class="empty">未返回 findings</p>'}${isNew?`<details><summary>未选择的候选与检查空缺</summary>${(r.not_selected||[]).map(x=>`<p>${esc(x.clause_ids.join('、'))}：${esc(x.reason)}</p>`).join('')}<ul>${c.excluded_clauses.map(x=>`<li>${esc(x.clause_id)} ${esc(x.text)}：${esc(x.reason)}</li>`).join('')}</ul></details>`:''}<details><summary>完整返回 JSON</summary><pre>${esc(JSON.stringify(r,null,2))}</pre></details></div>`;
}
function render() {
  queue();progress();
  if(!state.current){$('content').innerHTML='<p class="empty">没有符合筛选条件的案例。</p>';return;}
  const c=state.current,i=state.filtered.findIndex(x=>x.case_id===c.case_id);
  $('content').innerHTML=`<div class="title"><div><small>${esc(c.case_id)} · ${esc(c.plan_id)}</small><h1>学生 ${esc(c.student_id)} · ${esc(c.move_name)}</h1></div><div class="navbuttons"><button id="prev" title="上一案例" aria-label="上一案例" ${i<=0?'disabled':''}><i data-lucide="chevron-left"></i></button><button id="next" title="下一案例" aria-label="下一案例" ${i<0||i>=state.filtered.length-1?'disabled':''}><i data-lucide="chevron-right"></i></button></div></div><p class="technique">${esc(c.technique)}</p><section class="images">${[['reference','参考锚点 · BV1WE411W7JB',c.reference_time],['student','学生 '+c.student_id,c.student_time]].map(([role,label,time])=>`<figure><figcaption>${esc(label)}<span>${fmt(time)} s</span></figcaption><button class="image-button" data-image="${esc(c.images[role])}" title="查看完整定势帧" aria-label="${esc(label)}大图"><img src="${esc(c.images[role])}" alt="${esc(label)}结束定势"></button></figure>`).join('')}</section><section class="columns">${panel('plan_feedback','新方法 · 计划驱动')}${panel('F','原 F 组')}</section>`;
  icons();
}
function edit(element) {
  const holder=element.closest('[data-record]');if(!holder)return;
  const id=holder.dataset.record,base=state.drafts[id]?.base||state.data.annotations[id];
  const value={...annotation(id),manual_verdict:holder.querySelector('input:checked').value,notes:holder.querySelector('textarea').value,reviewer:$('reviewer').value,updated_at:new Date().toISOString()};
  if(same(value,state.data.annotations[id])&&!state.inflight[id])delete state.drafts[id];
  else state.drafts[id]={base,value};
  persist();progress();queue();
}
async function request(url,options={}) {
  const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),15000);
  try {const response=await fetch(url,{...options,signal:controller.signal});const text=await response.text();let result;try{result=JSON.parse(text);}catch{throw Error('HTTP '+response.status);}if(!response.ok)throw Error(result.error||'HTTP '+response.status);return result;}
  finally {clearTimeout(timeout);}
}
async function save() {
  if(state.saving||state.conflicts.length||!Object.keys(state.drafts).length)return;
  const snapshot=JSON.parse(JSON.stringify(state.drafts));state.saving=true;state.inflight=snapshot;progress();
  try {
    const result=await request('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':state.data.token},body:JSON.stringify({revision:state.data.revision,updates:Object.entries(snapshot).map(([record_id,d])=>({record_id,...d.value}))})});
    state.data.annotations=result.annotations;state.data.revision=result.revision;
    for(const [id,d] of Object.entries(state.drafts)) {if(snapshot[id]&&same(d.value,snapshot[id].value))delete state.drafts[id];else d.base=result.annotations[id];}
    persist();message('标注已保存。');
  } catch(e) {message(`保存失败：${e.message}。未保存草稿仍保留；若版本冲突，请刷新后核对。`,true);}
  finally {state.saving=false;state.inflight={};progress();}
}
function choose(id) { save();queue();state.current=state.filtered.find(c=>c.case_id===id)||state.filtered[0];render();window.scrollTo(0,0); }
function exportCsv() {
  const cell=value=>{let text=String(value??'');if(/^[\s]*[=+@-]/.test(text))text="'"+text;return '"'+text.replace(/"/g,'""')+'"';};
  const fields=['record_id','case_id','student_id','move_id','move_name','source','review_scope','item_id','content','manual_verdict','notes','reviewer','updated_at','save_status'];
  const rows=[fields.map(cell).join(',')];
  for(const [id,record] of Object.entries(state.data.records)) {
    const row={...record,...annotation(id),content:typeof record.content==='string'?record.content:JSON.stringify(record.content),save_status:state.drafts[id]?'local_draft':'saved'};
    rows.push(fields.map(k=>cell(row[k])).join(','));
  }
  const url=URL.createObjectURL(new Blob(['\ufeff'+rows.join('\r\n')+'\r\n'],{type:'text/csv;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download=state.data.export_filename;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  message(Object.keys(state.drafts).length?'CSV 已导出，包含未保存草稿（local_draft）。':'CSV 已导出。');
}
function resolveConflict(keep) {
  if(!confirm(keep?'以本地草稿覆盖冲突项？下一次保存时生效。':'放弃冲突项本地草稿，采用服务器标注？'))return;
  for(const id of state.conflicts){if(keep)state.drafts[id].base=state.data.annotations[id];else delete state.drafts[id];}
  state.conflicts=[];$('conflict').hidden=true;persist();render();
}
async function load() {
  state.data=await request('/api/data');state.key=`plan-feedback-review:${state.data.dataset_id}:${state.data.state_id}`;
  try {
    const stored=JSON.parse(localStorage.getItem(state.key)||'{}');
    for(const [id,d] of Object.entries(stored)) {
      if(!state.data.records[id]||!d?.value||!d?.base||!(d.value.manual_verdict in verdicts))continue;
      const saved=state.data.annotations[id];if(same(d.value,saved))continue;
      state.drafts[id]=d;if(!same(d.base,saved))state.conflicts.push(id);
    }
  } catch {message('读取本地草稿失败；服务器标注不受影响。',true);}
  if(state.conflicts.length)$('conflict').hidden=false;
  else if(Object.keys(state.drafts).length)message(`已恢复 ${Object.keys(state.drafts).length} 项本地草稿。`);
  $('modelNote').textContent=`新方法：${state.data.models.plan_feedback}；原 F 组：${state.data.models.F}。模型标识与输入不同，本轮为开发性对照。`;
  for(const sid of [...new Set(state.data.cases.map(c=>c.student_id))])$('student').add(new Option(sid,sid));
  for(const [mid,name] of new Map(state.data.cases.map(c=>[c.move_id,c.move_name])))$('move').add(new Option(name,mid));
  state.current=state.data.cases[0];render();
}
$('content').addEventListener('input',e=>edit(e.target));
$('content').addEventListener('click',e=>{
  const image=e.target.closest('[data-image]');if(image){$('largeImage').src=image.dataset.image;$('imageDialog').showModal();return;}
  const b=e.target.closest('button');if(b?.id==='prev'||b?.id==='next'){const i=state.filtered.indexOf(state.current);choose(state.filtered[i+(b.id==='next'?1:-1)]?.case_id);}
});
$('queue').onclick=e=>{const b=e.target.closest('[data-case]');if(b)choose(b.dataset.case);};
for(const id of ['student','move','status'])$(id).onchange=()=>choose(state.current?.case_id);
$('save').onclick=save;$('export').onclick=exportCsv;$('closeImage').onclick=()=>$('imageDialog').close();
$('useServer').onclick=()=>resolveConflict(false);$('keepDraft').onclick=()=>resolveConflict(true);
window.addEventListener('beforeunload',e=>{if(Object.keys(state.drafts).length){e.preventDefault();e.returnValue='';}});
load().catch(e=>{message('加载失败：'+e.message,true);$('content').innerHTML='';});
