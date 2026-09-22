"use strict";

const SIGNAL_OPTIONS = [
  ["value", "数值"],
  ["velocity", "速度"],
  ["acceleration", "加速度"],
];
const AGGREGATION_OPTIONS = [
  ["instant", "瞬时值"],
  ["mean", "均值"],
  ["median", "中位数"],
  ["min", "最小值"],
  ["max", "最大值"],
  ["range", "变化范围"],
  ["std", "标准差"],
  ["slope", "趋势斜率"],
];
const OPERATOR_OPTIONS = [
  ["within_teacher_range", "教师合理范围内"],
  ["below_teacher_range", "低于教师范围"],
  ["above_teacher_range", "高于教师范围"],
  ["increasing", "持续增加"],
  ["decreasing", "持续减少"],
  ["stable", "保持稳定"],
  ["local_minimum", "局部最小"],
  ["local_maximum", "局部最大"],
];
const ROLE_OPTIONS = [
  ["required", "必要"],
  ["supporting", "辅助"],
];

const dom = {};
let project = null;
let state = null;
let currentTask = null;
let currentAnnotation = null;
let currentFrame = 0;
let dirty = false;
let toastTimer = null;

function byId(id) {
  return document.getElementById(id);
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function frameName(frame) {
  return `${String(frame).padStart(5, "0")}.jpg`;
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function parseNullableInteger(value) {
  if (value === "" || value === null || value === undefined) {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
}

function parseNullableNumber(value) {
  if (value === "" || value === null || value === undefined) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const remaining = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remaining.toFixed(1).padStart(4, "0")}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload;
  try {
    payload = await response.json();
  } catch (_error) {
    payload = { error: `HTTP ${response.status}` };
  }
  if (!response.ok) {
    const error = new Error(payload.error || `HTTP ${response.status}`);
    error.payload = payload;
    error.status = response.status;
    throw error;
  }
  return payload;
}

function showToast(message, isError = false) {
  clearTimeout(toastTimer);
  dom.toast.textContent = message;
  dom.toast.classList.toggle("error", isError);
  dom.toast.classList.add("show");
  toastTimer = setTimeout(() => dom.toast.classList.remove("show"), 2600);
}

function setDirty(value = true) {
  dirty = value;
  dom.saveState.textContent = value ? "有未保存更改" : "已保存";
  dom.saveState.classList.toggle("dirty", value);
}

function cacheDom() {
  const ids = [
    "app", "loading", "video-id", "progress-label", "progress-fill",
    "move-filter", "status-filter", "task-list", "move-name", "task-title",
    "previous-task", "next-task", "save-button", "save-state", "technique-text",
    "existing-metrics", "frame-view", "video-view", "jump-start", "previous-frame",
    "next-frame", "jump-end", "current-time", "current-frame", "frame-slider",
    "move-start-time", "selection-time", "move-end-time", "filmstrip", "set-start",
    "set-representative", "set-end", "clear-range", "review-status", "observability",
    "temporal-scope", "observability-reason", "gold-start", "gold-end",
    "gold-representative", "annotator-notes", "unsupported-block", "unsupported-list",
    "llm-status", "open-prompt", "generate-query", "apply-candidate", "candidate-banner",
    "query-progress-start", "query-progress-end", "selection-target", "selection-strategy",
    "representative-strategy", "add-condition", "conditions-body", "query-rationale",
    "query-json", "refresh-json", "apply-json", "prompt-dialog", "system-prompt",
    "user-prompt", "copy-prompts", "candidate-import", "import-candidate", "toast",
  ];
  for (const id of ids) {
    dom[id.replace(/-([a-z])/g, (_match, char) => char.toUpperCase())] = byId(id);
  }
}

async function initialize() {
  cacheDom();
  try {
    [project, state] = await Promise.all([
      api("/api/project"),
      api("/api/annotations"),
    ]);
    dom.videoId.textContent = `${project.video_id} · ${project.sample_fps.toFixed(0)} FPS`;
    dom.llmStatus.textContent = project.llm.enabled
      ? `自动生成：${project.llm.model}`
      : "手动导入模式";
    populateFilters();
    bindEvents();
    renderTaskList();

    const requestedTask = decodeURIComponent(window.location.hash.replace(/^#/, ""));
    const initial = project.tasks.find((item) => item.technique_id === requestedTask)
      || project.tasks[0];
    await selectTask(initial.technique_id, false);
    dom.loading.hidden = true;
    dom.app.hidden = false;
  } catch (error) {
    dom.loading.textContent = `载入失败：${error.message}`;
    dom.loading.style.color = "#aa3434";
  }
}

function populateFilters() {
  const moves = [];
  const seen = new Set();
  for (const task of project.tasks) {
    if (!seen.has(task.move_id)) {
      seen.add(task.move_id);
      moves.push(task);
    }
  }
  dom.moveFilter.replaceChildren();
  dom.moveFilter.appendChild(option("all", "全部招式"));
  for (const move of moves) {
    dom.moveFilter.appendChild(
      option(String(move.move_id), `${move.move_id}. ${move.move_display_name}`)
    );
  }
}

function bindEvents() {
  dom.moveFilter.addEventListener("change", renderTaskList);
  dom.statusFilter.addEventListener("change", renderTaskList);
  dom.saveButton.addEventListener("click", () => saveCurrent(false));
  dom.previousTask.addEventListener("click", () => navigateTask(-1));
  dom.nextTask.addEventListener("click", () => navigateTask(1));

  dom.frameSlider.addEventListener("input", () => setCurrentFrame(Number(dom.frameSlider.value)));
  dom.previousFrame.addEventListener("click", () => setCurrentFrame(currentFrame - 1));
  dom.nextFrame.addEventListener("click", () => setCurrentFrame(currentFrame + 1));
  dom.jumpStart.addEventListener("click", () => setCurrentFrame(currentTask.move_start_frame_5fps));
  dom.jumpEnd.addEventListener("click", () => setCurrentFrame(currentTask.move_end_frame_5fps));
  dom.setStart.addEventListener("click", () => markFrame("start_frame_5fps"));
  dom.setEnd.addEventListener("click", () => markFrame("end_frame_5fps"));
  dom.setRepresentative.addEventListener("click", () => markFrame("representative_frame_5fps"));
  dom.clearRange.addEventListener("click", clearGoldRange);

  for (const tab of document.querySelectorAll(".view-tab")) {
    tab.addEventListener("click", () => switchMediaView(tab.dataset.view));
  }
  dom.videoView.addEventListener("timeupdate", () => {
    if (!currentTask || dom.videoView.paused) return;
    const frame = Math.round(dom.videoView.currentTime * project.sample_fps);
    if (frame >= currentTask.move_start_frame_5fps && frame <= currentTask.move_end_frame_5fps) {
      setCurrentFrame(frame, false);
    }
  });

  const simpleFields = [
    dom.reviewStatus, dom.observability, dom.observabilityReason,
    dom.goldStart, dom.goldEnd, dom.goldRepresentative, dom.annotatorNotes,
  ];
  for (const field of simpleFields) {
    field.addEventListener("input", updateAnnotationFromForm);
    field.addEventListener("change", updateAnnotationFromForm);
  }
  dom.temporalScope.addEventListener("change", updateScopeFromForm);

  const queryFields = [
    dom.queryProgressStart, dom.queryProgressEnd, dom.selectionTarget,
    dom.selectionStrategy, dom.representativeStrategy, dom.queryRationale,
  ];
  for (const field of queryFields) {
    field.addEventListener("input", updateQueryFromForm);
    field.addEventListener("change", updateQueryFromForm);
  }

  dom.addCondition.addEventListener("click", addCondition);
  dom.refreshJson.addEventListener("click", refreshQueryJson);
  dom.applyJson.addEventListener("click", applyQueryJson);
  dom.openPrompt.addEventListener("click", openPromptDialog);
  dom.generateQuery.addEventListener("click", generateCandidate);
  dom.applyCandidate.addEventListener("click", applyCandidate);
  dom.copyPrompts.addEventListener("click", copyPrompts);
  dom.importCandidate.addEventListener("click", importCandidate);

  document.addEventListener("keydown", (event) => {
    const target = event.target;
    const isEditing = target instanceof HTMLInputElement
      || target instanceof HTMLTextAreaElement
      || target instanceof HTMLSelectElement;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveCurrent(false);
      return;
    }
    if (!isEditing && event.key === "ArrowLeft") {
      event.preventDefault();
      setCurrentFrame(currentFrame - 1);
    } else if (!isEditing && event.key === "ArrowRight") {
      event.preventDefault();
      setCurrentFrame(currentFrame + 1);
    }
  });

  window.addEventListener("beforeunload", (event) => {
    if (!dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
}

function option(value, label, selectedValue = null) {
  const element = document.createElement("option");
  element.value = value;
  element.textContent = label;
  element.selected = value === selectedValue;
  return element;
}

function filteredTasks() {
  const move = dom.moveFilter.value;
  const status = dom.statusFilter.value;
  return project.tasks.filter((task) => {
    const annotation = state.annotations[task.technique_id];
    return (move === "all" || String(task.move_id) === move)
      && (status === "all" || annotation.review_status === status);
  });
}

function renderTaskList() {
  dom.taskList.replaceChildren();
  const tasks = filteredTasks();
  let previousMove = null;
  for (const task of tasks) {
    if (task.move_id !== previousMove) {
      const heading = document.createElement("div");
      heading.className = "task-group-heading";
      heading.textContent = `${task.move_id}. ${task.move_display_name}`;
      heading.style.cssText = "padding:10px 9px 5px;color:#789087;font-size:10px;font-weight:700;";
      dom.taskList.appendChild(heading);
      previousMove = task.move_id;
    }
    const annotation = state.annotations[task.technique_id];
    const button = document.createElement("button");
    button.type = "button";
    button.className = `task-item${currentTask && currentTask.technique_id === task.technique_id ? " active" : ""}`;
    button.addEventListener("click", () => selectTask(task.technique_id));

    const statusBar = document.createElement("span");
    statusBar.className = `task-status-bar ${annotation.review_status}`;
    const copyBlock = document.createElement("span");
    copyBlock.className = "task-item-copy";
    const title = document.createElement("strong");
    title.textContent = task.stage_name;
    const subtitle = document.createElement("span");
    subtitle.textContent = observabilityLabel(annotation.observability);
    copyBlock.append(title, subtitle);
    const id = document.createElement("span");
    id.className = "task-id";
    id.textContent = task.technique_id;
    button.append(statusBar, copyBlock, id);
    dom.taskList.appendChild(button);
  }
  if (!tasks.length) {
    const empty = document.createElement("p");
    empty.textContent = "当前筛选条件下没有条目";
    empty.style.cssText = "padding:18px 10px;color:#91a69c;font-size:12px;";
    dom.taskList.appendChild(empty);
  }
  renderProgress();
}

function renderProgress() {
  const total = project.tasks.length;
  const reviewed = project.tasks.filter(
    (task) => state.annotations[task.technique_id].review_status === "reviewed"
  ).length;
  dom.progressLabel.textContent = `${reviewed} / ${total}`;
  dom.progressFill.style.width = `${total ? (100 * reviewed) / total : 0}%`;
}

function observabilityLabel(value) {
  return {
    observable: "可观测",
    partially_observable: "部分可观测",
    not_observable: "不可观测",
    uncertain: "待判断",
  }[value] || value;
}

async function selectTask(techniqueId, saveBeforeSwitch = true) {
  if (saveBeforeSwitch && dirty) {
    const saved = await saveCurrent(true);
    if (!saved) return;
  }
  const task = project.tasks.find((item) => item.technique_id === techniqueId);
  if (!task) return;
  currentTask = task;
  currentAnnotation = clone(state.annotations[techniqueId]);
  currentFrame = currentAnnotation.gold.representative_frame_5fps
    ?? currentAnnotation.gold.start_frame_5fps
    ?? task.move_start_frame_5fps;
  window.location.hash = encodeURIComponent(techniqueId);
  renderCurrentTask();
  renderTaskList();
  setDirty(false);
}

function renderCurrentTask() {
  dom.moveName.textContent = `第 ${currentTask.move_id} 式 · ${currentTask.move_display_name}`;
  dom.taskTitle.textContent = `${currentTask.technique_id} ${currentTask.stage_name}`;
  dom.techniqueText.textContent = currentTask.technique;
  renderExistingMetrics();
  renderUnsupported();
  renderAnnotationForm();
  configureTimeline();
  setCurrentFrame(currentFrame);
  renderCandidate();
  updateNavigationButtons();
}

function renderExistingMetrics() {
  dom.existingMetrics.replaceChildren();
  for (const check of currentTask.existing_checks || []) {
    const chip = document.createElement("span");
    chip.className = "metric-chip";
    chip.textContent = check.metric_id;
    chip.title = check.check_id || check.metric_id;
    dom.existingMetrics.appendChild(chip);
  }
}

function renderUnsupported() {
  const observations = currentTask.unsupported_observations || [];
  dom.unsupportedBlock.hidden = observations.length === 0;
  dom.unsupportedList.replaceChildren();
  for (const observation of observations) {
    const item = document.createElement("li");
    item.textContent = `${observation.requirement}：${observation.reason}`;
    dom.unsupportedList.appendChild(item);
  }
}

function renderAnnotationForm() {
  dom.reviewStatus.value = currentAnnotation.review_status;
  dom.observability.value = currentAnnotation.observability;
  dom.temporalScope.value = currentAnnotation.temporal_scope;
  dom.observabilityReason.value = currentAnnotation.observability_reason || "";
  dom.goldStart.value = currentAnnotation.gold.start_frame_5fps ?? "";
  dom.goldEnd.value = currentAnnotation.gold.end_frame_5fps ?? "";
  dom.goldRepresentative.value = currentAnnotation.gold.representative_frame_5fps ?? "";
  dom.annotatorNotes.value = currentAnnotation.annotator_notes || "";
  renderQueryForm();
}

function configureTimeline() {
  dom.frameSlider.min = currentTask.move_start_frame_5fps;
  dom.frameSlider.max = currentTask.move_end_frame_5fps;
  dom.goldStart.min = currentTask.move_start_frame_5fps;
  dom.goldStart.max = currentTask.move_end_frame_5fps;
  dom.goldEnd.min = currentTask.move_start_frame_5fps;
  dom.goldEnd.max = currentTask.move_end_frame_5fps;
  dom.goldRepresentative.min = currentTask.move_start_frame_5fps;
  dom.goldRepresentative.max = currentTask.move_end_frame_5fps;
  dom.moveStartTime.textContent = formatTime(currentTask.move_start_seconds);
  dom.moveEndTime.textContent = formatTime(currentTask.move_end_seconds);
}

function setCurrentFrame(frame, syncVideo = true) {
  if (!currentTask) return;
  currentFrame = clamp(
    Math.round(frame),
    currentTask.move_start_frame_5fps,
    currentTask.move_end_frame_5fps
  );
  dom.frameSlider.value = currentFrame;
  dom.frameView.src = `/media/frames/${frameName(currentFrame)}`;
  dom.currentTime.textContent = formatTime(currentFrame / project.sample_fps);
  dom.currentFrame.textContent = `5 FPS 帧 ${currentFrame}`;
  if (syncVideo && dom.videoView.paused) {
    dom.videoView.currentTime = currentFrame / project.sample_fps;
  }
  renderFilmstrip();
  updateTimelineSelection();
}

function renderFilmstrip() {
  const startBound = currentTask.move_start_frame_5fps;
  const endBound = currentTask.move_end_frame_5fps;
  const visibleCount = Math.min(7, endBound - startBound + 1);
  let start = clamp(currentFrame - 3, startBound, Math.max(startBound, endBound - visibleCount + 1));
  dom.filmstrip.replaceChildren();
  for (let frame = start; frame < start + visibleCount; frame += 1) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "filmstrip-button";
    if (frame === currentFrame) button.classList.add("active");
    const goldStart = currentAnnotation.gold.start_frame_5fps;
    const goldEnd = currentAnnotation.gold.end_frame_5fps;
    if (goldStart !== null && goldEnd !== null && frame >= goldStart && frame <= goldEnd) {
      button.classList.add("in-range");
    }
    button.addEventListener("click", () => setCurrentFrame(frame));
    const image = document.createElement("img");
    image.src = `/media/frames/${frameName(frame)}`;
    image.alt = `帧 ${frame}`;
    image.loading = "lazy";
    const label = document.createElement("span");
    label.textContent = frame;
    button.append(image, label);
    dom.filmstrip.appendChild(button);
  }
}

function updateTimelineSelection() {
  const start = currentAnnotation.gold.start_frame_5fps;
  const end = currentAnnotation.gold.end_frame_5fps;
  if (start === null && end === null) {
    dom.selectionTime.textContent = "尚未标注区间";
  } else {
    const startText = start === null ? "?" : formatTime(start / project.sample_fps);
    const endText = end === null ? "?" : formatTime(end / project.sample_fps);
    dom.selectionTime.textContent = `${startText} – ${endText}`;
  }
}

function markFrame(key) {
  currentAnnotation.gold[key] = currentFrame;
  if (key === "start_frame_5fps") {
    const end = currentAnnotation.gold.end_frame_5fps;
    if (end !== null && currentFrame > end) currentAnnotation.gold.end_frame_5fps = currentFrame;
  } else if (key === "end_frame_5fps") {
    const start = currentAnnotation.gold.start_frame_5fps;
    if (start !== null && currentFrame < start) currentAnnotation.gold.start_frame_5fps = currentFrame;
  }
  if (currentAnnotation.review_status === "unannotated") {
    currentAnnotation.review_status = "draft";
  }
  renderAnnotationForm();
  renderFilmstrip();
  updateTimelineSelection();
  setDirty();
}

function clearGoldRange() {
  currentAnnotation.gold = {
    start_frame_5fps: null,
    end_frame_5fps: null,
    representative_frame_5fps: null,
  };
  renderAnnotationForm();
  renderFilmstrip();
  updateTimelineSelection();
  setDirty();
}

function switchMediaView(view) {
  const frameMode = view === "frame";
  dom.frameView.hidden = !frameMode;
  dom.videoView.hidden = frameMode;
  for (const tab of document.querySelectorAll(".view-tab")) {
    tab.classList.toggle("active", tab.dataset.view === view);
  }
  if (!frameMode) dom.videoView.currentTime = currentFrame / project.sample_fps;
}

function updateAnnotationFromForm() {
  currentAnnotation.review_status = dom.reviewStatus.value;
  currentAnnotation.observability = dom.observability.value;
  currentAnnotation.observability_reason = dom.observabilityReason.value;
  currentAnnotation.gold.start_frame_5fps = parseNullableInteger(dom.goldStart.value);
  currentAnnotation.gold.end_frame_5fps = parseNullableInteger(dom.goldEnd.value);
  currentAnnotation.gold.representative_frame_5fps = parseNullableInteger(dom.goldRepresentative.value);
  currentAnnotation.annotator_notes = dom.annotatorNotes.value;
  renderFilmstrip();
  updateTimelineSelection();
  setDirty();
}

function updateScopeFromForm() {
  currentAnnotation.temporal_scope = dom.temporalScope.value;
  currentAnnotation.query.scope = dom.temporalScope.value;
  if (dom.temporalScope.value === "instant_event") {
    currentAnnotation.query.selection.target = "frame";
    currentAnnotation.query.selection.representative_frame = "center";
  } else if (dom.temporalScope.value === "whole_move") {
    currentAnnotation.query.selection.target = "whole_move";
    currentAnnotation.query.selection.representative_frame = "none";
  } else {
    currentAnnotation.query.selection.target = "window";
    if (currentAnnotation.query.selection.representative_frame === "none") {
      currentAnnotation.query.selection.representative_frame = "center";
    }
  }
  renderQueryForm();
  setDirty();
}

function renderQueryForm() {
  const query = currentAnnotation.query;
  dom.queryProgressStart.value = query.search_region.start;
  dom.queryProgressEnd.value = query.search_region.end;
  dom.selectionTarget.value = query.selection.target;
  dom.selectionStrategy.value = query.selection.strategy;
  dom.representativeStrategy.value = query.selection.representative_frame;
  dom.queryRationale.value = query.rationale || "";
  renderConditions();
  refreshQueryJson();
}

function updateQueryFromForm() {
  const query = currentAnnotation.query;
  query.search_region.start = Number(dom.queryProgressStart.value);
  query.search_region.end = Number(dom.queryProgressEnd.value);
  query.selection.target = dom.selectionTarget.value;
  query.selection.strategy = dom.selectionStrategy.value;
  query.selection.representative_frame = dom.representativeStrategy.value;
  query.rationale = dom.queryRationale.value;
  refreshQueryJson();
  setDirty();
}

function createSelect(options, value, onChange) {
  const select = document.createElement("select");
  for (const [optionValue, label] of options) {
    select.appendChild(option(optionValue, label, value));
  }
  select.addEventListener("change", () => onChange(select.value));
  return select;
}

function metricOptions() {
  return Object.entries(project.metrics)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([metricId, definition]) => [
      metricId,
      `${metricId} · ${definition.unit || definition.type || "metric"}`,
    ]);
}

function renderConditions() {
  dom.conditionsBody.replaceChildren();
  const conditions = currentAnnotation.query.conditions;
  if (!conditions.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 7;
    cell.className = "empty-conditions";
    cell.textContent = "尚未添加查询条件";
    row.appendChild(cell);
    dom.conditionsBody.appendChild(row);
    return;
  }
  conditions.forEach((condition, index) => {
    const row = document.createElement("tr");
    const metricCell = document.createElement("td");
    metricCell.appendChild(createSelect(metricOptions(), condition.metric_id, (value) => {
      condition.metric_id = value;
      queryChanged();
    }));
    const signalCell = document.createElement("td");
    signalCell.appendChild(createSelect(SIGNAL_OPTIONS, condition.signal, (value) => {
      condition.signal = value;
      queryChanged();
    }));
    const aggregationCell = document.createElement("td");
    aggregationCell.appendChild(createSelect(AGGREGATION_OPTIONS, condition.aggregation, (value) => {
      condition.aggregation = value;
      queryChanged();
    }));
    const operatorCell = document.createElement("td");
    operatorCell.appendChild(createSelect(OPERATOR_OPTIONS, condition.operator, (value) => {
      condition.operator = value;
      queryChanged();
    }));
    const roleCell = document.createElement("td");
    roleCell.appendChild(createSelect(ROLE_OPTIONS, condition.role, (value) => {
      condition.role = value;
      queryChanged();
    }));
    const durationCell = document.createElement("td");
    const duration = document.createElement("input");
    duration.type = "number";
    duration.min = "0";
    duration.step = "0.1";
    duration.placeholder = "可空";
    duration.value = condition.min_duration_seconds ?? "";
    duration.addEventListener("input", () => {
      condition.min_duration_seconds = parseNullableNumber(duration.value);
      queryChanged();
    });
    durationCell.appendChild(duration);
    const deleteCell = document.createElement("td");
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "delete-condition";
    deleteButton.textContent = "删除";
    deleteButton.title = "删除条件";
    deleteButton.addEventListener("click", () => {
      currentAnnotation.query.conditions.splice(index, 1);
      currentAnnotation.query.temporal_relations = currentAnnotation.query.temporal_relations
        .filter((relation) => relation.first_condition !== index && relation.second_condition !== index)
        .map((relation) => ({
          ...relation,
          first_condition: relation.first_condition > index ? relation.first_condition - 1 : relation.first_condition,
          second_condition: relation.second_condition > index ? relation.second_condition - 1 : relation.second_condition,
        }));
      renderConditions();
      queryChanged();
    });
    deleteCell.appendChild(deleteButton);
    row.append(metricCell, signalCell, aggregationCell, operatorCell, roleCell, durationCell, deleteCell);
    dom.conditionsBody.appendChild(row);
  });
}

function preferredMetric() {
  const check = (currentTask.existing_checks || []).find(
    (item) => item.metric_id && project.metrics[item.metric_id]
  );
  return check ? check.metric_id : Object.keys(project.metrics).sort()[0];
}

function addCondition() {
  currentAnnotation.query.conditions.push({
    metric_id: preferredMetric(),
    signal: "value",
    aggregation: currentAnnotation.temporal_scope === "instant_event" ? "instant" : "mean",
    operator: "within_teacher_range",
    role: "required",
    min_duration_seconds: currentAnnotation.temporal_scope === "stable_window" ? 0.4 : null,
  });
  renderConditions();
  queryChanged();
}

function queryChanged() {
  refreshQueryJson();
  setDirty();
}

function refreshQueryJson() {
  dom.queryJson.value = JSON.stringify(currentAnnotation.query, null, 2);
}

function applyQueryJson() {
  try {
    const query = JSON.parse(dom.queryJson.value);
    if (!query || typeof query !== "object" || Array.isArray(query)) {
      throw new Error("查询必须是 JSON 对象");
    }
    currentAnnotation.query = query;
    currentAnnotation.temporal_scope = query.scope;
    dom.temporalScope.value = query.scope;
    renderQueryForm();
    setDirty();
    showToast("已应用 JSON 查询");
  } catch (error) {
    showToast(`JSON 无法应用：${error.message}`, true);
  }
}

function renderCandidate() {
  const candidate = currentAnnotation.llm_candidate;
  dom.applyCandidate.hidden = !candidate;
  dom.candidateBanner.hidden = !candidate;
  if (!candidate) return;
  const conditionCount = candidate.query && Array.isArray(candidate.query.conditions)
    ? candidate.query.conditions.length
    : 0;
  dom.candidateBanner.textContent = [
    `候选判断：${observabilityLabel(candidate.observability)}`,
    `时序类型：${candidate.temporal_scope || "未提供"}`,
    `条件数：${conditionCount}`,
    candidate.observability_reason || "",
  ].filter(Boolean).join(" · ");
}

async function loadPrompt() {
  const payload = await api(`/api/prompt/${encodeURIComponent(currentTask.technique_id)}`);
  dom.systemPrompt.value = payload.system_prompt;
  dom.userPrompt.value = payload.user_prompt;
}

async function openPromptDialog() {
  try {
    await loadPrompt();
    dom.candidateImport.value = currentAnnotation.llm_candidate
      ? JSON.stringify(currentAnnotation.llm_candidate, null, 2)
      : "";
    dom.promptDialog.showModal();
  } catch (error) {
    showToast(`提示词载入失败：${error.message}`, true);
  }
}

async function generateCandidate() {
  dom.generateQuery.disabled = true;
  dom.generateQuery.textContent = "正在生成...";
  try {
    const payload = await api(`/api/generate/${encodeURIComponent(currentTask.technique_id)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    currentAnnotation = clone(payload.annotation);
    state.annotations[currentTask.technique_id] = clone(payload.annotation);
    renderAnnotationForm();
    renderCandidate();
    renderTaskList();
    setDirty(false);
    showToast("候选查询已生成，请审核后采用");
  } catch (error) {
    if (error.payload && error.payload.manual_mode) {
      await openPromptDialog();
      showToast("当前为手动导入模式");
    } else {
      showToast(`候选生成失败：${error.message}`, true);
    }
  } finally {
    dom.generateQuery.disabled = false;
    dom.generateQuery.textContent = "生成候选查询";
  }
}

function normalizeCandidate(value) {
  if (value && value.schema_version === "0.1" && value.scope) {
    return {
      observability: dom.observability.value,
      observability_reason: dom.observabilityReason.value,
      temporal_scope: value.scope,
      query: value,
    };
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("候选结果必须是 JSON 对象");
  }
  if (!value.observability || !value.temporal_scope || !("query" in value)) {
    throw new Error("候选结果缺少 observability、temporal_scope 或 query");
  }
  return value;
}

function importCandidate() {
  try {
    const candidate = normalizeCandidate(JSON.parse(dom.candidateImport.value));
    currentAnnotation.llm_candidate = candidate;
    if (currentAnnotation.review_status === "unannotated") {
      currentAnnotation.review_status = "draft";
    }
    renderAnnotationForm();
    renderCandidate();
    setDirty();
    dom.promptDialog.close();
    showToast("候选查询已导入，请审核后采用");
  } catch (error) {
    showToast(`候选导入失败：${error.message}`, true);
  }
}

function applyCandidate() {
  const candidate = currentAnnotation.llm_candidate;
  if (!candidate) return;
  currentAnnotation.observability = candidate.observability;
  currentAnnotation.observability_reason = candidate.observability_reason || "";
  currentAnnotation.temporal_scope = candidate.temporal_scope;
  if (candidate.query) currentAnnotation.query = clone(candidate.query);
  if (currentAnnotation.review_status === "unannotated") {
    currentAnnotation.review_status = "draft";
  }
  renderAnnotationForm();
  renderCandidate();
  setDirty();
  showToast("候选已写入人工查询，可继续修改");
}

async function copyPrompts() {
  const combined = `SYSTEM\n${dom.systemPrompt.value}\n\nUSER\n${dom.userPrompt.value}`;
  try {
    await navigator.clipboard.writeText(combined);
    showToast("提示词已复制");
  } catch (_error) {
    dom.userPrompt.focus();
    dom.userPrompt.select();
    showToast("浏览器未开放剪贴板，请手动复制", true);
  }
}

function validateGoldLocally() {
  if (currentAnnotation.observability === "not_observable") return;
  const { start_frame_5fps: start, end_frame_5fps: end } = currentAnnotation.gold;
  if (currentAnnotation.review_status === "reviewed" && (start === null || end === null)) {
    throw new Error("标记为已审核前，需要填写起始帧和结束帧");
  }
  if (start !== null && end !== null && start > end) {
    throw new Error("起始帧不能晚于结束帧");
  }
}

async function saveCurrent(quiet = false) {
  if (!currentTask || !currentAnnotation) return true;
  try {
    validateGoldLocally();
    const saved = await api(`/api/annotations/${encodeURIComponent(currentTask.technique_id)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentAnnotation),
    });
    currentAnnotation = clone(saved);
    state.annotations[currentTask.technique_id] = clone(saved);
    setDirty(false);
    renderTaskList();
    renderCandidate();
    if (!quiet) showToast("标注已保存");
    return true;
  } catch (error) {
    showToast(`保存失败：${error.message}`, true);
    return false;
  }
}

async function navigateTask(direction) {
  const index = project.tasks.findIndex(
    (task) => task.technique_id === currentTask.technique_id
  );
  const nextIndex = clamp(index + direction, 0, project.tasks.length - 1);
  if (nextIndex !== index) await selectTask(project.tasks[nextIndex].technique_id);
}

function updateNavigationButtons() {
  const index = project.tasks.findIndex(
    (task) => task.technique_id === currentTask.technique_id
  );
  dom.previousTask.disabled = index <= 0;
  dom.nextTask.disabled = index >= project.tasks.length - 1;
}

initialize();
