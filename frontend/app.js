const state = {
  options: null,
  response: null,
  activeTab: "video",
};

const els = {};

document.addEventListener("DOMContentLoaded", () => {
  [
    "analysisForm",
    "runAnalysis",
    "modelType",
    "dialoguePath",
    "criterionPath",
    "evaluatedSpeaker",
    "blockSize",
    "apiKey",
    "skipPredict",
    "customDialogue",
    "clearCustom",
    "statusLine",
    "sessionTable",
    "sessionSubtitle",
    "participantName",
    "scoreValue",
    "skillsCount",
    "summaryCount",
    "summaryList",
    "transcriptStrip",
    "competencyList",
    "assessmentSubtitle",
    "scoreRing",
    "totalScore",
    "rawResult",
    "scenarioText",
  ].forEach((id) => {
    els[id] = document.getElementById(id);
  });

  bindEvents();
  loadOptions();
});

function bindEvents() {
  els.analysisForm.addEventListener("submit", (event) => {
    event.preventDefault();
    analyze();
  });
  els.runAnalysis.addEventListener("click", analyze);
  els.clearCustom.addEventListener("click", () => {
    els.customDialogue.value = "";
  });
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => setTab(button.dataset.tab));
  });
}

async function loadOptions() {
  setStatus("Загружаю локальные данные...");
  try {
    const response = await fetch("/api/options");
    if (!response.ok) throw new Error(await response.text());
    state.options = await response.json();
    populateControls();
    renderSessions();
    setStatus("Готов к анализу.");
    if (state.options.dialogues.length && state.options.criteria.length) {
      analyze();
    }
  } catch (error) {
    setStatus(`Не удалось загрузить опции: ${error.message}`, true);
  }
}

function populateControls() {
  const options = state.options;
  fillSelect(
    els.modelType,
    options.models.map((model) => ({ value: model, label: model })),
    options.defaults.model_type || "points",
  );
  fillSelect(
    els.dialoguePath,
    options.dialogues.map((dialogue) => ({
      value: dialogue.path,
      label: `${dialogue.file} · ${dialogue.turn_count || 0} реплик`,
    })),
  );
  fillSelect(
    els.criterionPath,
    options.criteria.map((criterion) => ({
      value: criterion.path,
      label: criterion.name || criterion.file,
    })),
  );
  els.blockSize.value = options.defaults.dialogue_block_size || 6;
  els.evaluatedSpeaker.value = options.defaults.evaluated_speaker || "B";
  els.skipPredict.checked = Boolean(options.defaults.skip_predict);
}

function fillSelect(select, items, selectedValue = "") {
  select.innerHTML = items
    .map((item) => `<option value="${escapeAttr(item.value)}">${escapeHtml(item.label)}</option>`)
    .join("");
  if (selectedValue) select.value = selectedValue;
}

function renderSessions() {
  const dialogues = state.options?.dialogues || [];
  if (!dialogues.length) {
    els.sessionTable.innerHTML = `<div class="empty-state">Стенограммы не найдены.</div>`;
    return;
  }

  els.sessionTable.innerHTML = dialogues
    .map(
      (dialogue, index) => `
        <div class="session-row">
          <span class="session-index">${index + 1}</span>
          <div>
            <strong class="session-title">${escapeHtml(dialogue.file)}</strong>
            <span class="session-meta">${dialogue.turn_count || 0} реплик · ${escapeHtml(dialogue.language || "язык не указан")}</span>
          </div>
          <button type="button" data-dialogue="${escapeAttr(dialogue.path)}">Открыть</button>
        </div>
      `,
    )
    .join("");

  els.sessionTable.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      els.dialoguePath.value = button.dataset.dialogue;
      analyze();
    });
  });
}

async function analyze() {
  if (!state.options) return;

  setStatus("Анализирую диалог...");
  setBusy(true);

  const customDialogue = els.customDialogue.value.trim();
  const payload = {
    model_type: els.modelType.value,
    criterion_path: els.criterionPath.value,
    dialogue_path: customDialogue ? null : els.dialoguePath.value,
    dialogue: customDialogue || null,
    evaluated_speaker: els.evaluatedSpeaker.value.trim() || "B",
    dialogue_block_size: Number(els.blockSize.value) || 6,
    skip_predict: els.skipPredict.checked,
    summary_skip_predict: els.skipPredict.checked,
    api_key: els.apiKey.value.trim() || null,
  };

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const message = await response.text();
      throw new Error(message);
    }

    state.response = await response.json();
    renderAnalysis();
    setStatus("Анализ завершен.");
  } catch (error) {
    setStatus(`Ошибка анализа: ${error.message}`, true);
  } finally {
    setBusy(false);
  }
}

function renderAnalysis() {
  const response = state.response;
  if (!response) return;

  const score = response.score || {};
  const session = response.session || {};
  const criterion = session.criterion || {};
  const dialogue = session.dialogue || {};

  els.sessionSubtitle.textContent = `${criterion.name || "Критерий"} · ${dialogue.file || "свой диалог"}`;
  els.participantName.textContent = session.evaluated_speaker || "B";
  els.scoreValue.textContent = score.score_100 ?? 0;
  els.skillsCount.textContent = `${score.evaluated || 0}/${score.total || 0}`;
  els.totalScore.textContent = score.score_100 ?? 0;
  els.rawResult.textContent = JSON.stringify(response.analysis || {}, null, 2);

  renderSummary(response.summary);
  renderTranscript(response.summary);
  renderCompetencies(score);
  renderScenario(session);
}

function renderSummary(summary) {
  const blocks = summary?.blocks || [];
  els.summaryCount.textContent = `${blocks.length} блоков`;

  if (!blocks.length) {
    els.summaryList.innerHTML = `<div class="empty-state">Суммаризация пока пуста.</div>`;
    return;
  }

  els.summaryList.innerHTML = blocks
    .map((block) => {
      const events = Array.isArray(block.events) ? block.events.slice(0, 3) : [];
      return `
        <article class="summary-item">
          <span class="item-kicker">Блок ${block.block_index}</span>
          <p>${escapeHtml(block.summary || "")}</p>
          <div class="events">
            ${events
              .map(
                (event) => `
                  <div class="event-line">
                    <strong>${escapeHtml(event.speaker || "")}</strong>
                    ${escapeHtml(event.text || "")}
                  </div>
                `,
              )
              .join("")}
          </div>
        </article>
      `;
    })
    .join("");
}

function renderTranscript(summary) {
  const blocks = summary?.blocks || [];
  if (!blocks.length) {
    els.transcriptStrip.textContent = "Стенограмма появится после анализа.";
    return;
  }
  els.transcriptStrip.textContent = blocks.map((block) => block.dialogue_block || "").join("\n\n");
}

function renderCompetencies(score) {
  const competencies = score?.competencies || [];
  const score10 = Number(score?.score_10 || 0);
  els.scoreRing.style.setProperty("--value", `${Math.max(0, Math.min(10, score10)) * 36}deg`);
  els.scoreRing.querySelector("span").textContent = score10.toFixed(1);
  els.assessmentSubtitle.textContent = `Оценено навыков: ${score?.evaluated || 0} из ${score?.total || 0}`;

  if (!competencies.length) {
    els.competencyList.innerHTML = `<div class="empty-state">Компетенции не найдены.</div>`;
    return;
  }

  els.competencyList.innerHTML = competencies
    .map((item) => {
      const label = item.label;
      const badgeClass = label === 1 ? "good" : label === -1 ? "bad" : label === 0 ? "neutral" : "";
      const badgeText = label === null || label === undefined ? "?" : label;
      const rationale = item.rationale || item.evidence || "Нет текстового результата для этого пункта.";
      return `
        <article class="competency-item">
          <div class="label-badge ${badgeClass}">${badgeText}</div>
          <div>
            <h3>${escapeHtml(item.point || "Критерий")}</h3>
            <p>${escapeHtml(rationale)}</p>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderScenario(session) {
  const criterion = session?.criterion || {};
  const dialogue = session?.dialogue || {};
  const points = criterion.points || [];

  els.scenarioText.innerHTML = `
    <p class="eyebrow">Сценарий оценки</p>
    <h2>${escapeHtml(criterion.name || "Критерий")}</h2>
    <p>${escapeHtml(dialogue.file || "Свой диалог")} · ${dialogue.turn_count || 0} реплик · модель ${escapeHtml(session.model_type || "")}</p>
    <ul class="scenario-points">
      ${points.map((point) => `<li>${escapeHtml(point)}</li>`).join("")}
    </ul>
  `;
}

function setTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tab}`);
  });
}

function setStatus(message, isError = false) {
  els.statusLine.textContent = message;
  els.statusLine.classList.toggle("error", isError);
}

function setBusy(isBusy) {
  els.runAnalysis.disabled = isBusy;
  els.analysisForm.querySelectorAll("button[type='submit']").forEach((button) => {
    button.disabled = isBusy;
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("\n", " ");
}
