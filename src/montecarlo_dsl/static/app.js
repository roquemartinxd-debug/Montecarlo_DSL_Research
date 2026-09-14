(function () {
  "use strict";

  var editor = document.getElementById("sourceEditor");
  var lineNumbers = document.getElementById("lineNumbers");
  var runButton = document.getElementById("runButton");
  var cancelButton = document.getElementById("cancelButton");
  var seedInput = document.getElementById("seedInput");
  var diagnostics = document.getElementById("diagnostics");
  var diagnosticList = document.getElementById("diagnosticList");
  var statusMessage = document.getElementById("statusMessage");
  var connectionState = document.getElementById("connectionState");
  var seedValue = document.getElementById("seedValue");
  var progressBar = document.getElementById("progressBar");
  var progressText = document.getElementById("progressText");
  var iterationsValue = document.getElementById("iterationsValue");
  var candidateValue = document.getElementById("candidateValue");
  var validValue = document.getElementById("validValue");
  var discardedValue = document.getElementById("discardedValue");
  var workersValue = document.getElementById("workersValue");
  var branchControls = document.getElementById("branchControls");
  var branchSelect = document.getElementById("branchSelect");
  var branchSummary = document.getElementById("branchSummary");
  var branchValid = document.getElementById("branchValid");
  var branchDiscarded = document.getElementById("branchDiscarded");
  var branchMcse = document.getElementById("branchMcse");
  var branchCi95 = document.getElementById("branchCi95");
  var histogramCount = document.getElementById("histogramCount");
  var emptyChart = document.getElementById("emptyChart");
  var runtimeMessage = document.getElementById("runtimeMessage");

  var state = {
    jobId: null,
    socket: null,
    terminal: true,
    totalIterations: 0,
    requestedStatistics: [],
    branchDescriptors: [],
    branchPayloads: [],
    reconcileTimer: null
  };

  function updateLineNumbers() {
    var count = editor.value.split("\n").length;
    var numbers = [];
    for (var i = 1; i <= count; i += 1) numbers.push(String(i));
    lineNumbers.textContent = numbers.join("\n");
    lineNumbers.scrollTop = editor.scrollTop;
  }

  editor.addEventListener("input", updateLineNumbers);
  editor.addEventListener("scroll", function () {
    lineNumbers.scrollTop = editor.scrollTop;
  });
  editor.addEventListener("keydown", function (event) {
    if (event.key !== "Tab") return;
    event.preventDefault();
    var start = editor.selectionStart;
    var end = editor.selectionEnd;
    editor.setRangeText("    ", start, end, "end");
    updateLineNumbers();
  });
  updateLineNumbers();

  function formatNumber(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    var number = Number(value);
    var abs = Math.abs(number);
    if (abs !== 0 && (abs >= 1000000 || abs < 0.0001)) return number.toExponential(4);
    return new Intl.NumberFormat("es-MX", { maximumFractionDigits: 6 }).format(number);
  }

  function formatInteger(value) {
    return new Intl.NumberFormat("es-MX", { maximumFractionDigits: 0 }).format(Number(value) || 0);
  }

  function setStatus(text, connectionText) {
    statusMessage.textContent = text;
    if (connectionText) connectionState.textContent = connectionText;
  }

  function setProgress(value) {
    var progress = Math.max(0, Math.min(1, Number(value) || 0));
    progressBar.value = progress;
    progressText.textContent = (progress * 100).toFixed(progress === 1 ? 0 : 1) + "%";
  }

  function clearDiagnostics() {
    diagnostics.hidden = true;
    diagnosticList.replaceChildren();
  }

  function showDiagnostics(items) {
    diagnosticList.replaceChildren();
    (items || []).forEach(function (item) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "diagnostic-item";
      var title = document.createElement("strong");
      title.textContent = item.code + " · " + item.message;
      var location = document.createElement("span");
      location.textContent = "Línea " + item.line + ", columna " + item.column;
      button.appendChild(title);
      button.appendChild(document.createElement("br"));
      button.appendChild(location);
      button.addEventListener("click", function () {
        var start = Math.max(0, Number(item.start_offset) || 0);
        var end = Math.max(start + 1, Number(item.end_offset) || start + 1);
        editor.focus();
        editor.setSelectionRange(start, Math.min(end, editor.value.length));
      });
      diagnosticList.appendChild(button);
    });
    diagnostics.hidden = !(items && items.length);
  }

  function setRequestedStatistics(names) {
    state.requestedStatistics = names || [];
    Array.prototype.forEach.call(document.querySelectorAll("[data-stat]"), function (card) {
      card.hidden = state.requestedStatistics.indexOf(card.getAttribute("data-stat")) === -1;
    });
  }

  function updateStats(stats) {
    stats = stats || {};
    if (Object.prototype.hasOwnProperty.call(stats, "avg")) document.getElementById("statAvg").textContent = formatNumber(stats.avg);
    if (Object.prototype.hasOwnProperty.call(stats, "min")) document.getElementById("statMin").textContent = formatNumber(stats.min);
    if (Object.prototype.hasOwnProperty.call(stats, "max")) document.getElementById("statMax").textContent = formatNumber(stats.max);
    if (Object.prototype.hasOwnProperty.call(stats, "var")) document.getElementById("statVar").textContent = formatNumber(stats.var);
    if (Object.prototype.hasOwnProperty.call(stats, "std")) document.getElementById("statStd").textContent = formatNumber(stats.std);
    if (Object.prototype.hasOwnProperty.call(stats, "count")) document.getElementById("statCount").textContent = formatInteger(stats.count);
    if (Object.prototype.hasOwnProperty.call(stats, "valid_rate")) document.getElementById("statValidRate").textContent = formatNumber(stats.valid_rate);
    if (Object.prototype.hasOwnProperty.call(stats, "discard_rate")) document.getElementById("statDiscardRate").textContent = formatNumber(stats.discard_rate);
    if (Object.prototype.hasOwnProperty.call(stats, "p05")) document.getElementById("statP05").textContent = formatNumber(stats.p05);
    if (Object.prototype.hasOwnProperty.call(stats, "p50")) document.getElementById("statP50").textContent = formatNumber(stats.p50);
    if (Object.prototype.hasOwnProperty.call(stats, "p95")) document.getElementById("statP95").textContent = formatNumber(stats.p95);
  }

  function resetResults() {
    setProgress(0);
    iterationsValue.textContent = "0 / 0";
    candidateValue.textContent = "0";
    validValue.textContent = "0";
    discardedValue.textContent = "0";
    workersValue.textContent = "—";
    seedValue.textContent = "—";
    histogramCount.textContent = "0 observaciones";
    setRequestedStatistics([]);
    ["statAvg", "statMin", "statMax", "statVar", "statStd", "statCount", "statValidRate", "statDiscardRate", "statP05", "statP50", "statP95"].forEach(function (id) {
      document.getElementById(id).textContent = "—";
    });
    runtimeMessage.hidden = true;
    runtimeMessage.textContent = "";
    state.branchDescriptors = [];
    state.branchPayloads = [];
    branchSelect.replaceChildren();
    branchControls.hidden = true;
    branchSummary.hidden = true;
    branchValid.textContent = "0";
    branchDiscarded.textContent = "0";
    branchMcse.textContent = "—";
    branchCi95.textContent = "—";
    clearHistogram();
  }

  function configureBranches(descriptors) {
    state.branchDescriptors = descriptors || [];
    branchSelect.replaceChildren();
    state.branchDescriptors.forEach(function (branch) {
      var option = document.createElement("option");
      option.value = branch.id;
      option.textContent = branch.label || branch.id;
      branchSelect.appendChild(option);
    });
    branchControls.hidden = state.branchDescriptors.length <= 1;
    branchSummary.hidden = state.branchDescriptors.length === 0;
  }

  function currentBranch() {
    if (!state.branchPayloads.length) return null;
    var requestedId = branchSelect.value || state.branchPayloads[0].id;
    for (var i = 0; i < state.branchPayloads.length; i += 1) {
      if (state.branchPayloads[i].id === requestedId) return state.branchPayloads[i];
    }
    return state.branchPayloads[0];
  }

  function formatInterval(interval) {
    if (!interval || interval.length !== 2) return "—";
    return "[" + formatNumber(interval[0]) + ", " + formatNumber(interval[1]) + "]";
  }

  function renderSelectedBranch() {
    var branch = currentBranch();
    if (!branch) {
      branchSummary.hidden = true;
      updateStats({});
      clearHistogram();
      return;
    }
    branchSummary.hidden = false;
    branchValid.textContent = formatInteger(branch.valid_results || 0);
    branchDiscarded.textContent = formatInteger(branch.discarded_results || 0);
    var numerical = branch.diagnostics || {};
    branchMcse.textContent = formatNumber(numerical.mcse_mean);
    branchCi95.textContent = formatInterval(numerical.ci95_mean);
    updateStats(branch.stats || {});
    renderHistogram(branch.histogram, branch.stats || {});
  }

  function updateBranchPayloads(branches) {
    state.branchPayloads = branches || [];
    renderSelectedBranch();
  }

  function clearHistogram() {
    d3.select("#histogram").selectAll("*").remove();
    emptyChart.hidden = false;
  }

  function renderHistogram(histogram, stats) {
    if (!histogram || !histogram.bins || histogram.bins.length === 0) {
      clearHistogram();
      histogramCount.textContent = "0 observaciones";
      return;
    }

    emptyChart.hidden = true;
    histogramCount.textContent = formatInteger(histogram.total_count) + " observaciones";

    var svg = d3.select("#histogram");
    svg.selectAll("*").remove();

    var width = 960;
    var height = 420;
    var margin = { top: 28, right: 28, bottom: 54, left: 70 };
    var innerWidth = width - margin.left - margin.right;
    var innerHeight = height - margin.top - margin.bottom;
    var bins = histogram.bins;
    var domainMin = Number(bins[0].x0);
    var domainMax = Number(bins[bins.length - 1].x1);
    var maxCount = d3.max(bins, function (bin) { return Number(bin.count); }) || 1;

    var x = d3.scale.linear().domain([domainMin, domainMax]).range([0, innerWidth]);
    var y = d3.scale.linear().domain([0, maxCount]).nice().range([innerHeight, 0]);
    var root = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var xAxis = d3.svg.axis().scale(x).orient("bottom").ticks(8).tickFormat(d3.format(".4g"));
    var yAxis = d3.svg.axis().scale(y).orient("left").ticks(6).tickFormat(d3.format("d"));

    root.append("g")
      .attr("class", "axis")
      .attr("transform", "translate(0," + innerHeight + ")")
      .call(xAxis);

    root.append("g").attr("class", "axis").call(yAxis);

    root.append("text")
      .attr("class", "axis-title")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight + 45)
      .attr("text-anchor", "middle")
      .text("Valor simulado");

    root.append("text")
      .attr("class", "axis-title")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerHeight / 2)
      .attr("y", -52)
      .attr("text-anchor", "middle")
      .text("Frecuencia");

    root.selectAll("rect.hist-bar")
      .data(bins)
      .enter()
      .append("rect")
      .attr("class", "hist-bar")
      .attr("x", function (bin) { return x(Number(bin.x0)) + 0.5; })
      .attr("width", function (bin) { return Math.max(1, x(Number(bin.x1)) - x(Number(bin.x0)) - 1); })
      .attr("y", innerHeight)
      .attr("height", 0)
      .transition()
      .duration(180)
      .attr("y", function (bin) { return y(Number(bin.count)); })
      .attr("height", function (bin) { return innerHeight - y(Number(bin.count)); });

    var markers = ["avg", "min", "max"];
    markers.forEach(function (name, index) {
      if (!stats || !Object.prototype.hasOwnProperty.call(stats, name)) return;
      var value = Number(stats[name]);
      if (!Number.isFinite(value) || value < domainMin || value > domainMax) return;
      var markerX = x(value);
      root.append("line")
        .attr("class", "hist-marker " + name)
        .attr("x1", markerX)
        .attr("x2", markerX)
        .attr("y1", 0)
        .attr("y2", innerHeight);
      root.append("text")
        .attr("class", "hist-marker-label " + name)
        .attr("x", markerX + 5)
        .attr("y", 14 + index * 15)
        .text(name + " " + formatNumber(value));
    });
  }

  function updateCounters(event) {
    state.totalIterations = Number(event.total_iterations || state.totalIterations || 0);
    iterationsValue.textContent = formatInteger(event.processed_iterations || 0) + " / " + formatInteger(state.totalIterations);
    candidateValue.textContent = formatInteger(event.candidate_results || 0);
    validValue.textContent = formatInteger(event.valid_results || 0);
    discardedValue.textContent = formatInteger(event.discarded_results || 0);
  }

  function clearReconcileTimer() {
    if (state.reconcileTimer !== null) {
      window.clearTimeout(state.reconcileTimer);
      state.reconcileTimer = null;
    }
  }

  function scheduleReconcile(delay) {
    if (!state.jobId || state.terminal || state.reconcileTimer !== null) return;
    state.reconcileTimer = window.setTimeout(function () {
      state.reconcileTimer = null;
      reconcileJob();
    }, delay || 500);
  }

  function finishUi(statusText, connectionText) {
    state.terminal = true;
    clearReconcileTimer();
    runButton.disabled = false;
    cancelButton.disabled = true;
    setStatus(statusText, connectionText);
  }

  function handleEvent(event) {
    if (!event || !event.type) return;

    if (event.type === "start") {
      state.terminal = false;
      state.totalIterations = Number(event.iterations || 0);
      seedValue.textContent = event.seed || "—";
      workersValue.textContent = String(event.workers || "—");
      iterationsValue.textContent = "0 / " + formatInteger(state.totalIterations);
      setRequestedStatistics(event.requested_statistics || []);
      configureBranches(event.branches || [{ id: "main", label: "Resultado" }]);
      setProgress(0);
      setStatus("Simulación en ejecución.", "WebSocket conectado");
      return;
    }

    if (event.type === "batch") {
      setProgress(event.progress);
      updateCounters(event);
      updateBranchPayloads(event.branches || []);
      setStatus("Procesando lote " + event.completed_batches + " de " + event.batch_count + ".", "Recibiendo lotes");
      return;
    }

    if (event.type === "complete") {
      setProgress(1);
      updateCounters(event);
      updateBranchPayloads(event.branches || []);
      finishUi("Simulación completada.", "Completado");
      return;
    }

    if (event.type === "cancelled") {
      finishUi("Simulación cancelada.", "Cancelado");
      return;
    }

    if (event.type === "error") {
      runtimeMessage.textContent = event.message || "La simulación terminó con un error.";
      runtimeMessage.hidden = false;
      finishUi("Error durante la simulación.", "Error");
    }
  }

  async function reconcileJob() {
    if (!state.jobId || state.terminal) return;
    try {
      var response = await fetch("/api/jobs/" + encodeURIComponent(state.jobId));
      if (!response.ok) {
        scheduleReconcile(1000);
        return;
      }
      var job = await response.json();
      if (job.last_event) handleEvent(job.last_event);
      if (job.terminal) {
        if (!state.terminal) finishUi("La ejecución finalizó.", job.status);
        return;
      }
      // Recupera el estado por HTTP si WebSocket se cierra antes del evento terminal.
      scheduleReconcile(500);
    } catch (_error) {
      scheduleReconcile(1000);
    }
  }

  function connectWebSocket(path) {
    var protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    var socket = new WebSocket(protocol + "//" + window.location.host + path);
    state.socket = socket;

    socket.addEventListener("open", function () {
      connectionState.textContent = "WebSocket conectado";
    });

    socket.addEventListener("message", function (message) {
      try {
        handleEvent(JSON.parse(message.data));
      } catch (_error) {
        runtimeMessage.textContent = "Se recibió un evento WebSocket inválido.";
        runtimeMessage.hidden = false;
      }
    });

    socket.addEventListener("error", function () {
      if (!state.terminal) {
        connectionState.textContent = "Error de WebSocket";
        reconcileJob();
      }
    });

    socket.addEventListener("close", function () {
      if (!state.terminal) {
        connectionState.textContent = "WebSocket cerrado";
        reconcileJob();
      }
    });
  }

  async function runModel() {
    if (!state.terminal) return;
    clearDiagnostics();
    resetResults();
    runButton.disabled = true;
    cancelButton.disabled = true;
    editor.disabled = true;
    state.terminal = false;
    setStatus("Compilando modelo…", "Enviando");

    try {
      var requestBody = { source: editor.value };
      var requestedSeed = seedInput.value.trim();
      if (requestedSeed) requestBody.seed = requestedSeed;

      var response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody)
      });
      var payload = await response.json();

      if (response.status === 422 && payload.error === "compilation_failed") {
        showDiagnostics(payload.diagnostics || []);
        finishUi("La compilación encontró errores.", "Compilación rechazada");
        return;
      }

      if (!response.ok) {
        runtimeMessage.textContent = payload.message || payload.error || "No fue posible iniciar la simulación.";
        runtimeMessage.hidden = false;
        finishUi("No se pudo iniciar el trabajo.", "Error HTTP");
        return;
      }

      state.jobId = payload.job_id;
      state.totalIterations = payload.iterations;
      seedValue.textContent = payload.seed;
      iterationsValue.textContent = "0 / " + formatInteger(payload.iterations);
      cancelButton.disabled = false;
      setStatus("Trabajo creado; conectando WebSocket…", "Conectando");
      connectWebSocket(payload.websocket_path);
    } catch (error) {
      runtimeMessage.textContent = "No se pudo contactar al backend: " + error.message;
      runtimeMessage.hidden = false;
      finishUi("Error de conexión con el backend.", "Sin conexión");
    } finally {
      editor.disabled = false;
    }
  }

  async function cancelJob() {
    if (!state.jobId || state.terminal) return;
    cancelButton.disabled = true;
    setStatus("Cancelando simulación…", "Cancelando");
    try {
      var response = await fetch("/api/jobs/" + encodeURIComponent(state.jobId) + "/cancel", { method: "POST" });
      if (!response.ok) throw new Error("HTTP " + response.status);
    } catch (error) {
      runtimeMessage.textContent = "No se pudo solicitar la cancelación: " + error.message;
      runtimeMessage.hidden = false;
      cancelButton.disabled = false;
    }
  }

  runButton.addEventListener("click", runModel);
  cancelButton.addEventListener("click", cancelJob);
  branchSelect.addEventListener("change", renderSelectedBranch);
})();
