/**
 * doctor.js — Telemedicine Doctor Visualization Dashboard
 * ========================================================
 *
 * Real-time clinical telemetry dashboard for the doctor during a telemedicine consultation.
 *
 * Requirements:
 * 1. Connect a WebSocket to ws://localhost:8000/ws/multimodal/doctor/room1
 * 2. Integrate Chart.js dynamic bar chart mapping 5 emotion classes:
 *    [Happy, Sad, Angry, Fear, Surprise]
 * 3. Place chart next to remoteVideo
 * 4. Listen for incoming JSON payloads; smoothly update Chart.js dataset with confidence scores
 */

(function (global) {
  'use strict';

  // ============================================================
  // CONFIGURATION & CONSTANTS
  // ============================================================

  const DEFAULT_WS_ENDPOINT = 'ws://localhost:8000/ws/multimodal/doctor/room1';

  // 5 Canonical target emotion classes
  const EMOTION_CLASSES = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise'];
  const EMOTION_KEYS = ['happy', 'sad', 'angry', 'fear', 'surprise'];

  // Color palette matching EmotionFusion design tokens
  const EMOTION_THEME = {
    happy: { bg: 'rgba(251, 191, 36, 0.75)', border: '#fbbf24', emoji: '😊' },
    sad: { bg: 'rgba(96, 165, 250, 0.75)', border: '#60a5fa', emoji: '😢' },
    angry: { bg: 'rgba(248, 113, 113, 0.75)', border: '#f87171', emoji: '😠' },
    fear: { bg: 'rgba(192, 132, 252, 0.75)', border: '#c084fc', emoji: '😨' },
    surprise: { bg: 'rgba(52, 211, 153, 0.75)', border: '#34d399', emoji: '😲' },
  };

  // ============================================================
  // DASHBOARD STATE
  // ============================================================

  let ws = null;
  let chartInstance = null;
  let isRunning = false;
  let activeWsUrl = DEFAULT_WS_ENDPOINT;
  let reconnectTimer = null;

  // Session analytics accumulator (Task 2)
  const sessionLog = [];

  // DOM Elements
  let $canvas = null;
  let $dominantEmoji = null;
  let $dominantName = null;
  let $dominantConf = null;
  let $dominantPill = null;
  let $statusBadge = null;
  let $modVision = null;
  let $modAudio = null;
  let $modText = null;
  let $transcriptText = null;

  // ============================================================
  // CHART.JS INITIALIZATION
  // ============================================================

  function initChart() {
    if (!$canvas) {
      $canvas = document.getElementById('emotionChart');
    }
    if (!$canvas) {
      console.warn('[DoctorDashboard] Canvas #emotionChart not found.');
      return null;
    }

    if (typeof Chart === 'undefined') {
      console.warn('[DoctorDashboard] Chart.js not loaded yet. Retrying in 200ms...');
      setTimeout(initChart, 200);
      return null;
    }

    if (chartInstance) {
      chartInstance.destroy();
      chartInstance = null;
    }

    const ctx = $canvas.getContext('2d');

    // Gradient background for bars
    const bgColors = EMOTION_KEYS.map(k => EMOTION_THEME[k].bg);
    const borderColors = EMOTION_KEYS.map(k => EMOTION_THEME[k].border);

    chartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: EMOTION_CLASSES,
        datasets: [{
          label: 'Confidence',
          data: [0, 0, 0, 0, 0], // Start at 0%
          backgroundColor: bgColors,
          borderColor: borderColors,
          borderWidth: 1.5,
          borderRadius: 8,
          borderSkipped: false,
          hoverBackgroundColor: borderColors,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: {
          padding: { top: 12, bottom: 4, left: 6, right: 12 },
        },
        animation: {
          duration: 400,
          easing: 'easeOutQuart',
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(14, 14, 22, 0.92)',
            titleColor: '#f0f0f5',
            bodyColor: '#22d3ee',
            titleFont: { family: "'Inter', sans-serif", size: 12, weight: '700' },
            bodyFont: { family: "'JetBrains Mono', monospace", size: 12 },
            borderColor: 'rgba(255, 255, 255, 0.12)',
            borderWidth: 1,
            padding: 10,
            cornerRadius: 8,
            displayColors: false,
            callbacks: {
              title: (items) => `${items[0].label} Emotion`,
              label: (context) => `Probability: ${context.parsed.y.toFixed(1)}%`,
            },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              color: '#f0f0f5',
              font: { family: "'Inter', sans-serif", size: 12, weight: 600 },
            },
          },
          y: {
            beginAtZero: true,
            max: 100,
            ticks: {
              stepSize: 25,
              color: '#8b8b9e',
              font: { family: "'JetBrains Mono', monospace", size: 11 },
              callback: (val) => val + '%',
            },
            grid: {
              color: 'rgba(255, 255, 255, 0.05)',
              drawBorder: false,
            },
          },
        },
      },
    });

    console.log('[DoctorDashboard] Chart.js initialized successfully.');
    return chartInstance;
  }

  // ============================================================
  // WEBSOCKET TELEMETRY CLIENT
  // ============================================================

  function connectWebSocket(url = DEFAULT_WS_ENDPOINT) {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    activeWsUrl = url;
    console.log('[DoctorDashboard] Connecting to WebSocket:', url);
    updateStatusBadge('Connecting…', 'connecting');

    try {
      ws = new WebSocket(url);
    } catch (err) {
      console.error('[DoctorDashboard] WebSocket connection failed:', err);
      scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      console.log('[DoctorDashboard] WebSocket connected to doctor telemetry channel.');
      updateStatusBadge('Live Telemetry', 'live');
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        handleIncomingTelemetry(payload);
      } catch (err) {
        console.warn('[DoctorDashboard] Error parsing incoming payload:', err, event.data);
      }
    };

    ws.onerror = (err) => {
      console.error('[DoctorDashboard] WebSocket error:', err);
      updateStatusBadge('Error', 'error');
    };

    ws.onclose = (event) => {
      console.warn('[DoctorDashboard] WebSocket closed (code:', event.code, ').');
      updateStatusBadge('Disconnected', 'disconnected');
      if (isRunning) {
        scheduleReconnect();
      }
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer || !isRunning) return;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      if (isRunning) {
        console.log('[DoctorDashboard] Attempting reconnect...');
        connectWebSocket(activeWsUrl);
      }
    }, 2000);
  }

  // ============================================================
  // TELEMETRY PAYLOAD HANDLER
  // ============================================================

  function handleIncomingTelemetry(data) {
    if (!data) return;

    let probsMap = null;
    let label = null;
    let confidence = 0;
    let modalitiesUsed = [];

    // Case A: Fusion prediction available (5-class target space)
    if (data.fusion_prediction && data.fusion_prediction.probs) {
      probsMap = data.fusion_prediction.probs;
      label = data.fusion_prediction.label;
      confidence = data.fusion_prediction.confidence || 0;
      modalitiesUsed = data.fusion_prediction.modalities_used || [];
    }
    // Case B: ViT fallback
    else if (data.vit_prediction) {
      label = data.vit_prediction.label;
      confidence = data.vit_prediction.confidence || 0;
      modalitiesUsed = ['vision'];
      probsMap = { [label.toLowerCase()]: confidence };
    }

    if (!probsMap) {
      return;
    }

    // Map probabilities into canonical [Happy, Sad, Angry, Fear, Surprise] order
    const updatedScores = EMOTION_KEYS.map((key) => {
      let val = probsMap[key];
      if (val == null) {
        // Try capitalized key
        val = probsMap[key.charAt(0).toUpperCase() + key.slice(1)];
      }
      if (val == null) return 0;
      // Convert to 0..100% scale
      const num = Number(val);
      return num <= 1 ? Number((num * 100).toFixed(1)) : Number(num.toFixed(1));
    });

    // Requirement 4: Smoothly update the Chart.js dataset
    if (chartInstance) {
      chartInstance.data.datasets[0].data = updatedScores;
      chartInstance.update(); // Executes smooth Chart.js interpolation animation
    } else {
      initChart();
      if (chartInstance) {
        chartInstance.data.datasets[0].data = updatedScores;
        chartInstance.update();
      }
    }

    // Update DOM indicators
    updateSummaryCards(label, confidence, modalitiesUsed, data.transcript);

    // Live transcript: directly assign to prevent text duplication
    const payload = data;
    if (document.getElementById('transcriptBox') && payload.transcript !== undefined) {
      document.getElementById('transcriptBox').textContent = payload.transcript;
    }

    // Task 2: Accumulate session log for post-call analytics
    if (data.transcript && data.transcript.trim()) {
      sessionLog.push({
        timestamp: Date.now(),
        label: label,
        probs: Object.assign({}, probsMap),
        text: data.transcript.trim(),
      });
    }
  }

  function updateSummaryCards(label, confidence, modalities, transcript) {
    const normKey = (label || '').toLowerCase();
    const theme = EMOTION_THEME[normKey] || { emoji: '🔍', border: '#6c8aff' };
    const confPct = (confidence <= 1 ? confidence * 100 : confidence).toFixed(1) + '%';

    if ($dominantEmoji) {
      $dominantEmoji.textContent = theme.emoji;
    }
    if ($dominantName) {
      $dominantName.textContent = label ? label.charAt(0).toUpperCase() + label.slice(1) : 'Detecting…';
    }
    if ($dominantConf) {
      $dominantConf.textContent = `${confPct} Confidence`;
    }
    if ($dominantPill) {
      $dominantPill.textContent = label ? `${label.toUpperCase()} · ${confPct}` : '—';
      $dominantPill.style.borderColor = theme.border;
      $dominantPill.style.color = theme.border;
    }

    // Modality indicators (Tailwind-native active states)
    function setPillActive(el, active) {
      if (!el) return;
      if (active) {
        el.className = 'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold bg-cyan-500/15 text-cyan-300 border border-cyan-500/40 transition-all';
      } else {
        el.className = 'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-gray-800 text-gray-500 border border-gray-700/60 transition-all';
      }
    }
    setPillActive($modVision, modalities.includes('vision'));
    setPillActive($modAudio, modalities.includes('audio'));
    setPillActive($modText, modalities.includes('text'));

    // Live speech transcript
    if ($transcriptText && transcript) {
      $transcriptText.textContent = `"${transcript}"`;
    }
  }

  function updateStatusBadge(text, state) {
    if (!$statusBadge) {
      $statusBadge = document.getElementById('telemetryStatusBadge');
    }
    if ($statusBadge) {
      $statusBadge.textContent = text;
      const base = 'text-[10px] font-semibold px-2.5 py-0.5 rounded-full border ';
      if (state === 'live') {
        $statusBadge.className = base + 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
      } else if (state === 'connecting') {
        $statusBadge.className = base + 'bg-amber-500/10 text-amber-400 border-amber-500/30';
      } else if (state === 'error' || state === 'disconnected') {
        $statusBadge.className = base + 'bg-red-500/10 text-red-400 border-red-500/30';
      } else {
        $statusBadge.className = base + 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30';
      }
    }
  }

  // ============================================================
  // POST-CALL ANALYTICS RENDERER (Task 3)
  // ============================================================

  let timelineChartInstance = null;

  function renderPostCallDashboard() {
    console.log('[DoctorDashboard] Rendering post-call analytics (' + sessionLog.length + ' entries).');

    // ── Timeline badge ──
    const $badge = document.getElementById('pcTimelineBadge');
    if ($badge) {
      $badge.textContent = sessionLog.length + ' data points';
    }

    // ── 1. Session Timeline Line Chart ──
    renderTimelineChart();

    // ── 2. GRE-Style Interactive Transcript ──
    renderInteractiveTranscript();
  }

  function renderTimelineChart() {
    const $canvas = document.getElementById('sessionTimelineChart');
    if (!$canvas || typeof Chart === 'undefined') return;

    if (timelineChartInstance) {
      timelineChartInstance.destroy();
      timelineChartInstance = null;
    }

    if (!sessionLog || sessionLog.length === 0) return;

    // Build time-axis labels (relative seconds from session start)
    const t0 = sessionLog[0].timestamp || Date.now();
    const timeLabels = sessionLog.map(e => {
      const ts = e.timestamp || t0;
      const elapsedSec = Math.max(0, (ts - t0) / 1000);
      return elapsedSec.toFixed(0) + 's';
    });

    // Build one dataset per emotion mapping timestamp and probs dictionaries from sessionLog
    const datasets = EMOTION_KEYS.map((key, idx) => {
      const theme = EMOTION_THEME[key];
      return {
        label: EMOTION_CLASSES[idx],
        data: sessionLog.map(e => {
          if (!e || !e.probs || typeof e.probs !== 'object') return 0;
          const v = e.probs[key] ?? e.probs[key.charAt(0).toUpperCase() + key.slice(1)] ?? 0;
          const num = Number(v);
          if (isNaN(num)) return 0;
          return num <= 1 ? Number(num.toFixed(3)) : Number((num / 100).toFixed(3));
        }),
        borderColor: theme.border,
        backgroundColor: theme.bg.replace('0.75', '0.12'),
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 5,
        tension: 0.35,
        fill: false,
      };
    });

    const ctx = $canvas.getContext('2d');
    timelineChartInstance = new Chart(ctx, {
      type: 'line',
      data: { labels: timeLabels, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        layout: { padding: { top: 8, bottom: 4, left: 4, right: 12 } },
        animation: { duration: 600, easing: 'easeOutQuart' },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: '#8b8b9e',
              font: { family: "'Inter', sans-serif", size: 11, weight: 600 },
              boxWidth: 10, boxHeight: 10, borderRadius: 3,
              padding: 14, usePointStyle: true, pointStyle: 'circle',
            },
          },
          tooltip: {
            backgroundColor: 'rgba(14,14,22,0.92)',
            titleColor: '#f0f0f5',
            bodyColor: '#22d3ee',
            titleFont: { family: "'Inter', sans-serif", size: 12, weight: '700' },
            bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
            borderColor: 'rgba(255,255,255,0.12)',
            borderWidth: 1, padding: 10, cornerRadius: 8,
            callbacks: {
              label: (ctx) => ctx.dataset.label + ': ' + (ctx.parsed.y * 100).toFixed(1) + '%',
            },
          },
        },
        scales: {
          x: {
            title: {
              display: true, text: 'Session Time',
              color: '#8b8b9e',
              font: { family: "'Inter', sans-serif", size: 11, weight: 600 },
            },
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: {
              color: '#8b8b9e',
              font: { family: "'JetBrains Mono', monospace", size: 10 },
              maxTicksLimit: 20,
            },
          },
          y: {
            beginAtZero: true, max: 1,
            title: {
              display: true, text: 'Confidence',
              color: '#8b8b9e',
              font: { family: "'Inter', sans-serif", size: 11, weight: 600 },
            },
            ticks: {
              stepSize: 0.25,
              color: '#8b8b9e',
              font: { family: "'JetBrains Mono', monospace", size: 10 },
              callback: (v) => (v * 100).toFixed(0) + '%',
            },
            grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
          },
        },
      },
    });
  }

  // Tailwind emotion highlighting styles for GRE-style sentiment analysis
  const TAILWIND_EMOTION_STYLES = {
    happy: {
      bg: 'bg-yellow-900/60',
      text: 'text-yellow-200',
      border: 'border-yellow-600/60',
      tagBg: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40',
    },
    sad: {
      bg: 'bg-blue-900/60',
      text: 'text-blue-200',
      border: 'border-blue-600/60',
      tagBg: 'bg-blue-500/20 text-blue-300 border-blue-500/40',
    },
    angry: {
      bg: 'bg-red-900/60',
      text: 'text-red-200',
      border: 'border-red-600/60',
      tagBg: 'bg-red-500/20 text-red-300 border-red-500/40',
    },
    fear: {
      bg: 'bg-purple-900/60',
      text: 'text-purple-200',
      border: 'border-purple-600/60',
      tagBg: 'bg-purple-500/20 text-purple-300 border-purple-500/40',
    },
    surprise: {
      bg: 'bg-emerald-900/60',
      text: 'text-emerald-200',
      border: 'border-emerald-600/60',
      tagBg: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    },
  };

  function renderInteractiveTranscript() {
    const $container = document.getElementById('interactiveTranscript');
    if (!$container) return;

    if (!sessionLog || sessionLog.length === 0) {
      $container.innerHTML =
        '<div class="h-full flex flex-col items-center justify-center text-gray-500 gap-2">' +
        '<span class="text-3xl">📝</span>' +
        '<p class="text-xs">No transcript data was captured during this session.</p>' +
        '</div>';
      return;
    }

    // Helper: Normalize text for matching (case-insensitive, strip punctuation/extra spaces)
    const cleanText = (str) =>
      (str || '')
        .toLowerCase()
        .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?!]/g, '')
        .replace(/\s+/g, ' ')
        .trim();

    // Helper: Split text into candidate sentences before punctuation marks or pauses
    const splitIntoSentences = (text) => {
      if (!text || !text.trim()) return [];
      // Protect common honorifics/abbreviations before splitting
      const protectedText = text.replace(/(?<=\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr))\./gi, '@@DOT@@');
      const matches = protectedText.match(/(?:[^.!?]|\d\.\d)+[.!?]*(?:\s+|$)/g);
      if (!matches) return [text.trim()];
      return matches
        .map((s) => s.replace(/@@DOT@@/g, '.').trim())
        .filter((s) => s.length > 0);
    };

    // Helper: Calculate dominant emotion (mode of labels collected while sentence was active)
    const getModeEmotion = (labels) => {
      if (!labels || labels.length === 0) return 'neutral';
      const counts = {};
      for (const raw of labels) {
        if (!raw) continue;
        const norm = String(raw).toLowerCase().trim();
        if (!norm) continue;
        counts[norm] = (counts[norm] || 0) + 1;
      }

      let dominant = 'neutral';
      let maxCount = 0;
      for (const [label, count] of Object.entries(counts)) {
        if (count > maxCount) {
          maxCount = count;
          dominant = label;
        } else if (count === maxCount && dominant === 'neutral') {
          dominant = label;
        }
      }
      return dominant;
    };

    // Sentence-Extraction Parser iterating through sessionLog
    const PAUSE_THRESHOLD_MS = 3500;
    const extracted = []; // array of { text, labels: [], lastTimestamp, hasPunctuation, isClosed }
    let prevTimestamp = 0;

    for (let i = 0; i < sessionLog.length; i++) {
      const entry = sessionLog[i];
      if (!entry || !entry.text || !entry.text.trim()) continue;

      const entryTime = entry.timestamp || 0;
      const isPause = prevTimestamp > 0 && entryTime - prevTimestamp > PAUSE_THRESHOLD_MS;
      prevTimestamp = entryTime;

      // When a pause is detected, finalize the preceding active sentence
      if (isPause && extracted.length > 0) {
        extracted[extracted.length - 1].isClosed = true;
      }

      const clauses = splitIntoSentences(entry.text);
      if (clauses.length === 0) continue;

      for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
        const clause = clauses[cIdx];
        const normClause = cleanText(clause);
        if (!normClause) continue;

        const endsWithPunct = /[.!?]$/.test(clause.trim());
        const isLastClauseInEntry = cIdx === clauses.length - 1;

        // Search recent sentences in rolling window (up to last 4 items)
        let matchIdx = -1;
        const searchStart = Math.max(0, extracted.length - 4);

        for (let j = extracted.length - 1; j >= searchStart; j--) {
          const item = extracted[j];
          const normExisting = cleanText(item.text);

          // Exact match
          if (normExisting === normClause) {
            matchIdx = j;
            break;
          }

          // If existing sentence is not yet closed/completed, check for prefix expansion
          if (!item.isClosed && !item.hasPunctuation) {
            if (normClause.startsWith(normExisting) || normExisting.startsWith(normClause)) {
              matchIdx = j;
              break;
            }
          }
        }

        if (matchIdx !== -1) {
          const matched = extracted[matchIdx];
          const normExisting = cleanText(matched.text);

          let wasExtended = false;
          // Retain only the longest/most complete string of text before a pause or punctuation mark
          if (clause.length > matched.text.length && normClause.startsWith(normExisting)) {
            matched.text = clause;
            matched.hasPunctuation = endsWithPunct;
            wasExtended = true;
          } else if (endsWithPunct && !matched.hasPunctuation) {
            matched.text = clause;
            matched.hasPunctuation = true;
            wasExtended = true;
          }

          // Sentence was active during this log entry if it was extended or is the latest spoken clause
          if (entry.label && (wasExtended || isLastClauseInEntry)) {
            matched.labels.push(entry.label);
          }
          matched.lastTimestamp = entryTime;
        } else {
          // New unique sentence
          extracted.push({
            text: clause,
            labels: entry.label ? [entry.label] : [],
            lastTimestamp: entryTime,
            hasPunctuation: endsWithPunct,
            isClosed: endsWithPunct,
          });
        }
      }
    }

    if (extracted.length === 0) {
      $container.innerHTML =
        '<div class="h-full flex flex-col items-center justify-center text-gray-500 gap-2">' +
        '<span class="text-3xl">📝</span>' +
        '<p class="text-xs">No distinct transcript segments found.</p>' +
        '</div>';
      return;
    }

    // Generate HTML string of <span class="gre-sentence"> elements
    let htmlString = '<p class="text-sm leading-relaxed text-gray-300">';
    for (let idx = 0; idx < extracted.length; idx++) {
      const item = extracted[idx];
      const sentence = item.text;
      const dominantEmotion = getModeEmotion(item.labels);
      htmlString += `<span class="gre-sentence p-1 rounded cursor-pointer transition-colors duration-200" data-emotion="${dominantEmotion}">${sentence}</span> `;
    }
    htmlString += '</p>';

    // Append finalized HTML string to interactiveTranscript.innerHTML
    $container.innerHTML = htmlString;

    // Attach click listeners for Tailwind emotion sentiment highlights
    const HIGHLIGHT_STYLES = {
      happy: ['bg-yellow-900/70', 'text-yellow-200', 'ring-1', 'ring-yellow-500/40'],
      sad: ['bg-blue-900/70', 'text-blue-200', 'ring-1', 'ring-blue-500/40'],
      angry: ['bg-red-900/70', 'text-red-200', 'ring-1', 'ring-red-500/40'],
      fear: ['bg-purple-900/70', 'text-purple-200', 'ring-1', 'ring-purple-500/40'],
      surprise: ['bg-emerald-900/70', 'text-emerald-200', 'ring-1', 'ring-emerald-500/40'],
      neutral: ['bg-gray-800', 'text-gray-200', 'ring-1', 'ring-gray-600'],
    };

    $container.querySelectorAll('.gre-sentence').forEach((span) => {
      span.addEventListener('click', () => {
        const emotion = (span.getAttribute('data-emotion') || 'neutral').toLowerCase();
        const classes = HIGHLIGHT_STYLES[emotion] || HIGHLIGHT_STYLES.neutral;
        const isHighlighted = span.classList.contains('is-highlighted');

        if (isHighlighted) {
          span.classList.remove('is-highlighted', ...classes);
          span.removeAttribute('title');
        } else {
          span.classList.add('is-highlighted', ...classes);
          span.setAttribute('title', `Dominant emotion: ${emotion.toUpperCase()}`);
        }
      });
    });
  }

  // ============================================================
  // TAILWIND DISPLAY TOGGLES (Task 2)
  // ============================================================

  function showPostCallDashboard() {
    const $mainStage = document.getElementById('mainStage');
    const $postCall = document.getElementById('postCallDashboard');
    if ($mainStage) $mainStage.classList.add('hidden');
    if ($postCall) {
      $postCall.classList.remove('hidden');
      $postCall.classList.add('flex');
    }
    renderPostCallDashboard();
  }

  function hidePostCallDashboard() {
    const $mainStage = document.getElementById('mainStage');
    const $postCall = document.getElementById('postCallDashboard');
    if ($postCall) {
      $postCall.classList.add('hidden');
      $postCall.classList.remove('flex');
    }
    if ($mainStage) $mainStage.classList.remove('hidden');
  }

  function leaveCall() {
    stopDashboard();
    showPostCallDashboard();
  }

  // ============================================================
  // PUBLIC CONTROLLER API
  // ============================================================

  function startDashboard(wsUrl = DEFAULT_WS_ENDPOINT) {
    bindDomElements();
    isRunning = true;
    initChart();
    connectWebSocket(wsUrl);
    console.log('[DoctorDashboard] Started dashboard with target:', wsUrl);
  }

  function stopDashboard() {
    isRunning = false;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (ws) {
      ws.close();
      ws = null;
    }
    if (chartInstance) {
      chartInstance.destroy();
      chartInstance = null;
    }
    updateStatusBadge('Stopped', 'disconnected');
    console.log('[DoctorDashboard] Stopped dashboard.');
  }

  function bindDomElements() {
    $canvas = document.getElementById('emotionChart');
    $dominantEmoji = document.getElementById('dominantEmoji');
    $dominantName = document.getElementById('dominantEmotionName');
    $dominantConf = document.getElementById('dominantConfidence');
    $dominantPill = document.getElementById('dominantPill');
    $statusBadge = document.getElementById('telemetryStatusBadge');
    $modVision = document.getElementById('modVisionBadge');
    $modAudio = document.getElementById('modAudioBadge');
    $modText = document.getElementById('modTextBadge');
    $transcriptText = document.getElementById('transcriptBox') || document.getElementById('doctorTranscriptText');
  }

  // ============================================================
  // GLOBAL EXPORT & AUTO-INITIALIZATION
  // ============================================================

  global.DoctorDashboard = {
    start: startDashboard,
    stop: stopDashboard,
    initChart: initChart,
    leaveCall: leaveCall,
    showPostCallDashboard: showPostCallDashboard,
    hidePostCallDashboard: hidePostCallDashboard,
    renderPostCallDashboard: renderPostCallDashboard,
    get chart() { return chartInstance; },
    get isRunning() { return isRunning; },
    get sessionLog() { return sessionLog; },
  };

  document.addEventListener('DOMContentLoaded', () => {
    bindDomElements();
    // Auto-start doctor telemetry on room1 by default
    startDashboard(DEFAULT_WS_ENDPOINT);
  });

})(typeof window !== 'undefined' ? window : this);

