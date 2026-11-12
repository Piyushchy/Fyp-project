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
    happy:    { bg: 'rgba(251, 191, 36, 0.75)',  border: '#fbbf24', emoji: '😊' },
    sad:      { bg: 'rgba(96, 165, 250, 0.75)',  border: '#60a5fa', emoji: '😢' },
    angry:    { bg: 'rgba(248, 113, 113, 0.75)', border: '#f87171', emoji: '😠' },
    fear:     { bg: 'rgba(192, 132, 252, 0.75)', border: '#c084fc', emoji: '😨' },
    surprise: { bg: 'rgba(52, 211, 153, 0.75)',  border: '#34d399', emoji: '😲' },
  };

  // ============================================================
  // DASHBOARD STATE
  // ============================================================

  let ws = null;
  let chartInstance = null;
  let isRunning = false;
  let activeWsUrl = DEFAULT_WS_ENDPOINT;
  let reconnectTimer = null;

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

    // Modality indicators
    if ($modVision) $modVision.classList.toggle('active', modalities.includes('vision'));
    if ($modAudio)  $modAudio.classList.toggle('active', modalities.includes('audio'));
    if ($modText)   $modText.classList.toggle('active', modalities.includes('text'));

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
      $statusBadge.className = 'card-badge status-' + state;
    }
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
    $transcriptText = document.getElementById('doctorTranscriptText');
  }

  // ============================================================
  // GLOBAL EXPORT & AUTO-INITIALIZATION
  // ============================================================

  global.DoctorDashboard = {
    start: startDashboard,
    stop: stopDashboard,
    initChart: initChart,
    get chart() { return chartInstance; },
    get isRunning() { return isRunning; },
  };

  document.addEventListener('DOMContentLoaded', () => {
    bindDomElements();
    // Auto-start doctor telemetry on room1 by default
    startDashboard(DEFAULT_WS_ENDPOINT);
  });

})(typeof window !== 'undefined' ? window : this);

