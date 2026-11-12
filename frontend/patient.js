/**
 * patient.js — Background ML Data Extraction for Patient Telemedicine Call
 * =========================================================================
 *
 * Runs on the patient side of the telemedicine session to continuously extract:
 *   1. Video frames: Hidden <canvas> snapshots from <video id="localVideo"> (every 1.5s, JPEG q=0.7)
 *   2. Audio PCM: Web Audio API AudioContext buffering mic stream (every 1.5s, 16-bit WAV base64)
 *   3. Text transcript: Continuous SpeechRecognition rolling buffer (cleared on isFinal)
 *
 * Packages image, audio, and text into a single JSON payload and streams it to:
 *   ws://localhost:8000/ws/multimodal/patient/room1
 *
 * Uses an `isAwaitingResponse` backpressure variable so subsequent frames are
 * never dispatched until the server acknowledges the previous inference cycle.
 */

(function (global) {
  'use strict';

  // ============================================================
  // CONFIGURATION & CONSTANTS
  // ============================================================

  const WS_ENDPOINT = 'ws://localhost:8000/ws/multimodal/patient/room1';
  const SAMPLE_INTERVAL_MS = 1500; // 1.5 seconds
  const JPEG_QUALITY = 0.7;

  // ============================================================
  // MODULE STATE
  // ============================================================

  let ws = null;
  let isExtractionActive = false;
  let isAwaitingResponse = false; // Backpressure lock
  let sampleTimer = null;

  // Image extraction state
  let hiddenCanvas = null;
  let canvasCtx = null;

  // Audio extraction state
  let audioContext = null;
  let audioInputSource = null;
  let scriptProcessor = null;
  let pcmChunks = [];
  let audioHookedStream = null;

  // Text / SpeechRecognition state (Task 2: Sliding Text Window)
  let speechRecognizer = null;
  const completedSentences = [];  // Last 3 finished sentences (sliding window)
  let interimTranscript = '';

  // Audio calibration state (Task 1: Pre-Call Noise Floor)
  let noiseFloor = 0;
  const NOISE_GATE_MULTIPLIER = 1.5;

  // ============================================================
  // 1. WEBSOCKET CONNECTION
  // ============================================================

  function connectWebSocket(url = WS_ENDPOINT) {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    console.log('[PatientML] Connecting to WebSocket:', url);
    ws = new WebSocket(url);

    ws.onopen = () => {
      console.log('[PatientML] WebSocket connected successfully.');
      isAwaitingResponse = false;
    };

    ws.onmessage = (event) => {
      // Server acknowledged receipt of the payload -> release backpressure
      isAwaitingResponse = false;
      try {
        const ack = JSON.parse(event.data);
        console.debug('[PatientML] Server acknowledgment received:', ack);
      } catch (e) {
        console.debug('[PatientML] Server response:', event.data);
      }
    };

    ws.onerror = (err) => {
      console.error('[PatientML] WebSocket error:', err);
      isAwaitingResponse = false;
    };

    ws.onclose = () => {
      console.warn('[PatientML] WebSocket disconnected.');
      isAwaitingResponse = false;
      if (isExtractionActive) {
        setTimeout(() => connectWebSocket(url), 2000);
      }
    };
  }

  // ============================================================
  // 2. IMAGE EXTRACTION (Hidden Canvas -> Base64 JPEG)
  // ============================================================

  function setupHiddenCanvas() {
    if (!hiddenCanvas) {
      hiddenCanvas = document.getElementById('patientHiddenCanvas');
      if (!hiddenCanvas) {
        hiddenCanvas = document.createElement('canvas');
        hiddenCanvas.id = 'patientHiddenCanvas';
        hiddenCanvas.style.display = 'none';
        hiddenCanvas.setAttribute('aria-hidden', 'true');
        document.body.appendChild(hiddenCanvas);
      }
    }
    if (!canvasCtx) {
      canvasCtx = hiddenCanvas.getContext('2d', { willReadFrequently: true });
    }
  }

  function captureLocalFrame() {
    const video = document.getElementById('localVideo');
    if (!video || !video.videoWidth || !video.videoHeight || video.paused || video.ended) {
      return null;
    }

    setupHiddenCanvas();

    hiddenCanvas.width = video.videoWidth;
    hiddenCanvas.height = video.videoHeight;
    canvasCtx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);

    const dataUrl = hiddenCanvas.toDataURL('image/jpeg', JPEG_QUALITY);
    const commaIndex = dataUrl.indexOf(',');
    return commaIndex !== -1 ? dataUrl.slice(commaIndex + 1) : dataUrl;
  }

  // ============================================================
  // 3. AUDIO EXTRACTION (AudioContext PCM -> 16-bit WAV Base64)
  // ============================================================

  function hookAudioStream(stream) {
    if (!stream || stream === audioHookedStream) return;
    const audioTracks = stream.getAudioTracks();
    if (!audioTracks || audioTracks.length === 0) return;

    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!audioContext) {
        audioContext = new AudioCtx({ sampleRate: 16000 });
      }

      if (audioContext.state === 'suspended') {
        audioContext.resume();
      }

      if (scriptProcessor) {
        scriptProcessor.disconnect();
      }
      if (audioInputSource) {
        audioInputSource.disconnect();
      }

      audioHookedStream = stream;
      audioInputSource = audioContext.createMediaStreamSource(stream);

      // Buffer size: 4096 samples, 1 input channel (mono)
      scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
      audioInputSource.connect(scriptProcessor);
      scriptProcessor.connect(audioContext.destination);

      pcmChunks = [];
      scriptProcessor.onaudioprocess = (event) => {
        if (!isExtractionActive) return;
        const channelData = event.inputBuffer.getChannelData(0);
        pcmChunks.push(new Float32Array(channelData));
      };

      console.log('[PatientML] Audio stream hooked into AudioContext at', audioContext.sampleRate, 'Hz');
    } catch (err) {
      console.warn('[PatientML] Failed to hook audio stream:', err);
    }
  }

  function getAndFlushAudioWav() {
    if (!audioContext || pcmChunks.length === 0) {
      return null;
    }

    // Merge PCM chunks
    let totalLength = 0;
    for (let i = 0; i < pcmChunks.length; i++) {
      totalLength += pcmChunks[i].length;
    }

    if (totalLength === 0) return null;

    const mergedPcm = new Float32Array(totalLength);
    let offset = 0;
    for (let i = 0; i < pcmChunks.length; i++) {
      mergedPcm.set(pcmChunks[i], offset);
      offset += pcmChunks[i].length;
    }
    pcmChunks = []; // Clear buffer for the next 1.5s interval

    // Task 1: Silence gate — drop audio chunk if RMS is below noise floor threshold
    if (noiseFloor > 0) {
      let sumSq = 0;
      for (let i = 0; i < mergedPcm.length; i++) {
        sumSq += mergedPcm[i] * mergedPcm[i];
      }
      const activeRms = Math.sqrt(sumSq / mergedPcm.length);
      if (activeRms < NOISE_GATE_MULTIPLIER * noiseFloor) {
        console.debug('[PatientML] Audio below noise gate (RMS:', activeRms.toFixed(6),
          '< threshold:', (NOISE_GATE_MULTIPLIER * noiseFloor).toFixed(6), ') — dropping.');
        return null;
      }
    }

    return encodeWavBase64(mergedPcm, audioContext.sampleRate);
  }

  function encodeWavBase64(samples, sampleRate) {
    const numSamples = samples.length;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);

    function writeStr(offset, str) {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    }

    // RIFF chunk descriptor
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeStr(8, 'WAVE');

    // "fmt " sub-chunk
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);          // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true);           // AudioFormat (1 = PCM)
    view.setUint16(22, 1, true);           // NumChannels (1 = Mono)
    view.setUint32(24, sampleRate, true);  // SampleRate
    view.setUint32(28, sampleRate * 2, true); // ByteRate (SampleRate * NumChannels * BitsPerSample/8)
    view.setUint16(32, 2, true);           // BlockAlign (NumChannels * BitsPerSample/8)
    view.setUint16(34, 16, true);          // BitsPerSample (16 bits)

    // "data" sub-chunk
    writeStr(36, 'data');
    view.setUint32(40, numSamples * 2, true);

    // Write 16-bit PCM samples
    let byteOffset = 44;
    for (let i = 0; i < numSamples; i++) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      s = s < 0 ? s * 0x8000 : s * 0x7FFF;
      view.setInt16(byteOffset, s, true);
      byteOffset += 2;
    }

    // Convert ArrayBuffer to binary string and base64 encode
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const chunkSize = 8192;
    for (let i = 0; i < bytes.byteLength; i += chunkSize) {
      const slice = bytes.subarray(i, i + chunkSize);
      binary += String.fromCharCode.apply(null, slice);
    }

    return btoa(binary);
  }

  // ============================================================
  // 4. TEXT EXTRACTION (Web Speech API Rolling Buffer)
  // ============================================================

  function initSpeechRecognition() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      console.warn('[PatientML] Web Speech API is not supported in this browser.');
      return;
    }

    speechRecognizer = new SpeechRec();
    speechRecognizer.continuous = true;
    speechRecognizer.interimResults = true;
    speechRecognizer.lang = 'en-US';

    speechRecognizer.onresult = (event) => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const text = result[0].transcript.trim();

        if (result.isFinal) {
          // Task 2: Push completed sentence into sliding window (max 3)
          completedSentences.push(text);
          if (completedSentences.length > 3) {
            completedSentences.shift();
          }
        } else {
          interim += (interim ? ' ' : '') + text;
        }
      }
      interimTranscript = interim;
    };

    speechRecognizer.onerror = (event) => {
      console.warn('[PatientML] Speech recognition warning/error:', event.error);
      if (isExtractionActive && (event.error === 'no-speech' || event.error === 'audio-capture')) {
        setTimeout(restartSpeechRecognition, 400);
      }
    };

    speechRecognizer.onend = () => {
      // Keep running continuously while extraction session is active
      if (isExtractionActive) {
        setTimeout(restartSpeechRecognition, 200);
      }
    };

    try {
      speechRecognizer.start();
      console.log('[PatientML] SpeechRecognition continuous listener started.');
    } catch (err) {
      console.warn('[PatientML] SpeechRecognition start failed:', err);
    }
  }

  function restartSpeechRecognition() {
    if (!speechRecognizer || !isExtractionActive) return;
    try {
      speechRecognizer.start();
    } catch (e) {
      // Ignore if already active
    }
  }

  function getAndConsumeTranscript() {
    // Task 2: Combine sliding window of completed sentences + current interim
    const sentenceContext = completedSentences.join(' ');
    const combined = (sentenceContext + (interimTranscript ? ' ' + interimTranscript : '')).trim();

    return combined || null;
  }

  // ============================================================
  // 5. PACKAGING & BACKPRESSURE DISPATCH (Every 1.5s)
  // ============================================================

  function dispatchExtractionCycle() {
    if (!isExtractionActive) return;

    // Check audio stream availability from localVideo if not yet hooked
    if (!audioHookedStream) {
      const video = document.getElementById('localVideo');
      if (video && video.srcObject) {
        hookAudioStream(video.srcObject);
      }
    }

    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }

    // Requirement 5: Backpressure check — wait until server acknowledges previous payload
    if (isAwaitingResponse) {
      console.debug('[PatientML] Awaiting server response — backpressure throttle active.');
      return;
    }

    // Collect modalities
    const imageB64 = captureLocalFrame();
    const audioB64 = getAndFlushAudioWav();
    const transcriptText = getAndConsumeTranscript();

    // Package into single payload
    const payload = {};
    if (imageB64) payload.image = imageB64;
    if (audioB64) payload.audio = audioB64;
    if (transcriptText) payload.text = transcriptText;

    // Dispatch only if at least one modality has active data
    if (Object.keys(payload).length === 0) {
      return;
    }

    isAwaitingResponse = true;
    try {
      ws.send(JSON.stringify(payload));
      console.debug('[PatientML] Dispatched multimodal payload (modalities:', Object.keys(payload).join(', '), ')');
    } catch (err) {
      console.error('[PatientML] Failed to send payload:', err);
      isAwaitingResponse = false;
    }
  }

  // ============================================================
  // 6. PRE-CALL AUDIO CALIBRATION (Task 1: Noise Floor RMS)
  // ============================================================

  /**
   * Capture 5 seconds of microphone audio and compute the RMS noise floor.
   * Returns a Promise that resolves after the calibration window.
   *
   * @param {MediaStream} stream — the getUserMedia stream to calibrate from.
   * @returns {Promise<number>} — the computed noise floor RMS.
   */
  function calibrateNoiseFloor(stream) {
    return new Promise((resolve, reject) => {
      if (!stream || stream.getAudioTracks().length === 0) {
        console.warn('[PatientML] calibrateNoiseFloor: no audio tracks.');
        noiseFloor = 0;
        resolve(0);
        return;
      }

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const calibCtx = new AudioCtx({ sampleRate: 16000 });
      const calibSource = calibCtx.createMediaStreamSource(stream);
      const calibProcessor = calibCtx.createScriptProcessor(4096, 1, 1);
      calibSource.connect(calibProcessor);
      calibProcessor.connect(calibCtx.destination);

      const calibSamples = [];
      calibProcessor.onaudioprocess = (evt) => {
        const data = evt.inputBuffer.getChannelData(0);
        calibSamples.push(new Float32Array(data));
      };

      console.log('[PatientML] Noise floor calibration started (5 seconds)…');

      setTimeout(() => {
        // Tear down temporary audio pipeline
        calibProcessor.onaudioprocess = null;
        calibProcessor.disconnect();
        calibSource.disconnect();
        try { calibCtx.close(); } catch (_) {}

        // Compute global RMS across all collected samples
        let sumSq = 0;
        let totalLen = 0;
        for (let i = 0; i < calibSamples.length; i++) {
          for (let j = 0; j < calibSamples[i].length; j++) {
            sumSq += calibSamples[i][j] * calibSamples[i][j];
          }
          totalLen += calibSamples[i].length;
        }

        if (totalLen > 0) {
          noiseFloor = Math.sqrt(sumSq / totalLen);
        } else {
          noiseFloor = 0;
        }

        console.log('[PatientML] Noise floor calibrated: RMS =', noiseFloor.toFixed(6),
          '| gate threshold =', (NOISE_GATE_MULTIPLIER * noiseFloor).toFixed(6));
        resolve(noiseFloor);
      }, 5000);
    });
  }

  // ============================================================
  // PUBLIC CONTROL API
  // ============================================================

  function startPatientExtraction(options = {}) {
    if (isExtractionActive) return;
    isExtractionActive = true;
    isAwaitingResponse = false;

    const targetUrl = options.wsUrl || WS_ENDPOINT;
    connectWebSocket(targetUrl);

    // Initialize speech recognition
    initSpeechRecognition();

    // Hook audio if stream provided or already on localVideo
    if (options.stream) {
      hookAudioStream(options.stream);
    } else {
      const video = document.getElementById('localVideo');
      if (video && video.srcObject) {
        hookAudioStream(video.srcObject);
      }
    }

    // Start 1.5s periodic sampling loop
    if (sampleTimer) clearInterval(sampleTimer);
    sampleTimer = setInterval(dispatchExtractionCycle, SAMPLE_INTERVAL_MS);

    console.log('[PatientML] Patient data extraction started (interval: 1.5s).');
  }

  function stopPatientExtraction() {
    isExtractionActive = false;
    isAwaitingResponse = false;

    if (sampleTimer) {
      clearInterval(sampleTimer);
      sampleTimer = null;
    }

    if (speechRecognizer) {
      try {
        speechRecognizer.abort();
      } catch (e) {}
      speechRecognizer = null;
    }

    if (scriptProcessor) {
      scriptProcessor.disconnect();
      scriptProcessor = null;
    }

    if (audioInputSource) {
      audioInputSource.disconnect();
      audioInputSource = null;
    }

    if (audioContext && audioContext.state !== 'closed') {
      try {
        audioContext.close();
      } catch (e) {}
      audioContext = null;
    }

    audioHookedStream = null;
    pcmChunks = [];
    completedSentences.length = 0;
    interimTranscript = '';
    noiseFloor = 0;

    if (ws) {
      ws.close();
      ws = null;
    }

    console.log('[PatientML] Patient data extraction stopped.');
  }

  // ============================================================
  // AUTO-HOOK & GLOBAL EXPORTS
  // ============================================================

  global.PatientMLExtractor = {
    start: startPatientExtraction,
    stop: stopPatientExtraction,
    hookAudioStream: hookAudioStream,
    calibrateNoiseFloor: calibrateNoiseFloor,
    get isAwaitingResponse() { return isAwaitingResponse; },
    get isRunning() { return isExtractionActive; },
    get currentNoiseFloor() { return noiseFloor; },
  };

  // Auto-start when DOM is loaded and localVideo starts streaming if desired
  document.addEventListener('DOMContentLoaded', () => {
    setupHiddenCanvas();

    const video = document.getElementById('localVideo');
    if (video) {
      video.addEventListener('playing', () => {
        if (video.srcObject && !isExtractionActive) {
          startPatientExtraction({ stream: video.srcObject });
        }
      });
    }
  });

})(typeof window !== 'undefined' ? window : this);

