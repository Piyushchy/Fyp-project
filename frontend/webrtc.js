/**
 * webrtc.js — P2P Video Call via simple-peer + FastAPI Signaling
 * ================================================================
 *
 * Establishes a Google-Meet-style 1:1 video call between two browsers
 * using WebRTC (via the simple-peer library) with signaling relayed
 * through a lightweight FastAPI WebSocket at /ws/signal/{room_id}.
 *
 * This module is strictly focused on the P2P connection lifecycle.
 * It does NOT contain any ML / emotion-recognition logic.
 *
 * Usage:
 *   <script src="https://unpkg.com/simple-peer@9/simplepeer.min.js"></script>
 *   <script src="webrtc.js"></script>
 */

// ============================================================
// CONFIG
// ============================================================

const WS_SIGNAL_PROTOCOL = location.protocol === 'https:' ? 'wss:' : 'ws:';

/** Default STUN/TURN servers for NAT traversal. */
const ICE_SERVERS = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
  { urls: 'stun:stun2.l.google.com:19302' },
];

// ============================================================
// STATE
// ============================================================

let localStream = null;
let peer = null;
let signalingSocket = null;
let currentRoomId = null;
let isInitiator = false;

// DOM references (set once the page loads)
let $localVideo = null;
let $remoteVideo = null;
let $statusText = null;

// ============================================================
// MEDIA — getUserMedia
// ============================================================

/**
 * Request camera + microphone permissions and attach the local
 * MediaStream to the <video id="localVideo"> element.
 */
async function acquireLocalMedia() {
  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user',
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
  } catch (err) {
    console.warn('Camera+Mic failed, trying video-only:', err.message);
    localStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  }

  if ($localVideo) {
    $localVideo.srcObject = localStream;
    $localVideo.muted = true;        // prevent local echo
    await $localVideo.play();
  }

  return localStream;
}

/**
 * Stop all tracks on the local stream and clear the video element.
 */
function releaseLocalMedia() {
  if (localStream) {
    localStream.getTracks().forEach(t => t.stop());
    localStream = null;
  }
  if ($localVideo) $localVideo.srcObject = null;
  if ($remoteVideo) $remoteVideo.srcObject = null;
}

// ============================================================
// SIGNALING — FastAPI WebSocket
// ============================================================

/**
 * Open the signaling WebSocket to /ws/signal/{roomId}.
 *
 * The backend simply relays JSON messages between the two peers in
 * the same room.  Messages carry a `type` field:
 *
 *   "signal"   — a simple-peer signaling blob (offer / answer / ICE)
 *   "ready"    — the backend tells us whether we are the initiator
 *   "peer_left" — the other side disconnected
 */
function connectSignaling(roomId) {
  currentRoomId = roomId;
  const url = `${WS_SIGNAL_PROTOCOL}//${location.host}/ws/signal/${roomId}`;
  signalingSocket = new WebSocket(url);

  signalingSocket.onopen = () => {
    console.log('[signal] connected to room', roomId);
    updateStatus('Waiting for peer…');
  };

  signalingSocket.onmessage = (evt) => {
    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }

    switch (msg.type) {
      case 'ready':
        // The server tells us our role
        isInitiator = msg.initiator;
        console.log('[signal] role:', isInitiator ? 'initiator' : 'joiner');
        createPeer(isInitiator);
        break;

      case 'signal':
        // Relay the signaling data into simple-peer
        if (peer && !peer.destroyed) {
          peer.signal(msg.data);
        }
        break;

      case 'peer_left':
        console.log('[signal] peer left the room');
        updateStatus('Peer disconnected');
        destroyPeer();
        break;

      default:
        console.warn('[signal] unknown message type:', msg.type);
    }
  };

  signalingSocket.onclose = () => {
    console.log('[signal] socket closed');
    updateStatus('Disconnected from signaling');
  };

  signalingSocket.onerror = (err) => {
    console.error('[signal] socket error', err);
  };
}

/**
 * Send a JSON message through the signaling socket.
 */
function sendSignal(data) {
  if (signalingSocket && signalingSocket.readyState === WebSocket.OPEN) {
    signalingSocket.send(JSON.stringify(data));
  }
}

/**
 * Close the signaling connection.
 */
function disconnectSignaling() {
  if (signalingSocket) {
    signalingSocket.close();
    signalingSocket = null;
  }
}

// ============================================================
// SIMPLE-PEER — WebRTC wrapper
// ============================================================

/**
 * Create a simple-peer instance.
 *
 * @param {boolean} initiator — true if this browser should send the
 *   initial offer.  The signaling server assigns this based on join
 *   order (first peer = initiator).
 */
function createPeer(initiator) {
  if (peer) destroyPeer();

  peer = new SimplePeer({
    initiator,
    stream: localStream || undefined,
    trickle: true,
    config: { iceServers: ICE_SERVERS },
  });

  // ── Signaling data (offer / answer / ICE candidates) ──
  peer.on('signal', (data) => {
    sendSignal({ type: 'signal', data });
  });

  // ── Remote stream arrives ──
  peer.on('stream', (remoteStream) => {
    console.log('[peer] received remote stream');
    if ($remoteVideo) {
      $remoteVideo.srcObject = remoteStream;
      $remoteVideo.play().catch(() => {});
    }
    updateStatus('Connected ✓');
  });

  // ── Connection established ──
  peer.on('connect', () => {
    console.log('[peer] data channel open');
    updateStatus('Connected ✓');
  });

  // ── Errors ──
  peer.on('error', (err) => {
    console.error('[peer] error:', err.message);
    updateStatus('Connection error — retrying…');
  });

  // ── Peer closed ──
  peer.on('close', () => {
    console.log('[peer] connection closed');
    if ($remoteVideo) $remoteVideo.srcObject = null;
    updateStatus('Call ended');
  });
}

/**
 * Tear down the current peer connection.
 */
function destroyPeer() {
  if (peer) {
    peer.destroy();
    peer = null;
  }
  if ($remoteVideo) $remoteVideo.srcObject = null;
}

// ============================================================
// PUBLIC API — called from the HTML page
// ============================================================

/**
 * Join (or create) a room and start the video call.
 *
 * @param {string} roomId — shared room identifier
 */
async function joinCall(roomId) {
  if (!roomId) {
    roomId = 'default-room';
  }

  updateStatus('Acquiring camera…');
  await acquireLocalMedia();

  updateStatus('Connecting to signaling…');
  connectSignaling(roomId);
}

/**
 * Leave the call, release media, tear down everything.
 */
function leaveCall() {
  destroyPeer();
  disconnectSignaling();
  releaseLocalMedia();
  updateStatus('Call ended');
}

/**
 * Toggle the local video track on/off (camera mute).
 * @returns {boolean} new enabled state
 */
function toggleVideo() {
  if (!localStream) return false;
  const track = localStream.getVideoTracks()[0];
  if (track) {
    track.enabled = !track.enabled;
    return track.enabled;
  }
  return false;
}

/**
 * Toggle the local audio track on/off (mic mute).
 * @returns {boolean} new enabled state
 */
function toggleAudio() {
  if (!localStream) return false;
  const track = localStream.getAudioTracks()[0];
  if (track) {
    track.enabled = !track.enabled;
    return track.enabled;
  }
  return false;
}

// ============================================================
// HELPERS
// ============================================================

function updateStatus(text) {
  if ($statusText) $statusText.textContent = text;
  console.log('[webrtc]', text);
}

// ============================================================
// INIT — bind DOM references when the page loads
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
  $localVideo  = document.getElementById('localVideo');
  $remoteVideo = document.getElementById('remoteVideo');
  $statusText  = document.getElementById('callStatus');
});

