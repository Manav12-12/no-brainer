"use strict";

const canvas = document.querySelector("#scene");
const ctx = canvas.getContext("2d", { alpha: false });
const restartButton = document.querySelector("#restart");
const recordButton = document.querySelector("#record");
const fullscreenButton = document.querySelector("#fullscreen");
const statusText = document.querySelector("#status");

const W = canvas.width;
const H = canvas.height;
const EVENT_MS = 1750;
const FRAME_MS = 1000 / 30;
const palette = {
  cyan: "#4cecff",
  blue: "#557bff",
  violet: "#9f62ff",
  magenta: "#ff55c7",
  red: "#ff365f",
  amber: "#ffc857",
  green: "#55f6ac",
  pale: "#e9fcff",
  muted: "#648193",
  dark: "#030712",
};

let config = null;
let layout = new Map();
let eventSource = null;
let eventQueue = [];
let currentEvent = null;
let currentEventIndex = -1;
let eventStartedAt = 0;
let streamState = "connecting";
let streamMessage = "Connecting to live compute";
let streamComplete = false;
let lastFrameAt = 0;
let frameSamples = [];
let renderedFps = 30;
let recorder = null;
let recordedChunks = [];
let particleSeeds = [];
const glowSprites = new Map();

const backgroundLayer = makeLayer();
const anatomyLayer = makeLayer();
const wiringLayer = makeLayer();

function makeLayer() {
  const layer = document.createElement("canvas");
  layer.width = W;
  layer.height = H;
  return layer;
}

function seededNoise(value) {
  const x = Math.sin(value * 12.9898 + 78.233) * 43758.5453;
  return x - Math.floor(x);
}

function roundedRect(target, x, y, width, height, radius) {
  target.beginPath();
  target.roundRect(x, y, width, height, radius);
}

function write(target, value, x, y, size, color, align = "left", weight = 500) {
  target.save();
  target.font = `${weight} ${size}px "JetBrains Mono", monospace`;
  target.fillStyle = color;
  target.textAlign = align;
  target.textBaseline = "middle";
  target.fillText(value, x, y);
  target.restore();
}

function orb(target, x, y, radius, color, alpha = 1) {
  const quantizedRadius = Math.max(1, Math.round(radius));
  const key = `${color}:${quantizedRadius}`;
  let sprite = glowSprites.get(key);
  if (!sprite) {
    const extent = quantizedRadius * 3.2;
    const size = Math.ceil(extent * 2);
    sprite = document.createElement("canvas");
    sprite.width = size;
    sprite.height = size;
    const spriteContext = sprite.getContext("2d");
    const gradient = spriteContext.createRadialGradient(
      size / 2,
      size / 2,
      0,
      size / 2,
      size / 2,
      extent,
    );
    gradient.addColorStop(0, color);
    gradient.addColorStop(0.22, color);
    gradient.addColorStop(1, "rgba(0,0,0,0)");
    spriteContext.fillStyle = gradient;
    spriteContext.fillRect(0, 0, size, size);
    glowSprites.set(key, sprite);
  }
  target.save();
  target.globalAlpha = alpha;
  target.drawImage(sprite, x - sprite.width / 2, y - sprite.height / 2);
  target.restore();
}

function buildLayout(graph) {
  const groups = {};
  graph.nodes.forEach((node) => (groups[node.population] ||= []).push(node));
  const regions = {
    ORN: { cx: 960, cy: 255, rx: 370, ry: 72, depth: 0.8 },
    PN: { cx: 960, cy: 350, rx: 300, ry: 105, depth: 0.5 },
    KC: { cx: 960, cy: 460, rx: 360, ry: 150, depth: 0.2 },
    MBON: { cx: 960, cy: 565, rx: 175, ry: 62, depth: 0.1 },
    APL: { cx: 960, cy: 454, rx: 18, ry: 18, depth: 1 },
    descending: { cx: 960, cy: 660, rx: 92, ry: 28, depth: 0.4 },
  };
  const positions = new Map();
  Object.entries(groups).forEach(([name, nodes]) => {
    const region = regions[name];
    nodes.forEach((node, index) => {
      const angle = (Math.PI * 2 * index) / nodes.length - Math.PI / 2;
      const ring = 0.52 + 0.46 * seededNoise(node.id + 7);
      positions.set(node.id, {
        x: region.cx + Math.cos(angle) * region.rx * ring,
        y: region.cy + Math.sin(angle) * region.ry * ring,
        z: region.depth + seededNoise(node.id + 31) * 0.4,
        population: name,
      });
    });
  });
  return positions;
}

function drawBackgroundLayer() {
  const target = backgroundLayer.getContext("2d");
  const gradient = target.createRadialGradient(960, 450, 30, 960, 450, 1000);
  gradient.addColorStop(0, "#0c1832");
  gradient.addColorStop(0.43, "#050b1b");
  gradient.addColorStop(1, "#010207");
  target.fillStyle = gradient;
  target.fillRect(0, 0, W, H);

  for (let index = 0; index < 320; index += 1) {
    const x = seededNoise(index * 3 + 1) * W;
    const y = seededNoise(index * 7 + 2) * H;
    const size = 0.4 + seededNoise(index * 11) * 1.4;
    target.fillStyle = `rgba(83, 207, 255, ${0.05 + seededNoise(index) * 0.24})`;
    target.fillRect(x, y, size, size);
  }

  target.strokeStyle = "rgba(56, 185, 233, 0.032)";
  target.lineWidth = 1;
  for (let x = 0; x <= W; x += 64) {
    target.beginPath(); target.moveTo(x, 0); target.lineTo(x, H); target.stroke();
  }
  for (let y = 0; y <= H; y += 64) {
    target.beginPath(); target.moveTo(0, y); target.lineTo(W, y); target.stroke();
  }

  const vignette = target.createRadialGradient(960, 520, 380, 960, 520, 1070);
  vignette.addColorStop(0, "rgba(0,0,0,0)");
  vignette.addColorStop(1, "rgba(0,0,0,0.74)");
  target.fillStyle = vignette;
  target.fillRect(0, 0, W, H);
}

function drawBrainVolume(target) {
  target.save();
  const brain = target.createRadialGradient(960, 410, 40, 960, 430, 470);
  brain.addColorStop(0, "rgba(89, 47, 145, 0.34)");
  brain.addColorStop(0.54, "rgba(40, 79, 137, 0.24)");
  brain.addColorStop(1, "rgba(6, 22, 45, 0.08)");
  target.fillStyle = brain;
  target.strokeStyle = "rgba(82, 224, 255, 0.34)";
  target.lineWidth = 2;
  target.beginPath();
  target.moveTo(960, 142);
  target.bezierCurveTo(730, 105, 610, 185, 595, 340);
  target.bezierCurveTo(505, 365, 506, 510, 610, 550);
  target.bezierCurveTo(690, 640, 830, 615, 960, 690);
  target.bezierCurveTo(1090, 615, 1230, 640, 1310, 550);
  target.bezierCurveTo(1414, 510, 1415, 365, 1325, 340);
  target.bezierCurveTo(1310, 185, 1190, 105, 960, 142);
  target.closePath(); target.fill(); target.stroke();

  const optic = [[560, 390], [1360, 390]];
  optic.forEach(([x, y]) => {
    const glow = target.createRadialGradient(x, y, 10, x, y, 145);
    glow.addColorStop(0, "rgba(32, 217, 255, 0.18)");
    glow.addColorStop(1, "rgba(20, 90, 145, 0.02)");
    target.fillStyle = glow;
    target.strokeStyle = "rgba(66, 213, 255, 0.25)";
    target.beginPath(); target.ellipse(x, y, 118, 165, 0, 0, Math.PI * 2);
    target.fill(); target.stroke();
  });

  target.strokeStyle = "rgba(183, 103, 255, 0.34)";
  target.lineWidth = 18;
  target.lineCap = "round";
  target.beginPath(); target.moveTo(805, 285); target.bezierCurveTo(840, 370, 842, 480, 876, 555); target.stroke();
  target.beginPath(); target.moveTo(1115, 285); target.bezierCurveTo(1080, 370, 1078, 480, 1044, 555); target.stroke();
  target.lineWidth = 4;
  target.strokeStyle = "rgba(255, 94, 201, 0.44)";
  target.stroke();

  for (let ring = 0; ring < 4; ring += 1) {
    target.strokeStyle = `rgba(255, 193, 75, ${0.28 - ring * 0.045})`;
    target.lineWidth = 3;
    target.beginPath();
    target.ellipse(960, 435, 55 + ring * 17, 25 + ring * 8, 0, 0, Math.PI * 2);
    target.stroke();
  }

  for (let index = 0; index < 560; index += 1) {
    const angle = seededNoise(index * 2) * Math.PI * 2;
    const radius = Math.sqrt(seededNoise(index * 5 + 1));
    const x = 960 + Math.cos(angle) * radius * 390;
    const y = 420 + Math.sin(angle) * radius * 250;
    const color = index % 7 === 0 ? "255,86,197" : "73,220,255";
    target.fillStyle = `rgba(${color},${0.035 + seededNoise(index + 9) * 0.12})`;
    target.beginPath(); target.arc(x, y, 0.7 + seededNoise(index + 3) * 1.8, 0, Math.PI * 2); target.fill();
  }
  target.restore();
}

function drawVncAnatomy(target) {
  target.save();
  target.strokeStyle = "rgba(255, 191, 73, 0.45)";
  target.lineWidth = 3;
  target.beginPath(); target.moveTo(946, 665); target.bezierCurveTo(938, 760, 940, 865, 944, 1000); target.stroke();
  target.beginPath(); target.moveTo(974, 665); target.bezierCurveTo(982, 760, 980, 865, 976, 1000); target.stroke();
  for (let index = 0; index < 7; index += 1) {
    const y = 704 + index * 45;
    const width = 34 - index * 1.8;
    const glow = target.createRadialGradient(960, y, 2, 960, y, 42);
    glow.addColorStop(0, "rgba(255, 206, 92, 0.42)");
    glow.addColorStop(1, "rgba(255, 159, 50, 0)");
    target.fillStyle = glow;
    target.beginPath(); target.ellipse(960, y, width * 1.8, 26, 0, 0, Math.PI * 2); target.fill();
    target.fillStyle = "rgba(12, 24, 41, 0.92)";
    target.strokeStyle = "rgba(255, 198, 74, 0.55)";
    target.beginPath(); target.ellipse(960, y, width, 12, 0, 0, Math.PI * 2); target.fill(); target.stroke();
  }
  target.restore();
}

function drawAnatomyLayer() {
  const target = anatomyLayer.getContext("2d");
  drawBrainVolume(target);
  drawVncAnatomy(target);
  write(target, "OPTIC LOBE", 525, 585, 9, "rgba(104,152,177,0.72)", "center", 600);
  write(target, "OPTIC LOBE", 1395, 585, 9, "rgba(104,152,177,0.72)", "center", 600);
  write(target, "MUSHROOM BODY", 960, 592, 10, "rgba(151,118,197,0.8)", "center", 600);
  write(target, "CENTRAL COMPLEX", 960, 486, 9, "rgba(200,153,82,0.72)", "center", 600);
  write(target, "VENTRAL NERVE CORD", 960, 1030, 11, "rgba(225,177,76,0.82)", "center", 700);
}

function drawWiringLayer() {
  const target = wiringLayer.getContext("2d");
  config.graph.edges.forEach((edge) => {
    const source = layout.get(edge.source);
    const destination = layout.get(edge.target);
    const midX = (source.x + destination.x) / 2 + (source.z - destination.z) * 28;
    const midY = (source.y + destination.y) / 2 - 18;
    target.strokeStyle = edge.excitatory ? "rgba(70,218,255,0.13)" : "rgba(255,82,196,0.16)";
    target.lineWidth = 0.6 + edge.weight * 0.08;
    target.beginPath(); target.moveTo(source.x, source.y); target.quadraticCurveTo(midX, midY, destination.x, destination.y); target.stroke();
  });
  config.graph.nodes.forEach((node) => {
    const point = layout.get(node.id);
    const color = node.population === "APL" ? palette.magenta : node.population === "descending" ? palette.amber : palette.cyan;
    target.globalAlpha = 0.48;
    target.fillStyle = color;
    target.beginPath(); target.arc(point.x, point.y, 2.2 + point.z, 0, Math.PI * 2); target.fill();
  });
  target.globalAlpha = 1;
}

function prepareLayers() {
  drawBackgroundLayer();
  drawAnatomyLayer();
  drawWiringLayer();
}

function hostPosition(index) {
  const side = index < 6 ? -1 : 1;
  return { x: side < 0 ? 195 : 1725, y: 300 + (index % 6) * 105, side };
}

function drawHostNetwork(event, phase) {
  event.hosts.forEach((host, index) => {
    const position = hostPosition(index);
    const target = host.id === event.target_host;
    const color = host.isolated ? palette.amber : host.compromised ? palette.red : palette.green;
    const branchY = 730 + (index % 6) * 34;
    ctx.strokeStyle = host.isolated ? "rgba(255,199,80,0.25)" : "rgba(78,232,174,0.13)";
    ctx.lineWidth = target ? 2 : 1;
    ctx.beginPath(); ctx.moveTo(960, branchY); ctx.quadraticCurveTo(960 + position.side * 340, branchY, position.x - position.side * 76, position.y); ctx.stroke();

    roundedRect(ctx, position.x - 72, position.y - 27, 144, 54, 8);
    ctx.fillStyle = target ? "rgba(255,43,91,0.16)" : "rgba(5,14,28,0.86)";
    ctx.strokeStyle = color;
    ctx.lineWidth = target ? 2.5 : 1;
    ctx.fill(); ctx.stroke();
    if (target) orb(ctx, position.x, position.y, 10 + 4 * Math.sin(phase * Math.PI), color, 0.44);
    write(ctx, host.id.toUpperCase(), position.x, position.y - 7, 12, color, "center", 750);
    write(ctx, host.segment.toUpperCase(), position.x, position.y + 11, 8, palette.muted, "center", 500);
  });
}

function bezierPoint(start, control, end, progress) {
  const inverse = 1 - progress;
  return {
    x: inverse * inverse * start.x + 2 * inverse * progress * control.x + progress * progress * end.x,
    y: inverse * inverse * start.y + 2 * inverse * progress * control.y + progress * progress * end.y,
  };
}

function drawMovingSignal(start, control, end, progress, color, radius) {
  const clamped = Math.max(0, Math.min(1, progress));
  for (let tail = 8; tail >= 0; tail -= 1) {
    const point = bezierPoint(start, control, end, Math.max(0, clamped - tail * 0.018));
    orb(ctx, point.x, point.y, radius * (1 - tail * 0.07), color, 0.12 + (8 - tail) * 0.075);
  }
}

function beginEvent(event) {
  currentEvent = event;
  currentEventIndex = event.step;
  eventStartedAt = performance.now();
  const active = event.brain?.active_nodes || [];
  particleSeeds = active.flatMap((node, index) => {
    const outgoing = config.graph.edges.filter((edge) => edge.source === node.id);
    return outgoing.slice(0, 4).map((edge, edgeIndex) => ({
      edge,
      offset: seededNoise(node.id * 17 + edgeIndex * 3 + index),
      speed: 0.6 + seededNoise(node.id + edgeIndex) * 0.7,
    }));
  });
}

function drawNeuralActivity(event, phase, now) {
  const active = new Map((event.brain?.active_nodes || []).map((node) => [node.id, node.spikes]));
  config.graph.nodes.forEach((node) => {
    const point = layout.get(node.id);
    const spikes = active.get(node.id) || 0;
    if (spikes > 0) {
      const pulse = 1 + 0.22 * Math.sin(now * 0.012 + node.id);
      orb(ctx, point.x, point.y, (5 + Math.min(7, spikes)) * pulse, palette.pale, 0.9);
      ctx.fillStyle = palette.pale;
      ctx.beginPath(); ctx.arc(point.x, point.y, 3.8 + Math.min(4, spikes), 0, Math.PI * 2); ctx.fill();
    }
  });

  particleSeeds.forEach((particle) => {
    const source = layout.get(particle.edge.source);
    const destination = layout.get(particle.edge.target);
    const control = {
      x: (source.x + destination.x) / 2 + (source.z - destination.z) * 28,
      y: (source.y + destination.y) / 2 - 18,
    };
    const progress = (phase * particle.speed + particle.offset) % 1;
    const color = particle.edge.excitatory ? palette.cyan : palette.magenta;
    const point = bezierPoint(source, control, destination, progress);
    orb(ctx, point.x, point.y, 2.8, color, 0.8);
  });
}

function drawSignals(event, phase) {
  const targetIndex = event.hosts.findIndex((host) => host.id === event.target_host);
  const host = hostPosition(Math.max(0, targetIndex));
  if (phase < 0.38) {
    drawMovingSignal(
      { x: host.x, y: host.y },
      { x: 960 + host.side * 390, y: 170 },
      { x: 960, y: 205 },
      phase / 0.38,
      palette.red,
      7,
    );
  }
  if (event.reflex.ascend && phase >= 0.3 && phase < 0.82) {
    drawMovingSignal(
      { x: 960, y: 205 },
      { x: 840, y: 480 },
      { x: 960, y: 680 },
      (phase - 0.3) / 0.52,
      palette.magenta,
      6,
    );
  }
  if (event.action !== "no_op" && phase >= 0.7) {
    drawMovingSignal(
      { x: 960, y: 700 },
      { x: 960 + host.side * 420, y: 820 },
      { x: host.x, y: host.y },
      (phase - 0.7) / 0.3,
      palette.amber,
      8,
    );
  }
}

function panel(x, y, width, height, title) {
  roundedRect(ctx, x, y, width, height, 8);
  ctx.fillStyle = "rgba(2,8,19,0.8)";
  ctx.strokeStyle = "rgba(64,215,244,0.2)";
  ctx.lineWidth = 1;
  ctx.fill(); ctx.stroke();
  write(ctx, title, x + 16, y + 21, 10, palette.muted, "left", 700);
}

function drawHud(event, phase, now) {
  write(ctx, "DROSOPHILA SENTINEL", 54, 51, 27, palette.pale, "left", 800);
  write(ctx, "LIVE BIO-CYBER NEURAL COMPUTE", 55, 84, 11, palette.cyan, "left", 700);
  write(ctx, "STYLIZED FLY ANATOMY", 1864, 49, 10, palette.magenta, "right", 700);
  write(ctx, "48-NODE CONNECTOME PROXY", 1864, 72, 9, palette.muted, "right", 500);

  const livePulse = 0.55 + 0.45 * Math.sin(now * 0.006);
  orb(ctx, 1668, 106, 4, streamComplete ? palette.amber : palette.red, livePulse);
  write(ctx, streamComplete ? "RUN COMPLETE" : "LIVE COMPUTE", 1684, 106, 10, streamComplete ? palette.amber : palette.red, "left", 700);

  panel(48, 124, 350, 125, "THREAT TELEMETRY");
  const stageColor = event.stage === "contained" ? palette.green : palette.red;
  write(ctx, event.stage.replaceAll("_", " ").toUpperCase(), 67, 172, 23, stageColor, "left", 800);
  write(ctx, `${event.target_host.toUpperCase()}  /  ${event.segment.toUpperCase()}`, 67, 207, 11, palette.pale);
  write(ctx, `SIM STEP ${String(event.step + 1).padStart(2, "0")}  ·  ${event.compute_ms.toFixed(1)} MS`, 67, 231, 9, palette.muted);

  panel(1522, 124, 350, 166, "DECISION PATH");
  const pathText = event.reflex.ascend ? "BRAIN ESCALATION" : "REFLEX ACTION";
  write(ctx, pathText, 1540, 170, 19, event.reflex.ascend ? palette.magenta : palette.cyan, "left", 800);
  write(ctx, `CONFIDENCE ${(event.reflex.confidence * 100).toFixed(1)}%`, 1540, 207, 11, palette.pale);
  write(ctx, `MEASURED SPIKES ${String(event.brain?.total_spikes || 0).padStart(3, "0")}`, 1540, 233, 11, palette.pale);
  write(ctx, `ACTION ${event.action.toUpperCase()}`, 1540, 259, 11, event.action === "no_op" ? palette.muted : palette.amber);

  panel(48, 898, 350, 120, "STREAM");
  write(ctx, `EPISODE ${event.episode} / ${config.episodes}`, 67, 942, 12, palette.pale);
  write(ctx, `EVENT ${String(event.step + 1).padStart(2, "0")} / ${config.total_steps}`, 67, 970, 12, palette.pale);
  write(ctx, `${eventQueue.length} EVENT${eventQueue.length === 1 ? "" : "S"} BUFFERED`, 67, 998, 9, palette.muted);

  panel(1522, 898, 350, 120, "NEURAL READOUT");
  const activity = event.brain?.population_activity || {};
  write(ctx, `ORN INPUT  ${String(event.brain?.active_nodes?.length || 0).padStart(2, "0")} ACTIVE`, 1540, 940, 11, palette.pale);
  write(ctx, `KC ${Number(activity.KC || 0).toFixed(2)}   MBON ${Number(activity.MBON || 0).toFixed(2)}`, 1540, 967, 10, palette.pale);
  write(ctx, `RENDER ${renderedFps.toFixed(0)} FPS`, 1540, 996, 9, renderedFps < 24 ? palette.amber : palette.muted);

  if (event.action !== "no_op" && phase > 0.68) {
    write(ctx, "DEFENSIVE SIGNAL RELEASED", 960, 880, 18, palette.amber, "center", 800);
  } else if (event.reflex.ascend && phase > 0.32) {
    write(ctx, "UNCERTAIN PATTERN · BRIAN2 NETWORK RUNNING", 960, 880, 13, palette.magenta, "center", 700);
  }

  ctx.fillStyle = "rgba(69,104,129,0.22)";
  ctx.fillRect(440, 1041, 1040, 3);
  ctx.fillStyle = palette.cyan;
  ctx.fillRect(440, 1040, 1040 * ((event.step + phase) / config.total_steps), 5);
}

function drawWaiting(now) {
  ctx.drawImage(backgroundLayer, 0, 0);
  ctx.drawImage(anatomyLayer, 0, 0);
  const pulse = 0.5 + 0.5 * Math.sin(now * 0.004);
  orb(ctx, 960, 480, 24, palette.cyan, 0.25 + pulse * 0.25);
  write(ctx, "LIVE SIMULATION", 960, 455, 28, palette.pale, "center", 800);
  write(ctx, streamMessage.toUpperCase(), 960, 500, 12, palette.cyan, "center", 600);
  write(ctx, "THE NEXT FRAME APPEARS WHEN THE PYTHON SIMULATOR PRODUCES IT", 960, 532, 9, palette.muted, "center", 500);
}

function drawFrame(now) {
  if (now - lastFrameAt < FRAME_MS) {
    requestAnimationFrame(drawFrame);
    return;
  }
  const frameStarted = performance.now();
  lastFrameAt = now;
  if (!config || !currentEvent) {
    drawWaiting(now);
    requestAnimationFrame(drawFrame);
    return;
  }

  let phase = Math.min(1, (now - eventStartedAt) / EVENT_MS);
  if (phase >= 1 && eventQueue.length > 0) {
    beginEvent(eventQueue.shift());
    phase = 0;
  }

  ctx.drawImage(backgroundLayer, 0, 0);
  const driftX = Math.sin(now * 0.00022) * 3;
  const driftY = Math.cos(now * 0.00018) * 2;
  ctx.save();
  ctx.translate(driftX, driftY);
  ctx.drawImage(anatomyLayer, 0, 0);
  ctx.drawImage(wiringLayer, 0, 0);
  drawNeuralActivity(currentEvent, phase, now);
  ctx.restore();
  drawHostNetwork(currentEvent, phase);
  drawSignals(currentEvent, phase);
  drawHud(currentEvent, phase, now);

  const frameTime = performance.now() - frameStarted;
  frameSamples.push(frameTime);
  if (frameSamples.length > 45) frameSamples.shift();
  const average = frameSamples.reduce((sum, value) => sum + value, 0) / frameSamples.length;
  renderedFps = Math.min(30, 1000 / Math.max(FRAME_MS, average));
  requestAnimationFrame(drawFrame);
}

function handleStreamMessage(message) {
  const item = JSON.parse(message.data);
  if (item.type === "status") {
    streamState = item.state;
    streamMessage = item.message;
    statusText.textContent = item.message;
    return;
  }
  if (item.type === "simulation_event") {
    streamState = "running";
    streamMessage = "Receiving live Brian2 events";
    statusText.textContent = `Live · event ${item.step + 1}/${config.total_steps} · ${item.compute_ms.toFixed(1)} ms compute`;
    if (!currentEvent) beginEvent(item);
    else eventQueue.push(item);
    return;
  }
  if (item.type === "complete") {
    streamComplete = true;
    streamState = "complete";
    streamMessage = item.message;
    statusText.textContent = "Run complete · click New live run to restart";
    eventSource.close();
  }
}

function startLiveRun() {
  if (eventSource) eventSource.close();
  eventQueue = [];
  currentEvent = null;
  currentEventIndex = -1;
  streamComplete = false;
  streamState = "connecting";
  streamMessage = "Connecting to Python simulator";
  statusText.textContent = streamMessage;
  eventSource = new EventSource(`/api/stream?run=${Date.now()}`);
  eventSource.onmessage = handleStreamMessage;
  eventSource.onerror = () => {
    if (!streamComplete) {
      streamState = "error";
      streamMessage = "Live stream disconnected";
      statusText.textContent = streamMessage;
    }
  };
}

restartButton.addEventListener("click", startLiveRun);
fullscreenButton.addEventListener("click", () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.body.requestFullscreen();
});

recordButton.addEventListener("click", () => {
  if (recorder?.state === "recording") {
    recorder.stop();
    return;
  }
  const stream = canvas.captureStream(30);
  const formats = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  const mimeType = formats.find((format) => MediaRecorder.isTypeSupported(format));
  recordedChunks = [];
  recorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 10_000_000 } : undefined);
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size > 0) recordedChunks.push(event.data);
  });
  recorder.addEventListener("stop", () => {
    const blob = new Blob(recordedChunks, { type: recorder.mimeType || "video/webm" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "drosophila-sentinel-live.webm";
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    recordButton.textContent = "Record WebM";
    statusText.textContent = "Recording saved";
  });
  recorder.start(1000);
  startLiveRun();
  recordButton.textContent = "Stop + save";
  statusText.textContent = "● Recording new live run";
});

config = window.SENTINEL_LIVE_CONFIG;
if (config) {
  layout = buildLayout(config.graph);
  prepareLayers();
  startLiveRun();
} else {
  streamState = "error";
  streamMessage = "Simulation configuration unavailable";
  statusText.textContent = streamMessage;
}

requestAnimationFrame(drawFrame);
