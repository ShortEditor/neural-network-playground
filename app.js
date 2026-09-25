/* Neural Network Playground - a real MLP with backprop in vanilla JS.
   No libraries. DOM code is guarded so the math core also runs in Node for tests. */
(function () {
"use strict";

// ---------- math core ----------

var ACTIVATIONS = {
  tanh:     { f: function (z) { return Math.tanh(z); },
              d: function (z) { var t = Math.tanh(z); return 1 - t * t; } },
  relu:     { f: function (z) { return z > 0 ? z : 0; },
              d: function (z) { return z > 0 ? 1 : 0; } },
  leakyrelu:{ f: function (z) { return z > 0 ? z : 0.1 * z; },
              d: function (z) { return z > 0 ? 1 : 0.1; } },
  sigmoid:  { f: function (z) { return 1 / (1 + Math.exp(-z)); },
              d: function (z) { var s = 1 / (1 + Math.exp(-z)); return s * (1 - s); } }
};

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

function randn() {
  var u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// sizes: [input, ...hidden, 1 output]. Returns {w, b} with w[l][j][i].
function buildNet(sizes, rng) {
  rng = rng || Math.random;
  var w = [], b = [];
  for (var l = 1; l < sizes.length; l++) {
    var scale = Math.sqrt(6 / (sizes[l - 1] + sizes[l]));
    var wl = [];
    for (var j = 0; j < sizes[l]; j++) {
      var row = [];
      for (var i = 0; i < sizes[l - 1]; i++) row.push((rng() * 2 - 1) * scale);
      wl.push(row);
    }
    w.push(wl);
    var bl = [];
    for (var j2 = 0; j2 < sizes[l]; j2++) bl.push(0);
    b.push(bl);
  }
  return { w: w, b: b, sizes: sizes };
}

// Forward pass. Returns {a: activations per layer, z: pre-activations per layer}.
function forward(net, actName, input) {
  var act = ACTIVATIONS[actName];
  var a = [input.slice()];
  var z = [];
  for (var l = 0; l < net.w.length; l++) {
    var isOut = l === net.w.length - 1;
    var zl = [], al = [];
    for (var j = 0; j < net.w[l].length; j++) {
      var s = net.b[l][j];
      for (var i = 0; i < a[l].length; i++) s += net.w[l][j][i] * a[l][i];
      zl.push(s);
      al.push(isOut ? sigmoid(s) : act.f(s));
    }
    z.push(zl);
    a.push(al);
  }
  return { a: a, z: z };
}

// One gradient step over a batch. Binary cross-entropy + sigmoid output.
function trainBatch(net, actName, batch, lr, reg, regRate) {
  var act = ACTIVATIONS[actName];
  var gW = [], gB = [];
  for (var l = 0; l < net.w.length; l++) {
    gW.push(net.w[l].map(function (row) { return row.map(function () { return 0; }); }));
    gB.push(net.b[l].map(function () { return 0; }));
  }
  for (var s = 0; s < batch.length; s++) {
    var sample = batch[s];
    var fw = forward(net, actName, sample.input);
    var L = net.w.length - 1;
    var delta = [[fw.a[fw.a.length - 1][0] - sample.label]];
    for (var l2 = L - 1; l2 >= 0; l2--) {
      var dl = [];
      for (var j = 0; j < net.w[l2].length; j++) {
        var down = 0;
        for (var k = 0; k < net.w[l2 + 1].length; k++) down += net.w[l2 + 1][k][j] * delta[0][k];
        dl.push(down * act.d(fw.z[l2][j]));
      }
      delta.unshift(dl);
    }
    for (var l3 = 0; l3 < net.w.length; l3++) {
      for (var j2 = 0; j2 < net.w[l3].length; j2++) {
        gB[l3][j2] += delta[l3][j2];
        for (var i2 = 0; i2 < net.w[l3][j2].length; i2++) {
          gW[l3][j2][i2] += delta[l3][j2] * fw.a[l3][i2];
        }
      }
    }
  }
  var n = batch.length;
  for (var l4 = 0; l4 < net.w.length; l4++) {
    for (var j3 = 0; j3 < net.w[l4].length; j3++) {
      net.b[l4][j3] -= lr * gB[l4][j3] / n;
      for (var i3 = 0; i3 < net.w[l4][j3].length; i3++) {
        var grad = gW[l4][j3][i3] / n;
        if (reg === "l2") grad += regRate * net.w[l4][j3][i3];
        else if (reg === "l1") grad += regRate * (net.w[l4][j3][i3] > 0 ? 1 : net.w[l4][j3][i3] < 0 ? -1 : 0);
        net.w[l4][j3][i3] -= lr * grad;
      }
    }
  }
}

function lossOn(net, actName, set) {
  if (!set.length) return 0;
  var total = 0;
  for (var s = 0; s < set.length; s++) {
    var out = forward(net, actName, set[s].input).a;
    var p = Math.min(1 - 1e-9, Math.max(1e-9, out[out.length - 1][0]));
    total += -(set[s].label * Math.log(p) + (1 - set[s].label) * Math.log(1 - p));
  }
  return total / set.length;
}

// ---------- data ----------

var FEATURES = {
  x1:   { label: "X₁",      f: function (p) { return p.x / 6; } },
  x2:   { label: "X₂",      f: function (p) { return p.y / 6; } },
  x1sq: { label: "X₁²",     f: function (p) { return (p.x * p.x) / 18; } },
  x2sq: { label: "X₂²",     f: function (p) { return (p.y * p.y) / 18; } },
  x1x2: { label: "X₁·X₂",   f: function (p) { return (p.x * p.y) / 18; } },
  sin1: { label: "sin(X₁)", f: function (p) { return Math.sin(p.x); } },
  sin2: { label: "sin(X₂)", f: function (p) { return Math.sin(p.y); } }
};

function genData(dataset, n, noise) {
  var pts = [], i, x, y, r, t, label;
  var jitter = function () { return randn() * (noise / 50) * 1.2; };
  for (i = 0; i < n; i++) {
    if (dataset === "circle") {
      x = Math.random() * 12 - 6; y = Math.random() * 12 - 6;
      label = (x * x + y * y < 6) ? 1 : 0;
      x += jitter(); y += jitter();
    } else if (dataset === "xor") {
      x = Math.random() * 12 - 6; y = Math.random() * 12 - 6;
      label = (x * y >= 0) ? 1 : 0;
      x += jitter(); y += jitter();
    } else if (dataset === "gauss") {
      label = i % 2;
      var cx = label ? 2 : -2, cy = label ? 2 : -2;
      x = cx + randn() * (0.9 + noise / 40);
      y = cy + randn() * (0.9 + noise / 40);
    } else if (dataset === "moons") {
      label = i % 2;
      t = Math.random() * Math.PI;
      if (label === 0) { x = 3 * Math.cos(t); y = 3 * Math.sin(t); }
      else { x = 3 - 3 * Math.cos(t); y = 2 - 3 * Math.sin(t); }
      x -= 1.5; y -= 1.2;
      x *= 1.5; y *= 1.5;
      x += jitter(); y += jitter();
    } else { // spiral
      label = i % 2;
      t = (i / (n / 2)) * 2.2 * Math.PI + Math.random() * 0.25;
      r = t * 0.85;
      var dir = label ? 1 : -1;
      x = dir * r * Math.cos(t) * 0.55;
      y = dir * r * Math.sin(t) * 0.55;
      x += jitter(); y += jitter();
    }
    pts.push({ x: x, y: y, label: label });
  }
  return pts;
}

function withFeatures(pts, featKeys) {
  return pts.map(function (p) {
    return { input: featKeys.map(function (k) { return FEATURES[k].f(p); }), label: p.label, x: p.x, y: p.y };
  });
}

// Export for Node tests.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { buildNet: buildNet, forward: forward, trainBatch: trainBatch, lossOn: lossOn, genData: genData, withFeatures: withFeatures };
}

// ---------- UI (browser only) ----------

if (typeof document === "undefined") return;

var DOMAIN = 6; // data plane is [-6, 6] on both axes

var state = {
  dataset: "circle",
  feats: ["x1", "x2"],
  hidden: [4, 2],
  activation: "tanh",
  lr: 0.03,
  reg: "none",
  regRate: 0.001,
  noise: 10,
  split: 50,
  batch: 10,
  epoch: 0,
  running: false,
  net: null,
  train: [],
  test: [],
  trainHist: [],
  testHist: [],
  probeActs: null
};

var $ = function (id) { return document.getElementById(id); };
var dataCanvas = $("data-canvas"), dctx = dataCanvas.getContext("2d");
var netCanvas = $("net-canvas"), nctx = netCanvas.getContext("2d");
var lossCanvas = $("loss-canvas"), lctx = lossCanvas.getContext("2d");
var boundary = document.createElement("canvas");
boundary.width = 120; boundary.height = 120;
var bctx = boundary.getContext("2d");
var nodeLayout = []; // {x, y, layer, index} for hover hit-testing

function rebuildNet(resetEpoch) {
  var sizes = [state.feats.length].concat(state.hidden, [1]);
  state.net = buildNet(sizes);
  state.probeActs = null;
  if (resetEpoch !== false) {
    state.epoch = 0;
    state.trainHist = [];
    state.testHist = [];
  }
}

function regenData() {
  var raw = genData(state.dataset, 300, state.noise);
  // shuffle
  for (var i = raw.length - 1; i > 0; i--) {
    var j = Math.floor(Math.random() * (i + 1));
    var t = raw[i]; raw[i] = raw[j]; raw[j] = t;
  }
  var nTrain = Math.round(raw.length * state.split / 100);
  state.train = withFeatures(raw.slice(0, nTrain), state.feats);
  state.test = withFeatures(raw.slice(nTrain), state.feats);
}

function trainEpoch() {
  var idx = state.train.map(function (_, i) { return i; });
  for (var i = idx.length - 1; i > 0; i--) {
    var j = Math.floor(Math.random() * (i + 1));
    var t = idx[i]; idx[i] = idx[j]; idx[j] = t;
  }
  for (var k = 0; k < idx.length; k += state.batch) {
    var batch = idx.slice(k, k + state.batch).map(function (m) { return state.train[m]; });
    trainBatch(state.net, state.activation, batch, state.lr, state.reg, state.regRate);
  }
  state.epoch++;
}

function refreshProbe() {
  // average activation per neuron over a sample of train points
  var probe = state.train.slice(0, 24);
  if (!probe.length) return;
  var L = state.net.sizes.length;
  var sums = [];
  for (var l = 0; l < L; l++) {
    sums.push(state.net.sizes[l] === undefined ? [] : new Array(state.net.sizes[l]).fill(0));
  }
  for (var s = 0; s < probe.length; s++) {
    var fw = forward(state.net, state.activation, probe[s].input);
    for (var l2 = 0; l2 < fw.a.length; l2++) {
      for (var j = 0; j < fw.a[l2].length; j++) sums[l2][j] += fw.a[l2][j] / probe.length;
    }
  }
  state.probeActs = sums;
}

// ---------- drawing ----------

function domainToPx(v, size) { return (v + DOMAIN) / (2 * DOMAIN) * size; }

function drawData() {
  var S = dataCanvas.width;
  // decision boundary
  var img = bctx.createImageData(boundary.width, boundary.height);
  var hasNet = !!state.net;
  for (var py = 0; py < boundary.height; py++) {
    for (var px = 0; px < boundary.width; px++) {
      var x = (px / boundary.width) * 2 * DOMAIN - DOMAIN;
      var y = DOMAIN - (py / boundary.height) * 2 * DOMAIN;
      var out = 0.5;
      if (hasNet) {
        var p = { x: x, y: y };
        var inp = state.feats.map(function (k) { return FEATURES[k].f(p); });
        var fw = forward(state.net, state.activation, inp);
        out = fw.a[fw.a.length - 1][0];
      }
      var o = (py * boundary.width + px) * 4;
      // blue (class 0) -> white -> orange (class 1)
      var rr, gg, bb;
      if (out > 0.5) { var t = (out - 0.5) * 2; rr = 255; gg = 255 - t * 119; bb = 255 - t * 193; }
      else { var t2 = (0.5 - out) * 2; rr = 255 - t2 * 167; gg = 255 - t2 * 89; bb = 255; }
      img.data[o] = rr; img.data[o + 1] = gg; img.data[o + 2] = bb; img.data[o + 3] = 170;
    }
  }
  bctx.putImageData(img, 0, 0);
  dctx.imageSmoothingEnabled = true;
  dctx.clearRect(0, 0, S, S);
  dctx.drawImage(boundary, 0, 0, S, S);
  // axes
  dctx.strokeStyle = "rgba(139,148,158,0.25)";
  dctx.lineWidth = 1;
  dctx.beginPath();
  dctx.moveTo(S / 2, 0); dctx.lineTo(S / 2, S);
  dctx.moveTo(0, S / 2); dctx.lineTo(S, S / 2);
  dctx.stroke();
  // points
  function drawPoint(p, isTest) {
    var cx = domainToPx(p.x, S), cy = S - domainToPx(p.y, S);
    dctx.beginPath();
    dctx.arc(cx, cy, isTest ? 4 : 4.5, 0, Math.PI * 2);
    if (isTest) {
      dctx.strokeStyle = p.label ? "#f0883e" : "#58a6ff";
      dctx.lineWidth = 2;
      dctx.stroke();
    } else {
      dctx.fillStyle = p.label ? "#f0883e" : "#58a6ff";
      dctx.fill();
      dctx.strokeStyle = "rgba(0,0,0,0.5)";
      dctx.lineWidth = 1;
      dctx.stroke();
    }
  }
  state.train.forEach(function (p) { drawPoint(p, false); });
  state.test.forEach(function (p) { drawPoint(p, true); });
}

function weightColor(wv, maxAbs) {
  var a = Math.min(1, Math.abs(wv) / (maxAbs || 1));
  return wv >= 0 ? "rgba(240,136,62," + (0.15 + 0.85 * a) + ")" : "rgba(88,166,255," + (0.15 + 0.85 * a) + ")";
}

function actColor(v, isInput) {
  // orange positive, blue negative, gray ~0
  var t = Math.max(-1, Math.min(1, v));
  if (isInput) t = Math.max(-1, Math.min(1, t * 2));
  if (t >= 0) {
    var a = t;
    return "rgb(" + Math.round(22 + a * 218) + "," + Math.round(27 + a * 109) + "," + Math.round(34 + a * 28) + ")";
  }
  var b = -t;
  return "rgb(" + Math.round(22 + b * 66) + "," + Math.round(27 + b * 139) + "," + Math.round(34 + b * 221) + ")";
}

function drawNet() {
  var W = netCanvas.width, H = netCanvas.height;
  nctx.clearRect(0, 0, W, H);
  if (!state.net) return;
  var sizes = state.net.sizes;
  var L = sizes.length;
  var maxAbs = 1e-6;
  state.net.w.forEach(function (wl) { wl.forEach(function (row) { row.forEach(function (v) { maxAbs = Math.max(maxAbs, Math.abs(v)); }); }); });

  // layout
  nodeLayout = [];
  var padX = 60, padY = 30;
  for (var l = 0; l < L; l++) {
    var n = sizes[l];
    var x = padX + (W - 2 * padX) * (L === 1 ? 0.5 : l / (L - 1));
    for (var j = 0; j < n; j++) {
      var y = padY + (H - 2 * padY) * (n === 1 ? 0.5 : (j + 0.5) / n) ;
      nodeLayout.push({ x: x, y: y, layer: l, index: j });
    }
  }
  function nodeAt(l, j) {
    for (var q = 0; q < nodeLayout.length; q++) if (nodeLayout[q].layer === l && nodeLayout[q].index === j) return nodeLayout[q];
    return null;
  }
  // edges
  for (var l2 = 0; l2 < state.net.w.length; l2++) {
    for (var j2 = 0; j2 < state.net.w[l2].length; j2++) {
      for (var i2 = 0; i2 < state.net.w[l2][j2].length; i2++) {
        var a = nodeAt(l2, i2), b = nodeAt(l2 + 1, j2);
        var wv = state.net.w[l2][j2][i2];
        nctx.strokeStyle = weightColor(wv, maxAbs);
        nctx.lineWidth = 1 + 3.5 * Math.min(1, Math.abs(wv) / maxAbs);
        nctx.beginPath();
        nctx.moveTo(a.x, a.y);
        nctx.lineTo(b.x, b.y);
        nctx.stroke();
      }
    }
  }
  // nodes
  var labels = state.feats.map(function (k) { return FEATURES[k].label; });
  for (var l3 = 0; l3 < L; l3++) {
    var isInput = l3 === 0, isOutput = l3 === L - 1;
    var r = isOutput ? 18 : 15;
    for (var j3 = 0; j3 < sizes[l3]; j3++) {
      var nd = nodeAt(l3, j3);
      var av = state.probeActs ? state.probeActs[l3][j3] : 0;
      nctx.beginPath();
      nctx.arc(nd.x, nd.y, r, 0, Math.PI * 2);
      nctx.fillStyle = actColor(av, isInput);
      nctx.fill();
      nctx.strokeStyle = "#2d333b";
      nctx.lineWidth = 1.5;
      nctx.stroke();
      if (isInput) {
        nctx.fillStyle = "#8b949e";
        nctx.font = "11px sans-serif";
        nctx.textAlign = "right";
        nctx.fillText(labels[j3] || ("in" + j3), nd.x - r - 6, nd.y + 4);
      }
      if (isOutput) {
        nctx.fillStyle = "#8b949e";
        nctx.font = "11px sans-serif";
        nctx.textAlign = "left";
        nctx.fillText("output", nd.x + r + 6, nd.y + 4);
      }
    }
    // +/- neuron controls on hidden layers
    if (!isInput && !isOutput) {
      var top = nodeAt(l3, 0);
      nctx.fillStyle = "#8b949e";
      nctx.font = "bold 13px sans-serif";
      nctx.textAlign = "center";
      nctx.fillText("+", top.x, 14);
      nctx.fillText("−", top.x, H - 4);
    }
  }
  // layer captions
  nctx.fillStyle = "#6e7681";
  nctx.font = "10px sans-serif";
  nctx.textAlign = "center";
  var inp0 = nodeAt(0, 0), out0 = nodeAt(L - 1, 0);
  nctx.fillText("inputs", inp0.x, H - 4);
  nctx.fillText("prediction", out0.x, H - 4);
}

function drawLoss() {
  var W = lossCanvas.width, H = lossCanvas.height;
  lctx.clearRect(0, 0, W, H);
  var hist = state.trainHist, hist2 = state.testHist;
  if (hist.length < 2) {
    lctx.fillStyle = "#6e7681";
    lctx.font = "11px sans-serif";
    lctx.textAlign = "center";
    lctx.fillText("loss over epochs - press Train", W / 2, H / 2);
    return;
  }
  var max = 0;
  hist.concat(hist2).forEach(function (v) { max = Math.max(max, v); });
  max = Math.max(max, 0.1);
  lctx.strokeStyle = "rgba(139,148,158,0.15)";
  lctx.beginPath();
  for (var g = 1; g < 4; g++) { lctx.moveTo(0, H * g / 4); lctx.lineTo(W, H * g / 4); }
  lctx.stroke();
  function plot(arr, color) {
    lctx.strokeStyle = color;
    lctx.lineWidth = 1.8;
    lctx.beginPath();
    for (var i = 0; i < arr.length; i++) {
      var x = i / (arr.length - 1) * (W - 6) + 3;
      var y = H - 4 - (arr[i] / max) * (H - 10);
      if (i === 0) lctx.moveTo(x, y); else lctx.lineTo(x, y);
    }
    lctx.stroke();
  }
  plot(hist, "#f0883e");
  plot(hist2, "#58a6ff");
  lctx.fillStyle = "#f0883e"; lctx.font = "10px sans-serif"; lctx.textAlign = "left";
  lctx.fillText("train", 6, 12);
  lctx.fillStyle = "#58a6ff";
  lctx.fillText("test", 44, 12);
}

function updateStats() {
  $("epoch").textContent = state.epoch;
  var tl = state.trainHist.length ? state.trainHist[state.trainHist.length - 1] : null;
  var sl = state.testHist.length ? state.testHist[state.testHist.length - 1] : null;
  $("train-loss").textContent = tl === null ? "–" : tl.toFixed(3);
  $("test-loss").textContent = sl === null ? "–" : sl.toFixed(3);
}

function drawAll() {
  drawData();
  drawNet();
  drawLoss();
  updateStats();
}

// ---------- main loop ----------

var frame = 0;
function tick() {
  if (state.running) {
    for (var i = 0; i < 3; i++) trainEpoch();
    state.trainHist.push(lossOn(state.net, state.activation, state.train));
    state.testHist.push(lossOn(state.net, state.activation, state.test));
    if (state.trainHist.length > 400) { state.trainHist.shift(); state.testHist.shift(); }
    refreshProbe();
    drawAll();
  }
  requestAnimationFrame(tick);
}

// ---------- events ----------

function setPlayLabel() {
  $("btn-play").textContent = state.running ? "⏸ Pause" : "▶ Train";
}

$("btn-play").addEventListener("click", function () {
  state.running = !state.running;
  setPlayLabel();
});
$("btn-step").addEventListener("click", function () {
  trainEpoch();
  state.trainHist.push(lossOn(state.net, state.activation, state.train));
  state.testHist.push(lossOn(state.net, state.activation, state.test));
  refreshProbe();
  drawAll();
});
$("btn-reset").addEventListener("click", function () {
  rebuildNet(true);
  refreshProbe();
  drawAll();
});

document.querySelectorAll(".ds").forEach(function (btn) {
  btn.addEventListener("click", function () {
    document.querySelectorAll(".ds").forEach(function (b) { b.classList.remove("active"); });
    btn.classList.add("active");
    state.dataset = btn.dataset.ds;
    regenData();
    refreshProbe();
    drawAll();
  });
});

document.querySelectorAll(".feat").forEach(function (btn) {
  btn.addEventListener("click", function () {
    var k = btn.dataset.f;
    var i = state.feats.indexOf(k);
    if (i >= 0 && state.feats.length > 1) { state.feats.splice(i, 1); btn.classList.remove("active"); }
    else if (i < 0) { state.feats.push(k); btn.classList.add("active"); }
    regenData();
    rebuildNet(true);
    refreshProbe();
    drawAll();
  });
});

function sliderLog(v, minExp, maxExp) { return Math.pow(10, minExp + (v / 100) * (maxExp - minExp)); }

$("lr").addEventListener("input", function () {
  state.lr = sliderLog(+this.value, -4, 0);
  $("lr-v").textContent = state.lr >= 0.01 ? state.lr.toFixed(3) : state.lr.toExponential(0);
});
$("activation").addEventListener("change", function () {
  state.activation = this.value;
  rebuildNet(true);
  refreshProbe();
  drawAll();
});
$("reg").addEventListener("change", function () { state.reg = this.value; });
$("reg-rate").addEventListener("input", function () {
  state.regRate = sliderLog(+this.value, -5, -1);
  $("reg-v").textContent = state.regRate >= 0.001 ? state.regRate.toFixed(3) : state.regRate.toExponential(0);
});
$("noise").addEventListener("input", function () {
  state.noise = +this.value;
  $("noise-v").textContent = this.value;
  regenData(); refreshProbe(); drawAll();
});
$("split").addEventListener("input", function () {
  state.split = +this.value;
  $("split-v").textContent = this.value + "%";
  regenData(); refreshProbe(); drawAll();
});
$("batch").addEventListener("input", function () {
  state.batch = +this.value;
  $("batch-v").textContent = this.value;
});

$("btn-add-layer").addEventListener("click", function () {
  if (state.hidden.length >= 4) return;
  state.hidden.push(2);
  rebuildNet(true);
  refreshProbe();
  drawAll();
});

// +/- neurons and hover tooltips on the network canvas
netCanvas.addEventListener("click", function (e) {
  var rect = netCanvas.getBoundingClientRect();
  var sx = netCanvas.width / rect.width, sy = netCanvas.height / rect.height;
  var mx = (e.clientX - rect.left) * sx, my = (e.clientY - rect.top) * sy;
  var L = state.net.sizes.length;
  for (var l = 1; l < L - 1; l++) {
    var nd = null;
    for (var q = 0; q < nodeLayout.length; q++) if (nodeLayout[q].layer === l && nodeLayout[q].index === 0) nd = nodeLayout[q];
    if (!nd) continue;
    if (Math.abs(mx - nd.x) < 14 && my < 24) { // plus
      if (state.hidden[l - 1] < 8) state.hidden[l - 1]++;
      rebuildNet(true); refreshProbe(); drawAll(); return;
    }
    if (Math.abs(mx - nd.x) < 14 && my > netCanvas.height - 24) { // minus
      if (state.hidden[l - 1] > 1) state.hidden[l - 1]--;
      else if (state.hidden.length > 1) state.hidden.splice(l - 1, 1);
      rebuildNet(true); refreshProbe(); drawAll(); return;
    }
  }
});

netCanvas.addEventListener("mousemove", function (e) {
  var rect = netCanvas.getBoundingClientRect();
  var sx = netCanvas.width / rect.width, sy = netCanvas.height / rect.height;
  var mx = (e.clientX - rect.left) * sx, my = (e.clientY - rect.top) * sy;
  var tip = $("neuron-tip");
  for (var q = 0; q < nodeLayout.length; q++) {
    var nd = nodeLayout[q];
    if ((mx - nd.x) * (mx - nd.x) + (my - nd.y) * (my - nd.y) < 20 * 20) {
      var L = state.net.sizes.length;
      var name = nd.layer === 0 ? "Input · " + (state.feats.map(function (k) { return FEATURES[k].label; })[nd.index] || "in" + nd.index)
        : nd.layer === L - 1 ? "Output (sigmoid)"
        : "Hidden layer " + nd.layer + " · neuron " + (nd.index + 1);
      var av = state.probeActs ? state.probeActs[nd.layer][nd.index] : 0;
      var txt = name + "\navg activation: " + av.toFixed(3);
      if (nd.layer > 0) txt += "\nbias: " + state.net.b[nd.layer - 1][nd.index].toFixed(3);
      tip.textContent = txt;
      tip.classList.remove("hidden");
      tip.style.left = Math.min(rect.width - 150, e.clientX - rect.left + 16) + "px";
      tip.style.top = (e.clientY - rect.top + 12) + "px";
      return;
    }
  }
  tip.classList.add("hidden");
});
netCanvas.addEventListener("mouseleave", function () { $("neuron-tip").classList.add("hidden"); });

// ---------- init ----------

// defaults from the HTML slider positions
state.lr = sliderLog(+$("lr").value, -4, 0);
$("lr-v").textContent = state.lr.toFixed(3);
state.regRate = sliderLog(+$("reg-rate").value, -5, -1);
$("reg-v").textContent = state.regRate.toFixed(3);

regenData();
rebuildNet(true);
refreshProbe();
drawAll();
setPlayLabel();
requestAnimationFrame(tick);

})();
