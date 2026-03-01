// Custom v8 Canvas Background - Auto-inject
(function() {
  'use strict';

  // Create and inject canvas
  var canvas = document.createElement('canvas');
  canvas.id = 'bg-canvas';
  canvas.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;filter:blur(8px) contrast(1.05);';
  document.body.insertBefore(canvas, document.body.firstChild);

  var ctx = canvas.getContext('2d');
  var w, h, curves, t = 0;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }

  window.addEventListener('resize', resize);
  resize();

  curves = [];
  for (var i = 0; i < 5; i++) {
    curves.push({
      offset: i * 1.3,
      speed: 0.003 + i * 0.001,
      width: 2.5 + i * 0.8,
      hue: 210 + i * 12,
      sat: 70 + i * 5,
      glow: 40 + i * 15,
      amp: 0.15 + i * 0.06,
      freq: 0.6 + i * 0.15
    });
  }

  function drawCurve(c) {
    var points = [];
    var steps = 120;
    for (var i = 0; i <= steps; i++) {
      var pct = i / steps;
      var baseX = pct * w * 1.4 - w * 0.2;
      var wave1 = Math.sin(pct * Math.PI * c.freq + t * c.speed * 60 + c.offset) * h * c.amp;
      var wave2 = Math.cos(pct * Math.PI * c.freq * 0.7 + t * c.speed * 40 + c.offset * 1.5) * h * c.amp * 0.5;
      var wave3 = Math.sin(pct * Math.PI * 2 + t * c.speed * 20) * h * 0.03;
      var baseY = h * 0.35 + (pct - 0.5) * h * 0.3;
      points.push({ x: baseX, y: baseY + wave1 + wave2 + wave3 });
    }
    ctx.save();
    ctx.shadowColor = 'hsla(' + c.hue + ', ' + c.sat + '%, 60%, 0.4)';
    ctx.shadowBlur = c.glow * 1.6;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (var i = 1; i < points.length - 1; i++) {
      var mx = (points[i].x + points[i + 1].x) / 2;
      var my = (points[i].y + points[i + 1].y) / 2;
      ctx.quadraticCurveTo(points[i].x, points[i].y, mx, my);
    }
    var grad = ctx.createLinearGradient(0, 0, w, h);
    grad.addColorStop(0, 'hsla(' + c.hue + ', ' + c.sat + '%, 50%, 0.0)');
    grad.addColorStop(0.2, 'hsla(' + c.hue + ', ' + c.sat + '%, 55%, 0.7)');
    grad.addColorStop(0.5, 'hsla(' + (c.hue + 15) + ', ' + (c.sat + 10) + '%, 65%, 0.9)');
    grad.addColorStop(0.8, 'hsla(' + (c.hue - 10) + ', ' + c.sat + '%, 45%, 0.6)');
    grad.addColorStop(1, 'hsla(' + c.hue + ', ' + c.sat + '%, 40%, 0.0)');
    ctx.strokeStyle = grad;
    ctx.lineWidth = c.width;
    ctx.lineCap = 'round';
    ctx.stroke();
    ctx.shadowBlur = c.glow * 4;
    ctx.shadowColor = 'hsla(' + c.hue + ', ' + c.sat + '%, 70%, 0.2)';
    ctx.lineWidth = c.width * 0.3;
    ctx.strokeStyle = 'hsla(' + c.hue + ', 90%, 80%, 0.3)';
    ctx.stroke();
    ctx.restore();
  }

  function drawRefraction() {
    var rx = w * 0.65 + Math.sin(t * 0.008) * w * 0.05;
    var ry = h * 0.4 + Math.cos(t * 0.006) * h * 0.08;
    var rg = ctx.createRadialGradient(rx, ry, 0, rx, ry, h * 0.5);
    rg.addColorStop(0, 'hsla(30, 80%, 60%, 0.06)');
    rg.addColorStop(0.3, 'hsla(200, 70%, 50%, 0.03)');
    rg.addColorStop(0.7, 'hsla(220, 60%, 40%, 0.01)');
    rg.addColorStop(1, 'transparent');
    ctx.fillStyle = rg;
    ctx.fillRect(0, 0, w, h);
    var rx2 = w * 0.25 + Math.cos(t * 0.005) * w * 0.08;
    var ry2 = h * 0.6 + Math.sin(t * 0.007) * h * 0.06;
    var rg2 = ctx.createRadialGradient(rx2, ry2, 0, rx2, ry2, h * 0.4);
    rg2.addColorStop(0, 'hsla(210, 90%, 65%, 0.05)');
    rg2.addColorStop(0.5, 'hsla(230, 70%, 45%, 0.02)');
    rg2.addColorStop(1, 'transparent');
    ctx.fillStyle = rg2;
    ctx.fillRect(0, 0, w, h);
  }

  function drawSurface() {
    ctx.save();
    ctx.globalAlpha = 0.08;
    var surfacePoints = [];
    var steps = 80;
    for (var i = 0; i <= steps; i++) {
      var pct = i / steps;
      var x = pct * w;
      var y = h * 0.55 + Math.sin(pct * Math.PI * 1.2 + t * 0.01) * h * 0.12 + Math.cos(pct * Math.PI * 0.5 + t * 0.008) * h * 0.08;
      surfacePoints.push({ x: x, y: y });
    }
    ctx.beginPath();
    ctx.moveTo(surfacePoints[0].x, surfacePoints[0].y);
    for (var i = 1; i < surfacePoints.length - 1; i++) {
      var mx = (surfacePoints[i].x + surfacePoints[i + 1].x) / 2;
      var my = (surfacePoints[i].y + surfacePoints[i + 1].y) / 2;
      ctx.quadraticCurveTo(surfacePoints[i].x, surfacePoints[i].y, mx, my);
    }
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    var sg = ctx.createLinearGradient(0, h * 0.4, 0, h);
    sg.addColorStop(0, 'hsla(215, 80%, 45%, 1)');
    sg.addColorStop(0.4, 'hsla(220, 70%, 30%, 0.8)');
    sg.addColorStop(1, 'hsla(225, 60%, 10%, 0.3)');
    ctx.fillStyle = sg;
    ctx.fill();
    ctx.restore();
  }

  function draw() {
    t += 0.011;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#000509';
    ctx.fillRect(0, 0, w, h);
    drawSurface();
    drawRefraction();
    ctx.globalCompositeOperation = 'lighter';
    for (var i = 0; i < curves.length; i++) {
      drawCurve(curves[i]);
    }
    ctx.globalCompositeOperation = 'source-over';
    requestAnimationFrame(draw);
  }

  draw();
})();
