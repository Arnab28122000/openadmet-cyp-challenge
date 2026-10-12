/* Motion for the explainer. The HTML already holds every figure's final state;
   this file only animates toward it. With reduced motion it does nothing visual. */
(function () {
  "use strict";
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var clamp = function (v, a, b) { return Math.max(a, Math.min(b, v)); };
  var ease = function (t) { return t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
  var fmt = function (v, n) { return v.toFixed(n); };

  /* ---------- final states: used for reduced motion and as the resting state */
  if (reduce || !("IntersectionObserver" in window)) {
    $$(".reveal").forEach(function (el) { el.classList.add("is-in"); });
    $$("#fig-bracket .layer").forEach(function (el) { el.classList.add("on"); });
    $$(".step").forEach(function (el) { el.classList.add("active"); });
    document.documentElement.classList.add("rm");
    progressBar();
    return;
  }

  /* ---------- reading progress */
  function progressBar() {
    var bar = $(".progress span");
    if (!bar) return;
    var tick = function () {
      var h = document.documentElement.scrollHeight - innerHeight;
      bar.style.setProperty("--p", h > 0 ? clamp(scrollY / h, 0, 1).toFixed(4) : 0);
    };
    addEventListener("scroll", tick, { passive: true });
    addEventListener("resize", tick);
    tick();
  }
  progressBar();

  /* ---------- reveal-once figures */
  var onEnter = {};
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      e.target.classList.add("is-in");
      io.unobserve(e.target);
      if (onEnter[e.target.id]) onEnter[e.target.id](e.target);
    });
  }, { threshold: 0.35 });
  $$(".reveal").forEach(function (el) { io.observe(el); });

  /* ---------- metric: predictions travel from the mean to the model */
  (function () {
    var svg = $("#fig-metric");
    if (!svg) return;
    var mean = +svg.dataset.mean, x0 = +svg.dataset.x0, x1 = +svg.dataset.x1;
    var sx = function (v) { return x0 + (v - 2.5) / 5 * (x1 - x0); };
    var rows = $$(".m-row", svg).map(function (g) {
      return { lo: +g.dataset.lo, hi: +g.dataset.hi, p: +g.dataset.p,
               dot: $(".m-pred", g), err: $(".m-err", g) };
    });
    var sumEl = $(".m-sum", svg), ratioEl = $(".m-ratio", svg);
    var err = function (p, lo, hi) { return Math.max(lo - p, 0) + Math.max(p - hi, 0); };
    var base = rows.reduce(function (s, r) { return s + err(mean, r.lo, r.hi); }, 0);
    function draw(t) {
      var tot = 0;
      rows.forEach(function (r) {
        var p = mean + (r.p - mean) * t, e = err(p, r.lo, r.hi);
        tot += e;
        r.dot.setAttribute("cx", fmt(sx(p), 1));
        var a = p < r.lo ? p : p > r.hi ? r.hi : p, b = p < r.lo ? r.lo : p;
        r.err.setAttribute("x1", fmt(sx(a), 1));
        r.err.setAttribute("x2", fmt(sx(b), 1));
      });
      sumEl.textContent = fmt(tot, 2);
      ratioEl.textContent = fmt(tot / base, 2);
    }
    draw(0);
    onEnter["fig-metric"] = function () {
      var start = null, hold = 700, dur = 1800;
      requestAnimationFrame(function step(ts) {
        if (start === null) start = ts;
        var t = clamp((ts - start - hold) / dur, 0, 1);
        draw(ease(t));
        if (t < 1) requestAnimationFrame(step);
      });
    };
  })();

  /* ---------- scroll-bound figures: progress through the viewport */
  var scrubs = [];
  function scrubProgress(el) {
    var r = el.getBoundingClientRect(), vh = innerHeight;
    // 0 when the figure's top reaches 90 % of the viewport, 1 when its centre passes 40 %
    return clamp((vh * 0.9 - r.top) / (vh * 0.5 + r.height * 0.5), 0, 1);
  }

  // hinge: the V's tip at the median opens into the [lo, hi] window
  (function () {
    var svg = $("#fig-hinge");
    if (!svg) return;
    var lo = +svg.dataset.lo, med = +svg.dataset.med, hi = +svg.dataset.hi;
    var X0 = 60, X1 = 620, Y0 = 262, Y1 = 40, P = 3.0;
    var hx = function (v) { return X0 + v / 6 * (X1 - X0); };
    var hy = function (v) { return Y0 - v / 2.5 * (Y0 - Y1); };
    var path = $(".hinge", svg), win = $(".window", svg), loss = $(".h-loss", svg),
        dot = $(".h-pred", svg), val = $(".h-val", svg),
        winLab = $(".fix-t", svg);
    scrubs.push(function () {
      var t = ease(scrubProgress(svg));
      var a = med - (med - lo) * t, b = med + (hi - med) * t;
      path.setAttribute("d", "M" + [[0, a], [a, 0], [b, 0], [6, 6 - b]]
        .map(function (q) { return fmt(hx(q[0]), 1) + "," + fmt(hy(q[1]), 1); }).join(" L"));
      win.setAttribute("x", fmt(hx(a), 1));
      win.setAttribute("width", fmt(hx(b) - hx(a), 1));
      winLab.style.opacity = clamp((t - .7) / .3, 0, 1);
      var l = Math.max(a - P, 0) + Math.max(P - b, 0);
      loss.setAttribute("y2", fmt(hy(l), 1));
      dot.setAttribute("cy", fmt(hy(l), 1));
      val.textContent = fmt(l, 3);
    });
  })();

  // affine: raw predictions shift and stretch onto the population's scale
  (function () {
    var svg = $("#fig-affine");
    if (!svg) return;
    var pc = +svg.dataset.pc, ps = +svg.dataset.ps, tc = +svg.dataset.tc, ts = +svg.dataset.ts;
    var AX0 = 40, AX1 = 620, AY0 = 200;
    var ax = function (v) { return AX0 + v / 8 * (AX1 - AX0); };
    var pred = $(".dens-pred", svg);
    scrubs.push(function () {
      var t = ease(scrubProgress(svg));
      var c = pc + (tc - pc) * t, s = ps + (ts - ps) * t, h = 150 * ps / s;
      var pts = [];
      for (var k = 0; k <= 120; k++) {
        var v = c - 3.2 * s + k * 6.4 * s / 120;
        pts.push(fmt(ax(v), 1) + "," + fmt(AY0 - h * Math.exp(-0.5 * Math.pow((v - c) / s, 2)), 1));
      }
      pred.setAttribute("d", "M" + pts.join(" L"));
    });
  })();

  var pending = false;
  function runScrubs() { pending = false; scrubs.forEach(function (f) { f(); }); }
  function requestScrub() { if (!pending) { pending = true; requestAnimationFrame(runScrubs); } }
  addEventListener("scroll", requestScrub, { passive: true });
  addEventListener("resize", requestScrub);
  runScrubs();

  /* ---------- bracket scrollytelling: the step crossing mid-screen sets the layers */
  (function () {
    var svg = $("#fig-bracket"), steps = $$(".step");
    if (!svg || !steps.length) return;
    var layers = $$(".layer", svg);
    function setStep(n) {
      steps.forEach(function (s) { s.classList.toggle("active", +s.dataset.step === n); });
      layers.forEach(function (l) { l.classList.toggle("on", +l.dataset.from <= n); });
    }
    setStep(1);
    function pick() {
      var mid = innerHeight * 0.6, n = 1;
      steps.forEach(function (s) { if (s.getBoundingClientRect().top < mid) n = +s.dataset.step; });
      setStep(n);
    }
    var queued = false;
    addEventListener("scroll", function () {
      if (!queued) { queued = true; requestAnimationFrame(function () { queued = false; pick(); }); }
    }, { passive: true });
    pick();
  })();
})();
