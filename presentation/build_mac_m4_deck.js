// Builds presentation/Evol_inference_mac-m4.pptx (12 slides) from the results recorded in
// Doc/implementation_plan_mac-m4.md. Run:
//   NODE_PATH=<dir containing node_modules with pptxgenjs, react, react-dom, react-icons, sharp> \
//     node presentation/build_mac_m4_deck.js
// Initial version: numbers are taken from the plan's results sections (T0-T2). Speed/energy figures
// from the dry-run searches are SIMULATED and are labelled as such on the slides that show them.
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const path = require("path");
const { applyTheme } = require(process.env.PPTX_SKILL_DIR + "/scripts/apply_theme.js");

const OUT = path.join(__dirname, "Evol_inference_mac-m4.pptx");

const THEME = {
  name: "ArmPerf",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "10212B", lt1: "FFFFFF", dk2: "0B3C4F", lt2: "EDF2F5",
    accent1: "0A7EA4", accent2: "F2A541", accent3: "2E9E8A", accent4: "D1495B",
    accent5: "6C7A89", accent6: "7B6FB5", hlink: "0A7EA4", folHlink: "6C7A89",
  },
};
const H = THEME.colors; // hex values, for the few options that only take hex (charts, shadows)
const AMBER_TEXT = "9A5B00"; // amber darkened for text on light cards (accent2 itself is too pale there)

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "Evol_inference on Arm (mac-m4)";
  pres.author = "Nikolaus Almassy";
  const C = pres.SchemeColor;

  // ---------- layouts ----------
  pres.defineSlideMaster({
    title: "TITLE", background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 8.2, h: 1.7, fontFace: "+mj-lt", fontSize: 44, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.0, w: 8.2, h: 1.2, fontFace: "+mn-lt", fontSize: 20, color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: C.background1 },
    slideNumber: { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontFace: "Calibri", fontSize: 11, color: C.accent5, align: "right" },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontFace: "+mj-lt", fontSize: 28, bold: true, color: C.text2, align: "left", valign: "middle", margin: 0 }, text: "" } },
      { text: { text: "Evol_inference  |  Arm port (branch mac-m4)", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontFace: "Calibri", fontSize: 11, color: C.accent5, margin: 0 } } },
    ],
  });
  pres.defineSlideMaster({
    title: "CLOSE", background: { color: C.text2 },
    slideNumber: { x: 12.2, y: 7.0, w: 0.6, h: 0.3, fontFace: "Calibri", fontSize: 11, color: C.background2, align: "right" },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.95, fontFace: "+mj-lt", fontSize: 28, bold: true, color: C.background1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    ],
  });

  // ---------- icons ----------
  const I = {
    chip: await icon(fa.FaMicrochip, H.lt1), chipD: await icon(fa.FaMicrochip, H.dk2),
    cogs: await icon(fa.FaCogs, H.lt1), chart: await icon(fa.FaChartLine, H.lt1),
    bolt: await icon(fa.FaBolt, H.lt1), search: await icon(fa.FaSearch, H.lt1),
    code: await icon(fa.FaCode, H.lt1), users: await icon(fa.FaUsers, H.lt1),
    layers: await icon(fa.FaLayerGroup, H.lt1), check: await icon(fa.FaCheck, H.lt1),
    clock: await icon(fa.FaRegClock, H.lt1), warn: await icon(fa.FaExclamationTriangle, H.lt1),
  };

  // ---------- helpers ----------
  const FONT_CHART = "+mn-lt";
  const circleIcon = (s, x, y, d, data, fillColor, name) => {
    s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fillColor }, line: { color: fillColor, width: 0 }, objectName: name + " circle" });
    s.addImage({ data, x: x + d * 0.25, y: y + d * 0.25, w: d * 0.5, h: d * 0.5, objectName: name + " icon" });
  };
  const card = (s, x, y, w, h, name, fill = C.background2) =>
    s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: name });
  const stat = (s, x, y, w, big, label, color, name) => {
    s.addText(big, { x, y, w, h: 0.75, fontFace: "Cambria", fontSize: 40, bold: true, color, margin: 0, valign: "bottom", isTextBox: true, objectName: name + " value" });
    s.addText(label, { x, y: y + 0.78, w, h: 0.7, fontFace: "Calibri", fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: name + " label" });
  };
  const bullets = (items, o = {}) => items.map((t, i) => ({
    text: t, options: { bullet: { indent: 16 }, breakLine: i < items.length - 1, paraSpaceAfter: o.gap ?? 8, fontSize: o.size ?? 15, color: o.color ?? C.text1, fontFace: "Calibri" },
  }));
  const chartBase = (title) => ({
    showTitle: true, title, titleFontFace: FONT_CHART, titleFontSize: 14, titleColor: H.dk2,
    catAxisLabelFontFace: FONT_CHART, valAxisLabelFontFace: FONT_CHART, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    catAxisLabelColor: H.dk1, valAxisLabelColor: H.accent5,
    valGridLine: { color: "D5DEE4", size: 0.75 }, catGridLine: { style: "none" },
    dataLabelFontFace: FONT_CHART, dataLabelFontSize: 11, legendFontFace: FONT_CHART, legendFontSize: 12, legendColor: H.dk1,
  });
  const notes = (s, t) => s.addNotes(t);

  // ======================================================================
  pres.addSection({ title: "Context" });
  // 1. Title ------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Context" });
    s.addText("Evol_inference on Arm", { placeholder: "title" });
    s.addText("Multi-objective quantization search for LLM inference: accuracy, speed and power on Arm CPUs", { placeholder: "body" });
    s.addText("Branch mac-m4  |  all tiers complete: desktop, RTX 3080, aarch64 emulation, AWS Graviton3, Apple M4",
      { x: 0.8, y: 6.3, w: 9.5, h: 0.5, fontFace: "Calibri", fontSize: 14, color: C.accent2, margin: 0, isTextBox: true, objectName: "status line" });
    s.addShape(pres.ShapeType.ellipse, { x: 9.9, y: 1.7, w: 3.0, h: 3.0, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "title motif circle" });
    s.addImage({ data: I.chip, x: 10.55, y: 2.35, w: 1.7, h: 1.7, objectName: "title motif chip" });
    notes(s, "Overview of the Arm port of the evolutionary quantization project: what was built, what was measured, what was learned. All tiers are done: the Graviton3 run cost about two dollars, and the Apple M4 block (T4) about thirty.");
  }

  // 2. Why / role mapping -------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Context" });
    s.addText("The goal: tune LLM inference on Arm, and prove it with measurements", { placeholder: "title" });
    const rows = [
      [I.code, "Kernel-level optimization", "Hand-written NEON SDOT and SMMLA dot-product kernels for the Q8_0 and Q4_0 formats the search chooses between", H.accent1],
      [I.cogs, "Arm AI software stack", "llama.cpp with KleidiAI micro-kernels; NEON, dotprod, I8MM, SVE and SME code paths identified per run", H.accent3],
      [I.bolt, "Performance and power together", "Three objectives: accuracy loss, decode speed, energy per token, searched as a Pareto front", H.accent2],
      [I.search, "Profiling and analysis", "Memory roofline measured on Graviton3: Q8_0 decode sits at 95% of the bandwidth roof; perf counters and powermetrics next", H.accent6],
      [I.users, "Clear communication", "Reproducible harness, a decision log with null and revised results, and this deck", H.accent5],
    ];
    rows.forEach((r, i) => {
      const y = 1.55 + i * 1.05;
      card(s, 0.6, y, 12.1, 0.9, "row card " + (i + 1));
      circleIcon(s, 0.8, y + 0.12, 0.66, r[0], r[3], "row " + (i + 1));
      s.addText(r[1], { x: 1.75, y: y + 0.08, w: 3.6, h: 0.74, fontFace: "Calibri", fontSize: 18, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true, objectName: "row " + (i + 1) + " head" });
      s.addText(r[2], { x: 5.4, y: y + 0.08, w: 7.1, h: 0.74, fontFace: "Calibri", fontSize: 15, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: "row " + (i + 1) + " body" });
    });
    notes(s, "Mapping of the target role's requirements to what the project demonstrates. The roofline is measured on Graviton3; perf counters and the real power measurements are still to be done on hardware.");
  }

  // 3. Starting point ------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Context" });
    s.addText("Starting point: on the GPU, the proxy metric pointed the wrong way", { placeholder: "title" });
    stat(s, 0.6, 1.6, 3.7, "-4.5%", "INT4 is the smallest model yet 4.5% slower than FP16 on the RTX 3080 (batch 1)", H.accent4, "stat int4");
    stat(s, 4.6, 1.6, 3.7, "+22%", "uniform INT8 is 22% faster than FP16: bytes saved did not predict latency", H.accent3, "stat int8");
    stat(s, 8.6, 1.6, 4.1, "w >= 0.8", "accuracy weight at which the search first preferred a mixed genome over uniform INT8", H.accent1, "stat weight");
    card(s, 0.6, 3.75, 6.0, 2.9, "carry card");
    s.addText("Carried over to Arm", { x: 0.85, y: 3.9, w: 5.5, h: 0.45, fontFace: "Calibri", fontSize: 20, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "carry head" });
    s.addText(bullets(["Measure the cost model on the target silicon; never assume it", "Keep fitness a pure function of the genome, normalized to a fixed baseline", "Keep the asynchronous steady-state GA design"], { size: 16 }),
      { x: 0.85, y: 4.45, w: 5.5, h: 2.0, valign: "top", margin: 0, isTextBox: true, objectName: "carry bullets" });
    card(s, 6.9, 3.75, 5.8, 2.9, "revised card", C.text2);
    s.addText("Revised since", { x: 7.15, y: 3.9, w: 5.3, h: 0.45, fontFace: "Calibri", fontSize: 20, bold: true, color: C.accent2, margin: 0, isTextBox: true, objectName: "revised head" });
    s.addText("The per-block magnitudes in the GPU write-up (block 0 carrying 70% of INT8 cost) came from perplexity deltas that proved noisy at this scale. The corrected values are on slides 6 and 7.",
      { x: 7.15, y: 4.45, w: 5.3, h: 2.0, fontFace: "Calibri", fontSize: 16, color: C.background1, valign: "top", margin: 0, isTextBox: true, objectName: "revised text" });
    notes(s, "Findings from the base project on the RTX 3080 with bitsandbytes. The last box is an honest correction made during this port.");
  }

  // ======================================================================
  pres.addSection({ title: "Method" });
  // 4. Approach -------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
    s.addText("Approach: search per-block precision against three objectives", { placeholder: "title" });
    const boxes = [
      ["Genome", "8 genes, one per super-block of layers. Each gene: F16, Q8_0 or Q4_0", H.accent1],
      ["Assembler", "Splices pre-quantized GGUF tensors into one file; bit-identical to llama-quantize", H.accent3],
      ["Probes", "KL divergence (accuracy), llama-bench decode tok/s, energy per token", H.accent2],
      ["Pareto GA", "Steady-state, NSGA-II ranking; cache holds raw objective vectors only", H.accent6],
      ["Front", "Pick an operating point by accuracy budget, not by weights", H.dk2],
    ];
    const bw = 2.25, gap = 0.2, x0 = 0.6;
    boxes.forEach((b, i) => {
      const x = x0 + i * (bw + gap);
      s.addShape(pres.ShapeType.roundRect, { x, y: 1.7, w: bw, h: 2.6, rectRadius: 0.08, fill: { color: b[2] }, line: { color: b[2], width: 0 }, objectName: "pipeline box " + (i + 1) });
      const onAmber = b[2] === H.accent2;
      s.addText(b[0], { x: x + 0.15, y: 1.85, w: bw - 0.3, h: 0.5, fontFace: "Calibri", fontSize: 19, bold: true, color: onAmber ? C.text1 : C.background1, margin: 0, isTextBox: true, objectName: "pipeline head " + (i + 1) });
      s.addText(b[1], { x: x + 0.15, y: 2.4, w: bw - 0.3, h: 1.8, fontFace: "Calibri", fontSize: 14, color: onAmber ? C.text1 : C.background1, valign: "top", margin: 0, isTextBox: true, objectName: "pipeline text " + (i + 1) });
      if (i < boxes.length - 1)
        s.addShape(pres.ShapeType.line, { x: x + bw + 0.01, y: 3.0, w: gap - 0.02, h: 0, line: { color: H.dk1, width: 2, endArrowType: "triangle" }, objectName: "pipeline arrow " + (i + 1) });
    });
    const objs = [
      ["accuracy_penalty", "exp(KLD) - 1", "expected perplexity increase, measured against the original model's logits", H.accent4],
      ["speed_gain", "decode tok/s / baseline - 1", "decode is bandwidth-bound and what users feel on an edge device", H.accent3],
      ["energy_gain", "1 - J/token / baseline", "net of idle power, from the same run as the speed measurement", AMBER_TEXT],
    ];
    objs.forEach((o, i) => {
      const x = 0.6 + i * 4.1;
      card(s, x, 4.65, 3.9, 2.0, "objective card " + (i + 1));
      s.addText(o[0], { x: x + 0.2, y: 4.75, w: 3.5, h: 0.4, fontFace: "Calibri", fontSize: 17, bold: true, color: o[3], margin: 0, isTextBox: true, objectName: "objective name " + (i + 1) });
      s.addText(o[1], { x: x + 0.2, y: 5.18, w: 3.5, h: 0.4, fontFace: "Calibri", fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "objective formula " + (i + 1) });
      s.addText(o[2], { x: x + 0.2, y: 5.62, w: 3.5, h: 0.95, fontFace: "Calibri", fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "objective text " + (i + 1) });
    });
    notes(s, "The genome keeps the same shape as the GPU project, so the search machinery carried over. The GGUF assembler replaces bitsandbytes module substitution because bitsandbytes does not run on Arm or macOS.");
  }

  // 5. Dev strategy ---------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
    s.addText("Strategy: spend Mac dollars only on what only the Mac can answer", { placeholder: "title" });
    const tiers = [
      ["T0", "Desktop, no GPU", "Unit tests, GA, GGUF assembler, parsers, energy-meter backends", "free", true],
      ["T1", "Desktop + RTX 3080", "Real models, real accuracy, 2 search models, dry-run searches", "free", true],
      ["T2", "aarch64 emulation", "Cross-built llama.cpp and NEON kernels under QEMU: correctness only", "free", true],
      ["T3", "AWS Graviton3", "Native Arm Linux: KleidiAI on/off, roofline, kernels, and a dress rehearsal of the Mac workflow", "done: a few dollars", true],
      ["T4", "AWS Mac, Apple M4", "The only source of M4 speed and energy: both models measured over the full space in one 24-hour Dedicated Host block", "done: about $30", true],
    ];
    const w = 2.3, gap = 0.15;
    tiers.forEach((t, i) => {
      const x = 0.6 + i * (w + gap);
      card(s, x, 1.65, w, 4.1, "tier card " + t[0]);
      s.addShape(pres.ShapeType.ellipse, { x: x + 0.2, y: 1.85, w: 0.85, h: 0.85, fill: { color: t[4] ? C.accent3 : C.accent2 }, line: { color: t[4] ? C.accent3 : C.accent2, width: 0 }, objectName: "tier badge " + t[0] });
      s.addText(t[0], { x: x + 0.2, y: 1.85, w: 0.85, h: 0.85, fontFace: "Calibri", fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "tier label " + t[0] });
      s.addText(t[1], { x: x + 0.2, y: 2.85, w: w - 0.4, h: 0.7, fontFace: "Calibri", fontSize: 17, bold: true, color: C.text2, valign: "top", margin: 0, isTextBox: true, objectName: "tier name " + t[0] });
      s.addText(t[2], { x: x + 0.2, y: 3.6, w: w - 0.4, h: 1.4, fontFace: "Calibri", fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "tier text " + t[0] });
      s.addText(t[3], { x: x + 0.2, y: 5.15, w: w - 0.4, h: 0.4, fontFace: "Calibri", fontSize: 14, bold: true, color: t[4] ? C.accent3 : AMBER_TEXT, margin: 0, isTextBox: true, objectName: "tier cost " + t[0] });
    });
    s.addText("Rule: promote a tier only when the one below is green. Anything a cheap tier could have caught is a process failure on the expensive one.",
      { x: 0.6, y: 6.0, w: 12.1, h: 0.7, fontFace: "Calibri", fontSize: 16, italic: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true, objectName: "tier rule" });
    notes(s, "Green badges are complete (T0-T4). The Graviton3 run found the KleidiAI Q8_0 accuracy loss at full scale. The Mac costs about $30 (1.23 USD/h) with a 24-hour minimum, so every check that does not need M4 hardware was moved to the desktop, the 3080 or emulation.");
  }

  // ======================================================================
  pres.addSection({ title: "Results" });
  // 6. Metric lesson ----------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
    s.addText("Perplexity deltas were noise; KL divergence is not", { placeholder: "title" });
    s.addChart(pres.charts.BAR, [{ name: "Relative standard error of the measured effect", labels: ["Perplexity ratio", "KL divergence"], values: [61, 5.3] }], {
      x: 0.6, y: 1.5, w: 6.0, h: 5.1, barDir: "col", ...chartBase("Relative error of the effect estimate (%), Q8_0 on 1023 tokens"),
      chartColors: [H.accent4, H.accent3], showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMaxVal: 70, valAxisMinVal: 0,
      barGapWidthPct: 60, varyColors: true,
    });
    s.addText(bullets([
      "Single-block penalties from perplexity came out negative (quantizing a block \"improved\" the model)",
      "Single-block sums disagreed with the uniform genome, and two backends' profiles did not correlate (Spearman -0.33 / -0.24)",
      "KL divergence to the original model's logits: ~5% relative error, noise floor 1e-5 nats",
      "The same Q8_0 effect: PPL ratio 1.0038 +/- 0.0023 vs KLD 0.00165 +/- 0.00009",
    ], { size: 16, gap: 10 }), { x: 7.0, y: 1.6, w: 5.7, h: 3.7, valign: "top", margin: 0, isTextBox: true, objectName: "metric bullets" });
    card(s, 7.0, 5.35, 5.7, 1.3, "metric callout", C.text2);
    s.addText("Any conclusion drawn from small perplexity differences on one fixed slice carries this noise, and a search can fit it.",
      { x: 7.2, y: 5.4, w: 5.3, h: 1.2, fontFace: "Calibri", fontSize: 15, bold: true, color: C.background1, valign: "middle", margin: 0, isTextBox: true, objectName: "metric callout text" });
    notes(s, "The first sensitivity sweep used perplexity differences and was unusable. Switching to KL divergence (llama-perplexity --kl-divergence) fixed it. Accuracy penalty is defined as exp(KLD) - 1, the expected fractional perplexity increase.");
  }

  // 7. Sensitivity -------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
    s.addText("4-bit sensitivity is quantizer-independent; 8-bit is not", { placeholder: "title" });
    s.addChart(pres.charts.BAR, [
      { name: "Q4_0 (llama.cpp)", labels: ["0", "1", "2", "3", "4", "5", "6", "7"], values: [0.566, 0.656, 0.739, 0.960, 0.654, 0.502, 0.420, 1.431] },
      { name: "NF4 (bitsandbytes)", labels: ["0", "1", "2", "3", "4", "5", "6", "7"], values: [0.544, 0.612, 0.763, 0.808, 0.525, 0.523, 0.422, 1.284] },
    ], {
      x: 0.6, y: 1.5, w: 7.6, h: 5.1, barDir: "col", barGrouping: "clustered", ...chartBase("Phi-3-mini: accuracy cost of quantizing one block to 4 bits (%)"),
      chartColors: [H.accent1, H.accent2], showLegend: true, legendPos: "b", showValue: false, catAxisTitle: "super-block (0 = nearest the input)", showCatAxisTitle: true,
      catAxisTitleFontFace: FONT_CHART, catAxisTitleFontSize: 12, catAxisTitleColor: H.accent5, valAxisMinVal: 0,
    });
    stat(s, 8.6, 1.55, 4.1, "0.976", "Spearman correlation of the Q4_0 and NF4 per-block profiles; the last block costs the most in both", H.accent3, "stat spearman");
    stat(s, 8.6, 3.2, 4.1, "0.35", "Spearman for 8-bit: bitsandbytes INT8 is front-loaded, Q8_0 is flat and ~16x cheaper in total", H.accent4, "stat int8 spearman");
    card(s, 8.6, 4.95, 4.1, 1.7, "readme card", C.text2);
    s.addText("README finding 3 revised: block 0 carries about 33% of bnb INT8 cost (3-5x an interior block), not 70% (30x).",
      { x: 8.8, y: 5.0, w: 3.7, h: 1.6, fontFace: "Calibri", fontSize: 14, bold: true, color: C.background1, valign: "middle", margin: 0, isTextBox: true, objectName: "readme text" });
    notes(s, "KL-divergence sensitivity tables. Q8_0 is almost exactly additive (single-block sum 0.068% vs uniform 0.067%). Q4_0 is mildly super-additive. The CPU build reproduces the CUDA Q4_0 profile with Spearman 1.0.");
  }

  // 8. Models ---------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
    s.addText("Across models the structure generalizes only partly", { placeholder: "title" });
    s.addChart(pres.charts.LINE, [
      { name: "Phi-3-mini 3.8B (vs F16)", labels: ["0", "1", "2", "3", "4", "5", "6", "7"], values: [9.5, 11.1, 12.5, 16.2, 11.0, 8.5, 7.1, 24.1] },
      { name: "Llama-3.1-8B (vs Q8_0)", labels: ["0", "1", "2", "3", "4", "5", "6", "7"], values: [26.2, 9.2, 9.9, 11.3, 10.0, 7.4, 7.6, 18.4] },
    ], {
      x: 0.6, y: 1.5, w: 7.2, h: 3.5, ...chartBase("Q4_0 cost share per block (% of the summed single-block cost)"),
      chartColors: [H.accent1, H.accent4], lineSize: 3, lineDataSymbolSize: 8, showLegend: true, legendPos: "b", valAxisMinVal: 0,
    });
    const rows = [
      [{ text: "Model", options: { bold: true, color: C.background1, fill: { color: C.text2 } } }, { text: "Layers", options: { bold: true, color: C.background1, fill: { color: C.text2 } } }, { text: "Gene alphabet", options: { bold: true, color: C.background1, fill: { color: C.text2 } } }, { text: "Role", options: { bold: true, color: C.background1, fill: { color: C.text2 } } }],
      ["Phi-3-mini", "32", "F16 / Q8_0 / Q4_0", "search target"],
      ["Llama-3.1-8B", "32", "Q8_0 / Q4_0", "scale test (F16 will not fit)"],
      ["SmolLM2-360M", "32", "F16 / Q8_0 / Q4_0", "emulation proxy"],
      ["Llama-3.2-3B", "28", "pending licence", "second 3B-class model"],
    ];
    s.addTable(rows, { x: 0.6, y: 5.15, w: 7.2, colW: [1.7, 0.8, 2.2, 2.5], fontFace: "Calibri", fontSize: 12, color: H.dk1, border: { type: "solid", pt: 0.5, color: "D5DEE4" }, rowH: 0.3, valign: "middle", autoPage: false, objectName: "model table" });
    stat(s, 8.2, 1.55, 4.5, "0.55", "Spearman between the two Q4_0 profiles: moderate agreement, not high", H.accent6, "stat model spearman");
    s.addText(bullets([
      "Last block is top-2 in both models",
      "Block 0 is the most sensitive in Llama-3.1-8B (26%) but 6th of 8 in Phi-3",
      "Edge-to-interior ratio: 2.4x (8B), 1.5x (Phi-3)",
      "So \"protect the last block\" generalizes; \"protect the first\" is model dependent",
    ], { size: 15, gap: 9 }), { x: 8.2, y: 3.25, w: 4.5, h: 3.4, valign: "top", margin: 0, isTextBox: true, objectName: "model bullets" });
    notes(s, "Pre-registered similarity criteria: edge blocks most sensitive in each model (partly met), profile rank correlation (0.55, moderate). The Pareto-front and genome-shape criteria need the Arm measurements.");
  }

  // 9. Search results -----------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
    s.addText("The GA recovers the true front, and so does a greedy rule", { placeholder: "title" });
    const acc = [0, 0.346, 0.698, 1.116, 1.565, 2.038, 2.581, 3.380, 4.371];
    const spd = [0, 3.6, 7.6, 11.8, 16.4, 21.3, 26.7, 32.7, 39.2]; // Graviton3-calibrated model (plan 19.5)
    s.addChart(pres.charts.SCATTER, [{ name: "X", values: acc }, { name: "True Pareto front (all 256 genomes)", values: spd }], {
      x: 0.6, y: 1.5, w: 6.6, h: 5.1, ...chartBase("Llama-3.1-8B: exhaustive Pareto front"),
      chartColors: [H.accent1], lineSize: 2, lineDataSymbolSize: 10, showLegend: false,
      showValAxisTitle: true, valAxisTitle: "simulated decode gain vs Q8_0 (%)", valAxisTitleFontFace: FONT_CHART, valAxisTitleFontSize: 12, valAxisTitleColor: H.accent5,
      showCatAxisTitle: true, catAxisTitle: "accuracy cost (% perplexity increase, measured)", catAxisTitleFontFace: FONT_CHART, catAxisTitleFontSize: 12, catAxisTitleColor: H.accent5,
    });
    stat(s, 7.6, 1.5, 2.5, "1.000", "GA hypervolume ratio vs the exhaustive front (9 of 9 points)", H.accent3, "stat hv");
    stat(s, 10.3, 1.5, 2.4, "17", "measurements: the greedy rule gives the same front", H.accent1, "stat greedy");
    card(s, 7.6, 3.5, 5.1, 3.15, "search honesty card");
    s.addText("What this does and does not show", { x: 7.8, y: 3.6, w: 4.7, h: 0.45, fontFace: "Calibri", fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "honesty head" });
    s.addText(bullets([
      "With additive objectives the front is the sensitivity ordering: the GA is not needed here; its value must come from interactions, tested on measured Arm data",
      "Scalarized weights collapse: every weighting returns uniform Q4_0; pick points from the front by accuracy budget",
      "Phi-3: Q8_0 gives +261% decode for 0.07% accuracy, Q4_0 only +41% more for 6.8% (measured on Graviton3 within 4%)",
    ], { size: 13, gap: 7 }), { x: 7.8, y: 4.1, w: 4.7, h: 2.5, valign: "top", margin: 0, isTextBox: true, objectName: "honesty bullets" });
    notes(s, "All 256 genomes of the 8B model were measured (real KL-divergence accuracy), giving the exact Pareto front. Speed comes from a simulated Arm model driven by real tensor sizes and calibrated to the measured Graviton3 Phi-3 run (F16 decode reaches 48% of the memory roof, Q8_0 95%, Q4_0 72%); it is applied to the 8B model as an assumption. Energy is still a placeholder and is not shown. None of this is an Arm measurement of these searches.");
  }

  // 10. Arm findings -------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
    s.addText("On Graviton3, KleidiAI's Q8_0 path costs 37x the accuracy", { placeholder: "title" });
    s.addChart(pres.charts.BAR, [{ name: "Perplexity increase vs F16 (%)", labels: ["Q8_0 stock", "Q8_0 KleidiAI", "Q4_0 stock", "Q4_0 KleidiAI"], values: [0.08, 2.87, 8.97, 8.97] }], {
      x: 0.6, y: 1.5, w: 6.4, h: 5.1, barDir: "col", ...chartBase("Phi-3-mini on c7g.2xlarge: accuracy cost (% perplexity increase)"),
      chartColors: [H.accent5, H.accent4, H.accent5, H.accent5], varyColors: true, showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 10, barGapWidthPct: 50,
    });
    s.addText("Cause (from the source): KleidiAI re-quantizes each Q8_0 weight row to a single per-row scale. Q4_0 is unaffected: bit-identical to the stock path.",
      { x: 7.4, y: 1.55, w: 5.3, h: 1.2, fontFace: "Calibri", fontSize: 15, bold: true, color: C.text2, valign: "top", margin: 0, isTextBox: true, objectName: "cause text" });
    s.addText(bullets([
      "Against the stock Arm path KleidiAI adds no decode speed (0.94-1.05x) and no Q4_0 prefill; Q8_0 prefill +1.34-1.47x",
      "The 360M proxy under QEMU showed 8x; Phi-3 on real silicon shows 37x",
      "Fair baseline matters: --no-repack makes KleidiAI look 2.3x faster",
    ], { size: 14, gap: 7 }), { x: 7.4, y: 2.8, w: 5.3, h: 2.5, valign: "top", margin: 0, isTextBox: true, objectName: "t3 bullets" });
    card(s, 7.4, 5.35, 5.3, 1.3, "sme card", C.text2);
    circleIcon(s, 7.55, 5.58, 0.75, I.warn, H.accent4, "sme warn");
    s.addText("SME kernels give garbage under QEMU 8.2.2 (KLD 23-27). It becomes a selftest gate before any Mac run.",
      { x: 8.45, y: 5.4, w: 4.15, h: 1.2, fontFace: "Calibri", fontSize: 14, bold: true, color: C.background1, valign: "middle", margin: 0, isTextBox: true, objectName: "sme text" });
    notes(s, "Measured on an AWS c7g.2xlarge (Graviton3, 8 vCPUs). The stock bars are the build without KleidiAI, which keeps llama.cpp's own optimized Arm path. KleidiAI's Q8_0 loss is 37.5x in KL divergence (0.0283 vs 0.00076). Kernels chosen on Graviton3: Q4_0 SVE, Q8_0 I8MM, as predicted by emulation. The result does not predict the M4, where KleidiAI uses SME2 kernels and the stock path has nothing comparable.");
  }

  // 11. Kernel track + roofline ------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
    s.addText("Kernels bit-exact on Graviton3; the roofline shows where speed remains", { placeholder: "title" });
    s.addChart(pres.charts.BAR, [{ name: "Decode bandwidth as % of the memory roof", labels: ["F16", "Q8_0", "Q4_0"], values: [48, 95, 72] }], {
      x: 0.6, y: 1.45, w: 6.0, h: 3.2, barDir: "col", ...chartBase("Phi-3 decode, 8 threads: % of measured read bandwidth (158.7 GB/s)"),
      chartColors: [H.accent5, H.accent3, H.accent2], varyColors: true, showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMinVal: 0, valAxisMaxVal: 110, barGapWidthPct: 60,
    });
    s.addText(bullets([
      "Q8_0 decode is on the roof (95%): only moving fewer bytes helps",
      "Q4_0 reaches 72%: nibble unpacking limits it (up to ~1.3x left)",
      "F16 has no optimized kernel (48%): a poor baseline that inflates speedups",
      "Smaller is faster at decode: Q4_0 = 1.4-1.5x Q8_0 (opposite of the GPU)",
    ], { size: 13, gap: 5 }), { x: 0.6, y: 4.8, w: 6.0, h: 1.9, valign: "top", margin: 0, isTextBox: true, objectName: "roof bullets" });
    stat(s, 7.0, 1.55, 2.8, "108", "cases bit-identical to the reference, on real Graviton3 and under emulation", H.accent3, "stat cases");
    stat(s, 10.0, 1.55, 2.7, "1,710", "failures caught when the nibble offset is deliberately broken", H.accent4, "stat mutation");
    stat(s, 7.0, 3.2, 5.7, "47% / 31%", "of llama.cpp's single-thread bandwidth reached by my SDOT kernels (Q8_0 / Q4_0): correct, not yet fast", H.accent6, "stat kernel speed");
    card(s, 7.0, 5.0, 5.7, 1.65, "kernel next", C.text2);
    s.addText("Next: vector accumulation across blocks, several rows per pass, a repacked layout with SMMLA; then SME2 and a Triton/CUDA counterpart.",
      { x: 7.2, y: 5.05, w: 5.3, h: 1.55, fontFace: "Calibri", fontSize: 14, bold: true, color: C.background1, valign: "middle", margin: 0, isTextBox: true, objectName: "kernel next text" });
    notes(s, "Roofline: decode tokens per second times the bytes read per token, against the bw_probe read bandwidth at the same thread count. The own kernels (kernels/arm/qdot) are exact but reach 12 GB/s (Q8_0 SDOT) and 6.4 GB/s (Q4_0 SDOT) single-threaded, against 25.7 and 20.7 GB/s for llama.cpp's repacked kernels; likely reasons are listed in the plan as hypotheses, not yet verified.");
  }

  // 12. Rehearsal cost -------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
    s.addText("A rehearsal on Graviton3 measured what the Mac day costs", { placeholder: "title" });
    s.addChart(pres.charts.BAR, [
      { name: "KL-divergence run (accuracy)", labels: ["stock", "KleidiAI"], values: [56, 63] },
      { name: "speed and energy run", labels: ["stock", "KleidiAI"], values: [32, 50] },
    ], {
      x: 0.6, y: 1.5, w: 5.4, h: 3.5, barDir: "col", barGrouping: "stacked", ...chartBase("Seconds per genome and configuration (Phi-3, c7g.2xlarge)"),
      chartColors: [H.accent1, H.accent2], showLegend: true, legendPos: "b", showValue: true, dataLabelPosition: "ctr", dataLabelFormatCode: "0",
      dataLabelColor: H.lt1, valAxisMinVal: 0, barGapWidthPct: 70,
    });
    stat(s, 0.6, 5.1, 5.4, "88 s / 113 s", "total per genome for stock / KleidiAI; the F16 reference costs 9.3 / 7.4 minutes", H.accent1, "stat cost");
    s.addText("Hours to measure both models (256 genomes each) for a Mac that is faster than this machine by:", { x: 6.4, y: 1.5, w: 6.3, h: 0.7, fontFace: "Calibri", fontSize: 15, bold: true, color: C.text2, valign: "top", margin: 0, isTextBox: true, objectName: "projection head" });
    const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, align: "center" } });
    const cell = (t, o = {}) => ({ text: t, options: { align: "center", ...o } });
    const rows = [
      [hdr("design"), hdr("1x"), hdr("1.5x"), hdr("2x"), hdr("3x")],
      [{ text: "both configurations, all genomes", options: { align: "left" } }, cell("44 h", { color: H.accent4, bold: true }), cell("30 h", { color: H.accent4, bold: true }), cell("22 h", { color: H.accent4, bold: true }), cell("15 h", { color: H.accent3, bold: true })],
      [{ text: "KleidiAI in full + stock for 12 genomes", options: { align: "left", bold: true } }, cell("25.8 h", { color: H.accent4, bold: true }), cell("17.2 h", { color: H.accent3, bold: true }), cell("12.9 h", { color: H.accent3, bold: true }), cell("8.6 h", { color: H.accent3, bold: true })],
    ];
    s.addTable(rows, { x: 6.4, y: 2.25, w: 6.3, colW: [2.7, 0.9, 0.9, 0.9, 0.9], fontFace: "Calibri", fontSize: 13, color: H.dk1, border: { type: "solid", pt: 0.5, color: "D5DEE4" }, rowH: 0.5, valign: "middle", autoPage: false, objectName: "projection table" });
    s.addText("Green fits the 18-hour queue; red does not.", { x: 6.4, y: 3.85, w: 6.3, h: 0.35, fontFace: "Calibri", fontSize: 12, italic: true, color: C.accent5, margin: 0, isTextBox: true, objectName: "projection note" });
    card(s, 6.4, 4.4, 6.3, 2.25, "design card");
    s.addText("Design change for the Mac day", { x: 6.6, y: 4.5, w: 5.9, h: 0.4, fontFace: "Calibri", fontSize: 17, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "design head" });
    s.addText(bullets([
      "Full table in the configuration that ships (KleidiAI); the fair baseline (stock) only for the reference, uniform baselines and greedy chain",
      "The M4's speedup is unknown: measure decode speed first, then size the queue",
      "Measured speeds match the calibrated model within 4%",
    ], { size: 13, gap: 5 }), { x: 6.6, y: 4.95, w: 5.9, h: 1.65, valign: "top", margin: 0, isTextBox: true, objectName: "design bullets" });
    notes(s, "Nine genomes were measured in two configurations on a c7g.2xlarge with the mock energy meter, so energy is not a measurement. Per-evaluation cost: 88 s stock and 113 s KleidiAI, split between the 2048-token KL-divergence run and the llama-bench speed run. The 8B model is scaled by weight bytes (2.1x), an assumption the optional 8B rehearsal would replace. Full coverage of both models in both configurations does not fit one 24 hour block unless the Mac is about 3x faster than Graviton3, so the default design measures the shipped configuration in full and the baseline for 12 genomes.");
  }

  // 13. Rehearsal problems -------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
    s.addText("The rehearsal found five problems before the paid block", { placeholder: "title" });
    const rows = [
      [I.warn, "Scratch disk was RAM-backed", "Ubuntu 26.04 mounts /tmp as an 8 GB tmpfs, and the gate demanded space for every source file. The selftest stopped the queue early. Fix: scratch on disk, real space check.", H.accent4],
      [I.clock, "Idle time ate the budget", "The deadline was fixed at the first start, so two idle hours left the table job 49 minutes. Fix: --restart-clock re-anchors it.", H.accent2],
      [I.cogs, "A job was killed mid-genome", "A hard timeout lost the genome in flight and skipped the analysis. Fix: stop before a genome that would overrun; analysis jobs always run.", H.accent6],
      [I.search, "No baseline in a restricted table", "A Q8_0/Q4_0-only table had no F16 reference row (found on the desktop). Fix: always measure the reference first; the analyzer falls back with a warning.", H.accent1],
      [I.layers, "A sync overwrote tracked evidence", "Syncing results/ replaced the first Graviton files. Fix: restore from git, sync only rehearsal files; two instances agree within 2-3%.", H.accent5],
    ];
    rows.forEach((r, i) => {
      const y = 1.5 + i * 1.05;
      card(s, 0.6, y, 12.1, 0.92, "problem card " + (i + 1));
      circleIcon(s, 0.8, y + 0.13, 0.66, r[0], r[3], "problem " + (i + 1));
      s.addText(r[1], { x: 1.7, y: y + 0.06, w: 3.5, h: 0.8, fontFace: "Calibri", fontSize: 16, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true, objectName: "problem " + (i + 1) + " head" });
      s.addText(r[2], { x: 5.3, y: y + 0.06, w: 7.3, h: 0.8, fontFace: "Calibri", fontSize: 13, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: "problem " + (i + 1) + " body" });
    });
    notes(s, "All five were found with cheap tiers, each would have cost hours of a 24 hour paid block. The selftest gate worked as designed in the first case: it stopped the queue on a failed check instead of letting the table job fail hours in. The unit tests with synthetic tables had not caught the missing-baseline case; the desktop rehearsal with real binaries did.");
  }

  // ---------- Apple M4 data (read at build time; rebuild after the run to refresh the interim slide) ----------
  const fs = require("fs");
  const M4DIR = [path.join(__dirname, "..", "results"), path.join(__dirname, "..", "results_m4", "results")].find((d) => fs.existsSync(path.join(d, "m4_thread_scan.json")));
  const readJson = (f) => JSON.parse(fs.readFileSync(path.join(M4DIR, f), "utf8"));
  const scan = readJson("m4_thread_scan.json").rows;
  const scanVal = (cfg, prec, t, key) => scan.find((r) => r.config === cfg && r.precision === prec && r.threads === t)[key];
  const THREADS = [1, 2, 4, 6, 8, 10];

  pres.addSection({ title: "Apple M4" });
  // 14. M4 selftest ----------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Apple M4" });
    s.addText("On the Apple M4 the selftest passed: SME2 works and energy is measurable", { placeholder: "title" });
    const items = [
      ["SME2 output is sane", "F16 vs base logits 7.1e-05 nats (limit 1e-04). The garbage seen under QEMU was the emulator, not the kernel.", H.accent3, "OK"],
      ["KleidiAI Q8_0 loss reproduces", "KL divergence x25.6 vs stock (0.0286 vs 0.0011), Graviton3: x28-37. Same with and without SME.", H.accent4, "x25.6"],
      ["SME changes Q4_0 slightly", "Q4_0 KLD 1.4% apart with SME on (0.1126 vs 0.1110), 0.0% with SME off: a small numerical difference.", H.accent2, "1.4%"],
      ["Power is measurable", "powermetrics: 0.06 W idle, 8.7 W under load; cpu_power in mW parsed as assumed (key names confirmed).", H.accent1, "8.7 W"],
    ];
    items.forEach((it, i) => {
      const x = 0.6 + (i % 2) * 6.15, y = 1.5 + Math.floor(i / 2) * 2.15;
      card(s, x, y, 5.95, 1.95, "selftest card " + (i + 1));
      s.addText(it[3], { x: x + 0.25, y: y + 0.15, w: 1.9, h: 0.9, fontFace: "Cambria", fontSize: 34, bold: true, color: it[2], valign: "middle", margin: 0, isTextBox: true, objectName: "selftest value " + (i + 1) });
      s.addText(it[0], { x: x + 2.2, y: y + 0.15, w: 3.55, h: 0.9, fontFace: "Calibri", fontSize: 17, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true, objectName: "selftest head " + (i + 1) });
      s.addText(it[1], { x: x + 0.25, y: y + 1.1, w: 5.45, h: 0.8, fontFace: "Calibri", fontSize: 13, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "selftest text " + (i + 1) });
    });
    s.addText("Models regenerated on the Mac: Phi-3 weights are bit-identical to the desktop's (tensor digests), and the F16 perplexity agrees with Graviton3 to 0.02% (4.8225 vs 4.8236); both 8B files match byte for byte.",
      { x: 0.6, y: 5.9, w: 12.1, h: 0.8, fontFace: "Calibri", fontSize: 14, italic: true, color: C.accent5, valign: "top", margin: 0, isTextBox: true, objectName: "selftest footer" });
    notes(s, "Selftest on the mac-m4.metal host (macOS 26.7, Apple M4, 4 performance + 6 efficiency cores, 24 GiB), run before any measurement: platform, disk, memory, Low Power Mode, thermal state, that powermetrics power rises under load, CPU-only backends (Metal off), and per-configuration KL divergence against the stock build. Three configurations were usable: stock, KleidiAI with SME2, KleidiAI without SME. The SME2 path produced sane output on real hardware, which the QEMU run could not establish. The KleidiAI Q8_0 accuracy loss is the same finding as on Graviton3, and it does not depend on SME. Phi-3's GGUF was regenerated on the Mac with a newer converter: the file hashes differ but all 195 tensors are bit-identical to the desktop's.");
  }

  // 15. Thread scan ------------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Apple M4" });
    s.addText("Decode on the M4 is bandwidth-bound: 8 threads win, 10 lose", { placeholder: "title" });
    const ser = (cfg, prec, name) => ({ name, labels: THREADS.map(String), values: THREADS.map((t) => scanVal(cfg, prec, t, "decode_tps")) });
    s.addChart(pres.charts.LINE, [
      ser("stock", "int4", "Q4_0 stock"), ser("kai", "int4", "Q4_0 KleidiAI"), ser("stock", "int8", "Q8_0 stock"), ser("kai", "int8", "Q8_0 KleidiAI"),
    ], {
      x: 0.6, y: 1.5, w: 7.2, h: 5.1, ...chartBase("Phi-3-mini decode speed (tokens/s) by thread count"),
      chartColors: [H.accent1, H.accent2, H.accent5, H.accent4], lineSize: 2, lineDataSymbolSize: 7, showLegend: true, legendPos: "b",
      showValAxisTitle: true, valAxisTitle: "decode tokens/s", valAxisTitleFontFace: FONT_CHART, valAxisTitleFontSize: 12, valAxisTitleColor: H.accent5,
      showCatAxisTitle: true, catAxisTitle: "threads (4 performance + 6 efficiency cores)", catAxisTitleFontFace: FONT_CHART, catAxisTitleFontSize: 12, catAxisTitleColor: H.accent5, valAxisMinVal: 0,
    });
    const q4 = scanVal("stock", "int4", 8, "decode_tps"), q8 = scanVal("stock", "int8", 8, "decode_tps");
    stat(s, 8.2, 1.5, 2.1, q4.toFixed(1), "tokens/s, Q4_0 stock at 8 threads", H.accent1, "stat q4");
    stat(s, 10.5, 1.5, 2.2, (q4 / q8).toFixed(2) + "x", "Q4_0 over Q8_0 decode (Graviton3: 1.41-1.52x)", H.accent3, "stat ratio");
    card(s, 8.2, 3.2, 4.5, 3.4, "scan card");
    s.addText("What the scan shows", { x: 8.4, y: 3.3, w: 4.1, h: 0.4, fontFace: "Calibri", fontSize: 17, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "scan head" });
    s.addText(bullets([
      "4 threads (the performance cores alone) do not saturate memory: Q4_0 32.6 vs 45.6 tokens/s at 6",
      "10 threads cost 10-13% decode and more energy per token than 8",
      "KleidiAI adds no decode speed: +6% (Q8_0), -2% (Q4_0) at 8 threads. At 1 thread it wins on Q4_0 (29 vs 20), until bandwidth saturates",
      "Chosen N = 8 for every later measurement",
    ], { size: 12.5, gap: 5 }), { x: 8.4, y: 3.75, w: 4.1, h: 2.8, valign: "top", margin: 0, isTextBox: true, objectName: "scan bullets" });
    notes(s, "Thread scan on the M4 with the real powermetrics meter, Phi-3 uniform Q8_0 and uniform Q4_0, stock and KleidiAI builds. macOS cannot pin threads, so the count decides which cores run. Decode is memory-bandwidth-bound: Q4_0 at 47.7 tokens/s moves roughly 105 GB/s of a 120 GB/s peak (approximate, from file sizes). The best decode speed is at 8 threads in all four series; two efficiency cores help, four hurt. Energy per token at 6 and 8 threads is within a few percent. One thread is the energy-optimal point for KleidiAI Q4_0 (0.21 J/token) at 29 tokens/s, so the thread count is itself a speed versus energy trade-off that the genome does not capture.");
  }

  // 16. KleidiAI on M4 vs Graviton3 ------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Apple M4" });
    s.addText("KleidiAI on the M4: the Graviton3 result holds, and energy is worse", { placeholder: "title" });
    const ratio = (prec, key) => scanVal("kai", prec, 8, key) / scanVal("stock", prec, 8, key);
    const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, align: "center" } });
    const cell = (t, o = {}) => ({ text: t, options: { align: "center", ...o } });
    const left = (t, o = {}) => ({ text: t, options: { align: "left", ...o } });
    const f = (v) => "x" + v.toFixed(2);
    const rows = [
      [hdr("KleidiAI vs stock llama.cpp"), hdr("Graviton3 (8 threads)"), hdr("Apple M4 (8 threads)")],
      [left("Decode, Q8_0"), cell("x0.94-1.05"), cell(f(ratio("int8", "decode_tps")), { bold: true })],
      [left("Decode, Q4_0"), cell("x0.94-1.05"), cell(f(ratio("int4", "decode_tps")), { bold: true })],
      [left("Prefill, Q8_0"), cell("x1.34-1.47"), cell(f(ratio("int8", "prefill_tps")), { bold: true, color: H.accent3 })],
      [left("Prefill, Q4_0"), cell("x1.00"), cell(f(ratio("int4", "prefill_tps")), { bold: true })],
      [left("Q8_0 accuracy loss (KL divergence)"), cell("x37.5"), cell("x25.6", { bold: true, color: H.accent4 })],
      [left("Energy per token, Q8_0"), cell("not measured"), cell(f(ratio("int8", "joules_per_token")), { bold: true, color: H.accent4 })],
      [left("Energy per token, Q4_0"), cell("not measured"), cell(f(ratio("int4", "joules_per_token")), { bold: true })],
    ];
    s.addTable(rows, { x: 0.6, y: 1.55, w: 7.6, colW: [3.4, 2.1, 2.1], fontFace: "Calibri", fontSize: 14, color: H.dk1, border: { type: "solid", pt: 0.5, color: "D5DEE4" }, rowH: 0.52, valign: "middle", autoPage: false, objectName: "kleidiai table" });
    s.addText("Ratios of the KleidiAI build to the stock build; below x1 is worse. Phi-3 uniform Q8_0 and Q4_0.", { x: 0.6, y: 5.8, w: 7.6, h: 0.4, fontFace: "Calibri", fontSize: 12, italic: true, color: C.accent5, margin: 0, isTextBox: true, objectName: "kleidiai note" });
    card(s, 8.6, 1.55, 4.1, 5.05, "kleidiai card");
    s.addText("Reading it", { x: 8.8, y: 1.65, w: 3.7, h: 0.4, fontFace: "Calibri", fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "kleidiai head" });
    s.addText(bullets([
      "SME2 does not speed up decode: bandwidth is the limit, not compute",
      "It does speed up Q8_0 prefill, by about 1.8x with 8 threads and 2.6x with 4",
      "Q8_0 prefill peaks at 4 threads: a plausible cause is an SME unit shared per core cluster (a hypothesis, untested)",
      "Q8_0 through KleidiAI costs accuracy and, per token, energy: the search must treat the kernel path as part of the genome's cost",
    ], { size: 13, gap: 7 }), { x: 8.8, y: 2.1, w: 3.7, h: 4.4, valign: "top", margin: 0, isTextBox: true, objectName: "kleidiai bullets" });
    notes(s, "Same Phi-3 models and the same two builds on both machines, ratios at 8 threads. On Graviton3 energy could not be measured. Energy ratios here are joules per decoded token from powermetrics CPU power with the idle baseline handled by the driver; the Q8_0 energy ratio above 1 means KleidiAI uses more energy per token, consistent with its extra work and no decode gain. The prefill result is where KleidiAI helps, and prefill is not the objective of a decode-speed search, only part of the cost of each evaluation.");
  }

  // 17. 8B table (final) -------------------------------------------------------------------------
  const table = (f) => fs.readFileSync(path.join(M4DIR, f), "utf8").trim().split("\n").map((l) => JSON.parse(l)).filter((r) => !r.error);
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Apple M4" });
    const rowsAll = table("m4_table_llama3.1-8b.jsonl").filter((r) => r.config === "kai");
    const pts = rowsAll.map((r) => ({ x: (Math.exp(r.kld) - 1) * 100, y: r.decode_tps, g: r.genome }));
    const front = (p) => !pts.some((q) => q !== p && q.x <= p.x && q.y >= p.y && (q.x < p.x || q.y > p.y));
    const nFront = pts.filter(front).length;
    s.addText("Llama-3.1-8B on the M4: all " + pts.length + " genomes measured", { placeholder: "title" });
    const sorted = [...pts].sort((a, b) => a.x - b.x);
    s.addChart(pres.charts.SCATTER, [
      { name: "X", values: sorted.map((p) => p.x) },
      { name: "dominated genome", values: sorted.map((p) => (front(p) ? null : p.y)) },
      { name: "front for accuracy and decode", values: sorted.map((p) => (front(p) ? p.y : null)) },
    ], {
      x: 0.6, y: 1.5, w: 7.6, h: 5.1, ...chartBase("KleidiAI configuration, 8 threads, every genome measured"),
      chartColors: [H.accent5, H.accent4], lineSize: 0, lineDataSymbolSize: 8, showLegend: true, legendPos: "b",
      showValAxisTitle: true, valAxisTitle: "decode tokens/s (measured)", valAxisTitleFontFace: FONT_CHART, valAxisTitleFontSize: 12, valAxisTitleColor: H.accent5,
      showCatAxisTitle: true, catAxisTitle: "accuracy cost, exp(KL divergence) - 1, in %", catAxisTitleFontFace: FONT_CHART, catAxisTitleFontSize: 12, catAxisTitleColor: H.accent5,
    });
    stat(s, 8.6, 1.5, 2.0, String(readJson("m4_analysis_llama3.1-8b.json").configs.kai.front_size), "Pareto front points (accuracy, speed, energy)", H.accent1, "stat front");
    stat(s, 10.6, 1.5, 2.1, "0.9999", "hypervolume ratio: GA (400 evals) and the greedy rule", H.accent3, "stat hv");
    card(s, 8.6, 3.35, 4.1, 3.25, "result card");
    s.addText("What the measurements say", { x: 8.8, y: 3.45, w: 3.7, h: 0.4, fontFace: "Calibri", fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "result head" });
    s.addText(bullets([
      "Speed, energy and accuracy are almost additive over the 8 blocks (R2 0.993-0.997): a sensitivity ordering already gives the front",
      "Energy per token tracks decode time (correlation 0.9975): three objectives act as about two",
      "Decode ranges from 14.3 to 23.1 tokens/s (+62%); uniform Q4_0 is the fastest genome",
    ], { size: 12.5, gap: 6 }), { x: 8.8, y: 3.9, w: 3.7, h: 2.65, valign: "top", margin: 0, isTextBox: true, objectName: "result bullets" });
    notes(s, "All 256 genomes of the 8B model were measured on the M4 in the KleidiAI configuration (138 s each, 10.2 hours), plus 12 genomes in the stock configuration. Accuracy is KL divergence against the Q8_0 reference converted to exp(KLD)-1; speed is llama-bench decode with 8 threads; energy comes from powermetrics CPU power. The genetic search with 400 evaluations over ten seeds reaches 0.9999 of the exact front's hypervolume, and so does the nine-genome sensitivity-greedy chain. An additive per-block fit explains seconds per token with R2 0.997, joules per token 0.995, KL divergence 0.993, so the search problem is nearly separable on this hardware.");
  }

  // 18. Phi-3: KleidiAI vs stock per genome ------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Apple M4" });
    s.addText("Phi-3 on the M4: KleidiAI costs accuracy and energy only on the Q8_0 blocks", { placeholder: "title" });
    const rows = table("m4_table_phi3-mini.jsonl").filter((r) => /^[48]+$/.test(r.genome));
    const kai = {}, stk = {};
    rows.forEach((r) => (r.config === "kai" ? kai : stk)[r.genome] = r);
    const med = (a) => { const b = [...a].sort((x, y) => x - y); return b[Math.floor(b.length / 2)]; };
    const byN = {};
    Object.keys(kai).filter((g) => stk[g]).forEach((g) => {
      const n = g.split("").filter((c) => c === "4").length;
      (byN[n] = byN[n] || []).push({ d: kai[g].decode_tps / stk[g].decode_tps, e: kai[g].joules_per_token / stk[g].joules_per_token, p: kai[g].prefill_tps / stk[g].prefill_tps });
    });
    const ns = Object.keys(byN).map(Number).sort((a, b) => a - b);
    s.addChart(pres.charts.LINE, [
      { name: "prefill speed", labels: ns.map(String), values: ns.map((n) => +med(byN[n].map((x) => x.p)).toFixed(3)) },
      { name: "decode speed", labels: ns.map(String), values: ns.map((n) => +med(byN[n].map((x) => x.d)).toFixed(3)) },
      { name: "energy per token", labels: ns.map(String), values: ns.map((n) => +med(byN[n].map((x) => x.e)).toFixed(3)) },
    ], {
      x: 0.6, y: 1.5, w: 7.2, h: 5.1, ...chartBase("KleidiAI / stock, median over the genomes with n Q4_0 blocks (n = 256 genomes in all)"),
      chartColors: [H.accent3, H.accent1, H.accent4], lineSize: 2, lineDataSymbolSize: 7, showLegend: true, legendPos: "b",
      showValAxisTitle: true, valAxisTitle: "ratio (1.0 = same as stock)", valAxisTitleFontFace: FONT_CHART, valAxisTitleFontSize: 12, valAxisTitleColor: H.accent5,
      showCatAxisTitle: true, catAxisTitle: "number of blocks (of 8) at Q4_0; the rest are Q8_0", catAxisTitleFontFace: FONT_CHART, catAxisTitleFontSize: 12, catAxisTitleColor: H.accent5, valAxisMinVal: 0.8,
    });
    stat(s, 8.2, 1.5, 2.2, "x31", "KL divergence of all-Q8_0 under KleidiAI vs stock; x1.0 for all-Q4_0", H.accent4, "stat kld");
    stat(s, 10.5, 1.5, 2.2, "6%", "more hypervolume on the stock front than on the KleidiAI front", H.accent1, "stat hv6");
    card(s, 8.2, 3.35, 4.5, 3.25, "phi card");
    s.addText("Both configurations fully measured", { x: 8.4, y: 3.45, w: 4.1, h: 0.4, fontFace: "Calibri", fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "phi head" });
    s.addText(bullets([
      "256 genomes in each configuration; fronts of 36 (KleidiAI) and 35 (stock) points",
      "The GA reaches 0.9998 and 0.9999 of the true hypervolume",
      "Under KleidiAI the accuracy cost is the one quantity that is not additive (R2 0.83 vs 0.99 stock)",
    ], { size: 12.5, gap: 6 }), { x: 8.4, y: 3.9, w: 4.1, h: 2.65, valign: "top", margin: 0, isTextBox: true, objectName: "phi bullets" });
    notes(s, "Phi-3-mini, every genome of the Q8_0/Q4_0 space measured in both the KleidiAI and the stock builds on the M4 (about 45 seconds per genome and configuration). The curves are the per-genome ratio KleidiAI over stock, grouped by how many of the eight blocks are Q4_0. KleidiAI only changes the Q8_0 path: at zero Q4_0 blocks it gives +78% prefill and +8% decode but 31x the KL divergence and 19% more energy per token; with all blocks Q4_0 the two builds agree on accuracy and KleidiAI is slightly slower. Hypervolume is measured against the stock reference genome, so the stock front's 6% advantage is the cost of shipping the KleidiAI path.");
  }

  // 19. Lessons ------------------------------------------------------------------------------
  {
    const s = pres.addSlide({ masterName: "CLOSE", sectionTitle: "Apple M4" });
    s.addText("Lessons learned, and what comes next", { placeholder: "title" });
    const lessons = [
      ["Validate the cost model on the target", "Bytes did not predict latency, and perplexity did not resolve sensitivity. The metric is part of the result."],
      ["Report revised and null results", "The GA ties a greedy rule under additive objectives; an earlier per-block claim shrank from 70% to 33%."],
      ["Hardware decides accuracy, not just speed", "KleidiAI's Q8_0 re-quantization costs 37x accuracy on Phi-3 and adds no decode speed over the stock path."],
      ["Build cheap tiers and gates", "A few dollars of rehearsal found five problems and sized the 24-hour Mac block; its selftest then passed on first use."],
    ];
    lessons.forEach((l, i) => {
      const y = 1.5 + i * 1.3;
      s.addShape(pres.ShapeType.ellipse, { x: 0.6, y: y + 0.12, w: 0.6, h: 0.6, fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 }, objectName: "lesson badge " + (i + 1) });
      s.addText(String(i + 1), { x: 0.6, y: y + 0.12, w: 0.6, h: 0.6, fontFace: "Calibri", fontSize: 18, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "lesson number " + (i + 1) });
      s.addText(l[0], { x: 1.4, y, w: 5.2, h: 0.45, fontFace: "Calibri", fontSize: 17, bold: true, color: C.background1, margin: 0, isTextBox: true, objectName: "lesson head " + (i + 1) });
      s.addText(l[1], { x: 1.4, y: y + 0.45, w: 5.2, h: 0.8, fontFace: "Calibri", fontSize: 13, color: C.background2, valign: "top", margin: 0, isTextBox: true, objectName: "lesson text " + (i + 1) });
    });
    card(s, 7.1, 1.5, 5.6, 5.15, "next card", C.text1);
    s.addText("Next steps", { x: 7.35, y: 1.65, w: 5.1, h: 0.5, fontFace: "Calibri", fontSize: 22, bold: true, color: C.accent2, margin: 0, isTextBox: true, objectName: "next head" });
    s.addText(bullets([
      "Done on the M4: both spaces measured (256 genomes per model and configuration, 8B stock for 12). Next: choose operating points on the front by accuracy budget",
      "Test where the GA could still win: the non-additive accuracy cost under KleidiAI, and thread count as a gene (speed against energy)",
      "Optimize the Q4_0 kernel toward the memory roof; SME2 and Triton/CUDA variants",
      "Report the Q8_0 finding upstream; third model once licence access is granted",
      "Honest limits: one machine and one run per genome; energy is CPU power only; no hardware performance counters; greedy comparison missing for Phi-3",
    ], { size: 14, color: C.background1, gap: 9 }), { x: 7.35, y: 2.25, w: 5.1, h: 4.3, valign: "top", margin: 0, isTextBox: true, objectName: "next bullets" });
    notes(s, "Closing slide: four lessons and the concrete next steps. The limits are stated explicitly so the results are not over-read. All tiers T0-T4 are complete: the M4 block measured both models over the full space.");
  }

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
}

main().catch((e) => { console.error(e); process.exit(1); });
