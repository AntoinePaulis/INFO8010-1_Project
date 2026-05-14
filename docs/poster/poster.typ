// ─── Dependencies ─────────────────────────────────────────────────────────────
#import "@preview/cetz:0.4.2": canvas, draw

// ─── Document meta ────────────────────────────────────────────────────────────
#set document(
  title:  "Tennis Video Analysis: Ball Tracking, Court Detection & Player Localization",
  author: "Andy Jalloh",
)

// ─── Typography ───────────────────────────────────────────────────────────────
#set par(justify: true)
#set text(font: "sans-serif", size: 8pt, weight: "light", hyphenate: false)
#set strong(delta: 200)
#set math.equation(numbering: "(1)")
#show heading: set text(size: 1em, weight: "medium")
#show heading: set block(above: 1.1em, below: 0.55em)
#set list(tight: true, marker: strong[•])
#set figure.caption(separator: [. ])
#set figure(gap: 0.4em)

// ─── Colour palette ───────────────────────────────────────────────────────────
#let blue  = rgb("#1565C0")
#let green = rgb("#2E7D32")
#let red   = rgb("#C62828")
#let amber = rgb("#E65100")

// ─── Helpers ──────────────────────────────────────────────────────────────────
#let showcase(inset: 0.55em, body) = rect(
  fill: rgb("#F0F4F8"), inset: inset, radius: 0.4em, width: 100%, body
)

#let metric-box(label, value, col: blue) = rect(
  fill: col.lighten(88%), inset: (x: 0.4em, y: 0.4em),
  radius: 0.3em, width: 100%,
  stack(dir: ttb, spacing: 0.08em,
    text(size: 0.68em, fill: col, weight: "medium", label),
    text(size: 1.25em, weight: "bold", fill: col, value),
  )
)

#let task-bar(col, body) = rect(
  fill: col, inset: (x: 0.6em, y: 0.3em),
  radius: 0.25em, width: 100%,
  text(fill: white, weight: "medium", body)
)

#let todo(body) = rect(
  fill: rgb("#FFF8F0"),
  stroke: (paint: amber.lighten(40%), dash: "dashed"),
  inset: 0.5em, radius: 0.3em, width: 100%, height: 2cm,
  align(center + horizon,
    text(fill: amber, style: "italic", size: 0.85em, [*\[TODO\]* — ] + body)
  )
)

// ─── Page layout ──────────────────────────────────────────────────────────────
#set page(paper: "a4", flipped: true, margin: 0.5cm, columns: 3)
#set columns(gutter: 0.5cm)

// ─── Title banner ─────────────────────────────────────────────────────────────
#place(top, float: true, clearance: 0.4cm, scope: "parent",
  stack(dir: ltr, spacing: 1fr,
    align(horizon,
      stack(dir: ttb, spacing: 0.45em,
        text(size: 1.8em, weight: "medium",
          [Tennis Video Analysis: Ball Tracking, Court Detection & Player Localization]),
        text(size: 1.05em,
          [Andy Jalloh · #text(style: "italic")[\[Co-author 2\]] · #text(style: "italic")[\[Co-author 3\]]
           #h(1em)—#h(1em)
           INFO8010-1 Deep Learning · University of Liège · 2025–2026]),
      )
    ),
    align(horizon + center,
      rect(fill: blue, inset: (x: 0.85em, y: 0.4em), radius: 0.3em,
        stack(dir: ttb, spacing: 0.12em,
          text(fill: white, size: 0.95em, weight: "medium", [ULiège]),
          text(fill: white.lighten(30%), size: 0.72em, [INFO8010-1]),
        )
      )
    ),
  )
)

// ═══════════════════════════════════════════════════════════════════════════════
// COLUMN 1
// ═══════════════════════════════════════════════════════════════════════════════

= Abstract

We present three independent deep learning pipelines for spatial analysis of broadcast tennis video. *Ball tracking:* TrackNet, a fully-convolutional encoder-decoder taking 3 stacked RGB frames (9-ch, 640×360) and regressing a 256-class Gaussian heatmap with focal loss (F1 = 0.927, 96.4% of detections within 5 px). *Court detection:* identical backbone re-headed to 15 keypoint heatmaps—partial convergence after lr correction (F1 = 0.129, 12.9% keypoint accuracy). *Player localization:* fine-tuned YOLOv8s (mAP\@0.5 = 0.978, mAP\@0.5:0.95 = 0.711). All models trained on the Alan HPC cluster and tracked via Weights & Biases.

= Problem Statement

Classical tracking fails under broadcast tennis conditions. Key challenges:

- *Ball:* only 2–5 px wide at 640×360; high speed causes motion blur; frequently occluded by players or net.
- *Court:* 14 boundary keypoints must be located simultaneously under perspective distortion and camera motion.
- *Players:* large scale variation, mutual occlusion, cluttered advertising backgrounds.

#figure(
  image("figures/heatmap_variance.png"),
  caption: [Gaussian heatmap targets at σ²∈{7,10,15}. σ²=10 gives a ~3.7 px 50%-intensity radius, matching the real ball size at 640×360.]
)

= Methodology

== Datasets

#showcase[
  #set text(size: 7.5pt)
  #table(
    columns: (auto, 1fr, auto, auto),
    stroke: none,
    inset: (x: 0.45em, y: 0.3em),
    fill: (_, row) => if calc.even(row) { rgb("#E4EBF5") } else { white },
    [*Task*], [*Dataset*], [*Frames*], [*Split*],
    [Ball],   [TrackNet Kaggle (10 games, ≤15 clips/game)], [19 842], [70/15/15%],
    [Court],  [GitHub keypoint dataset (JSON, 14 kp + centre)], [—], [—],
    [Player], [Roboflow tennis-player (1 class, bbox)], [—], [80/10/10%],
  )
]

#colbreak()

// ═══════════════════════════════════════════════════════════════════════════════
// COLUMN 2
// ═══════════════════════════════════════════════════════════════════════════════

== Ball Tracking — TrackNet

#task-bar(blue, [Ball Tracking — TrackNet])

Three consecutive RGB frames stacked → *9-ch 640×360 input*, giving the network explicit motion context. Fully-sequential VGG encoder-decoder (*no skip connections*): 3 MaxPool stages halve spatial resolution to 80×45; 3 Upsample stages restore it.

#figure(
  canvas(length: 1.85em, {
    draw.set-style(
      mark: (scale: 0.42, fill: black),
      stroke: (thickness: 0.055em),
    )

    let boxes = (
      (0.0,  "Input\n9-ch",       rgb("#DCEEFB")),
      (2.1,  "Enc A\n64-ch ×2",  rgb("#84B9E3")),
      (4.2,  "Enc B\n128-ch ×2", rgb("#4A90D9")),
      (6.3,  "Enc C\n256→512ch", rgb("#1565C0")),
      (8.4,  "Dec A\n512→256ch", rgb("#388E3C")),
      (10.5, "Dec B\n256→64ch",  rgb("#66BB6A")),
      (12.4, "Out\n256-ch",      rgb("#FDD835")),
    )

    for i in range(boxes.len()) {
      let x   = boxes.at(i).at(0)
      let lbl = boxes.at(i).at(1)
      let col = boxes.at(i).at(2)
      draw.rect((x - 0.78, -0.38), (x + 0.78, 0.38), fill: col)
      draw.content((x, 0), text(size: 5.5pt, lbl))
      if i < boxes.len() - 1 {
        draw.line((x + 0.78, 0), (boxes.at(i + 1).at(0) - 0.78, 0), mark: (end: ">"))
      }
    }

    // MaxPool annotations below encoder arrows
    for (mid, lbl) in ((1.05, "↓MP"), (3.15, "↓MP"), (5.25, "↓MP")) {
      draw.content((mid, -0.58), text(size: 4pt, fill: gray, lbl))
    }
    // Upsample annotations above decoder arrows
    for (mid, lbl) in ((9.45, "↑US"), (11.45, "↑US")) {
      draw.content((mid, 0.56), text(size: 4pt, fill: gray, lbl))
    }
  }),
  caption: [TrackNet: sequential encoder-decoder. Block = Conv2d–BN–ReLU. MP=MaxPool, US=Upsample (×2). Kaiming init, Dropout2d p=0.3.]
)

*Heatmap target* — Gaussian centred on ball, discretised to 256 intensity classes (σ²=10):
$ y_(h,w) = floor(255 dot e^(-norm((h,w)-bold(p)^*)^2 slash 20)) $ <eq:heatmap>

*Focal loss* (γ=2) down-weights the dominant background class:
$ cal(L) = -(1\/N) sum_i (1-hat(p)_i)^2 log hat(p)_i $ <eq:focal>

#showcase(inset: 0.5em)[
  *Training:* Adam · lr=5×10⁻⁴ · batch 4 · 30 epochs · fp16 via HuggingFace Accelerate · grad clip ‖g‖≤1.0 · dropout p=0.3 · ImageNet normalisation
]

*Detection rule:* $max_(h,w) hat(y) > 2.55$. TP if predicted centroid within *5 px* of ground truth; TN if ball absent and suppressed.

== Court Detection — TrackNetCourt

#task-bar(green, [Court Detection — TrackNetCourt])

Identical encoder-decoder backbone as TrackNet; output head replaced with *15 heatmaps* (14 court boundary keypoints + centre, via diagonal intersection). Single RGB frame at 320×176. Weighted MSE loss (pos\_weight = 1000) to compensate for sparse keypoint activations (\<0.01% of pixels active per heatmap).

== Player Tracking — YOLOv8s

#task-bar(red, [Player Tracking — YOLOv8s])

Pretrained *YOLOv8s* fine-tuned end-to-end, single class *tennis-player*, multi-scale CIoU bounding-box regression. Evaluated with COCO mAP\@0.5 and mAP\@0.5:0.95.

#colbreak()

// ═══════════════════════════════════════════════════════════════════════════════
// COLUMN 3
// ═══════════════════════════════════════════════════════════════════════════════

= Results

== Ball Tracking

#grid(columns: (1fr,) * 4, gutter: 0.28em,
  metric-box("F1",        "0.927"),
  metric-box("Precision", "0.961"),
  metric-box("Recall",    "0.895"),
  metric-box("Accuracy",  "0.868"),
)
#v(0.22em)
#grid(columns: (1fr,) * 4, gutter: 0.28em,
  metric-box("TP",  "3137"),
  metric-box("FP",  "126"),
  metric-box("FN",  "370"),
  metric-box("TN",  "117"),
)
#v(0.22em)
#figure(
  image("figures/ball_error_dist.png", width: 100%),
  caption: [Positioning error on 3 253 detected balls (visible frames). Blue = TP (< 5 px); grey = FP. 3.4% of detections > 10 px not shown.]
)

== Court Detection

#grid(columns: (1fr,) * 3, gutter: 0.28em,
  metric-box("TP / total", "3 845 / 29 835", col: green),
  metric-box("Recall",     "12.9%",           col: green),
  metric-box("F1",         "0.129",           col: green),
)
#v(0.22em)
Partial convergence at lr = 10⁻⁵ (vs. 10⁻³ in the first run, which gave TP = 16). Remaining bottlenecks: weighted MSE with pos\_weight = 1000 is unstable when positive signal covers under 0.01% of pixels; 320×176 resolution limits spatial precision at the 7 px threshold.

== Player Tracking

#grid(columns: (1fr,) * 3, gutter: 0.28em,
  metric-box("mAP\@0.5",     "0.978", col: red),
  metric-box("mAP\@0.5:0.95","0.711", col: red),
  metric-box("F1",            "0.957", col: red),
)
#v(0.22em)
#figure(
  image("figures/player_pred.jpg", width: 90%),
  caption: [YOLOv8s detections on a validation batch. Per-box confidence scores shown for class *tennis-player*.]
)

== Conclusion

TrackNet achieves F1 = 0.927 with 96.4% of detections within 5 px, validating the 3-frame motion-context design. YOLOv8s fine-tuning reaches mAP\@0.5 = 0.978 with minimal task-specific engineering. Court keypoint regression failed to converge—lr schedule and focal heatmap loss are the primary remediation targets.

#v(0.3em)

= Future Work

#rect(
  fill: rgb("#FFF8F0"),
  stroke: (paint: amber.lighten(40%), dash: "dashed"),
  inset: 0.5em, radius: 0.3em, width: 100%,
  text(fill: amber, size: 0.8em)[
    - *Court detection:* lower lr to 10⁻⁵, switch to focal heatmap loss, increase resolution. #linebreak()
    - *Ball tracking:* trajectory smoothing, bounce detection. #linebreak()
    - *Unified pipeline:* fuse all three outputs into a spatiotemporal analyser. #linebreak()
    - *Datasets:* larger/more diverse broadcast video; semi-supervised annotation.
  ]
)

#v(0.3em)
#showcase(inset: 0.5em)[
  *Acknowledgements.* Models trained on the *Alan* HPC cluster (ULiège). Experiments tracked via Weights & Biases · `uliege-tennis-tracking`.
]
