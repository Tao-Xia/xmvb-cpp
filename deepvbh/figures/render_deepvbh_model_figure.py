#!/usr/bin/env python3
"""Render a cleaner paper-style SVG figure for the DeepVBH model.

The layout is intentionally restrained:
1. panel (a): overall model pipeline
2. panel (b): pair-biased attention mechanism
3. panel (c): overlap-conditioned readout

The goal is a readable architecture figure suitable for manuscript refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class Palette:
  ink: str = "#1C1C1C"
  navy: str = "#273B67"
  navy_light: str = "#E9EEF8"
  green_light: str = "#EAF3E0"
  yellow_light: str = "#FFF3BF"
  orange_light: str = "#F8C8A8"
  lavender_light: str = "#DEE6F6"
  cream_light: str = "#FFF7D7"
  gray_light: str = "#F6F7FA"
  paper: str = "#FFFFFF"
  muted: str = "#666666"


class Svg:
  def __init__(self, width: int, height: int):
    self.width = width
    self.height = height
    self.p = Palette()
    self.parts: list[str] = []

  def add(self, fragment: str) -> None:
    self.parts.append(fragment)

  def rect(
      self,
      x: float,
      y: float,
      w: float,
      h: float,
      *,
      fill: str,
      stroke: str | None = None,
      sw: float = 3.0,
      rx: float = 16.0,
      dashed: bool = False,
  ) -> None:
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    self.add(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" '
        f'fill="{fill}" stroke="{stroke or self.p.navy}" stroke-width="{sw:.1f}"{dash} />'
    )

  def line(
      self,
      x1: float,
      y1: float,
      x2: float,
      y2: float,
      *,
      stroke: str | None = None,
      sw: float = 3.0,
      arrow: bool = False,
      dashed: bool = False,
  ) -> None:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    self.add(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke or self.p.navy}" stroke-width="{sw:.1f}" stroke-linecap="round"{marker}{dash} />'
    )

  def path(
      self,
      d: str,
      *,
      stroke: str | None = None,
      sw: float = 3.0,
      arrow: bool = False,
      fill: str = "none",
      dashed: bool = False,
  ) -> None:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    self.add(
        f'<path d="{d}" fill="{fill}" stroke="{stroke or self.p.navy}" stroke-width="{sw:.1f}" '
        f'stroke-linecap="round" stroke-linejoin="round"{marker}{dash} />'
    )

  def circle(
      self,
      cx: float,
      cy: float,
      r: float,
      *,
      fill: str,
      stroke: str | None = None,
      sw: float = 3.0,
  ) -> None:
    self.add(
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke or self.p.navy}" stroke-width="{sw:.1f}" />'
    )

  def text(
      self,
      x: float,
      y: float,
      text: str,
      *,
      size: int = 22,
      weight: str = "600",
      anchor: str = "start",
      fill: str | None = None,
      family: str = "'Times New Roman', Times, serif",
      italic: bool = False,
  ) -> None:
    style = "font-style:italic;" if italic else ""
    self.add(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill or self.p.ink}" font-family="{family}" style="{style}">{escape(text)}</text>'
    )

  def multiline(
      self,
      x: float,
      y: float,
      lines: list[str],
      *,
      size: int = 18,
      weight: str = "600",
      anchor: str = "middle",
      fill: str | None = None,
      family: str = "'Times New Roman', Times, serif",
      line_gap: int = 24,
  ) -> None:
    spans = []
    for i, line in enumerate(lines):
      dy = "0" if i == 0 else str(line_gap)
      spans.append(f'<tspan x="{x:.1f}" dy="{dy}">{escape(line)}</tspan>')
    self.add(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill or self.p.ink}" font-family="{family}">{"".join(spans)}</text>'
    )

  def plus(self, cx: float, cy: float, r: float = 16.0) -> None:
    self.circle(cx, cy, r, fill=self.p.paper)
    self.line(cx - 8, cy, cx + 8, cy, sw=3.0)
    self.line(cx, cy - 8, cx, cy + 8, sw=3.0)

  def defs(self) -> str:
    return f"""
<defs>
  <marker id="arrow" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="8" markerHeight="8" orient="auto">
    <path d="M 0 1 L 11 6 L 0 11 z" fill="{self.p.navy}" />
  </marker>
</defs>
"""

  def render(self) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
        f'viewBox="0 0 {self.width} {self.height}" role="img" aria-label="DeepVBH architecture figure">\n'
        f'{self.defs()}<rect width="{self.width}" height="{self.height}" fill="{self.p.paper}" />\n'
        + "\n".join(self.parts)
        + "\n</svg>\n"
    )


def box(svg: Svg, x: float, y: float, w: float, h: float, label: str, *, fill: str, size: int = 23) -> None:
  svg.rect(x, y, w, h, fill=fill, rx=14)
  lines = label.split("\n")
  start_y = y + h / 2 - (len(lines) - 1) * 11 + 8
  svg.multiline(x + w / 2, start_y, lines, size=size, line_gap=24)


def small_box(svg: Svg, x: float, y: float, w: float, h: float, label: str, *, fill: str | None = None, size: int = 18) -> None:
  svg.rect(x, y, w, h, fill=fill or svg.p.paper, rx=4, sw=2.6)
  svg.text(x + w / 2, y + h / 2 + 6, label, size=size, anchor="middle")


def draw_panel_a(svg: Svg) -> None:
  p = svg.p
  svg.text(28, 48, "(a)", size=30, weight="700")

  outer_x = 86
  outer_y = 22
  outer_w = 500

  svg.rect(outer_x, outer_y, outer_w, 162, fill="none", dashed=True, rx=28, sw=3.2)
  box(svg, 176, 50, 320, 56, "Input Tensors", fill=p.gray_light)
  box(svg, 176, 118, 320, 56, "Pair Feature Projector", fill=p.navy_light)
  svg.line(336, 106, 336, 118, arrow=True)

  stack_y = 218
  stack_h = 520
  svg.rect(outer_x, stack_y, outer_w, stack_h, fill=p.green_light, rx=34, sw=3.2)
  svg.rect(outer_x, stack_y, outer_w, stack_h, fill="none", dashed=True, rx=34, sw=3.0)

  box(svg, 160, 256, 352, 78, "Pair-Biased Attention", fill=p.yellow_light)
  box(svg, 160, 374, 352, 62, "Add & LayerNorm", fill=p.orange_light)
  box(svg, 160, 496, 352, 78, "Gated Feed-Forward", fill=p.cream_light)
  box(svg, 160, 614, 352, 62, "Add & LayerNorm", fill=p.lavender_light)

  svg.line(336, 174, 336, 256, arrow=True)
  svg.line(336, 334, 336, 374, arrow=True)
  svg.line(336, 436, 336, 496, arrow=True)
  svg.line(336, 574, 336, 614, arrow=True)

  svg.path("M 160 295 L 132 295 L 132 405 L 320 405", arrow=True)
  svg.plus(336, 405)
  svg.path("M 160 535 L 132 535 L 132 645 L 320 645", arrow=True)
  svg.plus(336, 645)
  svg.text(484, 716, "× L", size=34, italic=True, weight="500")

  svg.text(112, 770, "R and m condition every Transformer block", size=18, weight="500", fill=p.muted)
  svg.text(112, 796, "S_IJ is injected only at the readout stage", size=18, weight="500", fill=p.muted)

  bottom_y = 776
  svg.rect(outer_x, bottom_y, outer_w, 142, fill=p.cream_light, rx=22, sw=3.2)
  box(svg, 148, 816, 376, 62, "Overlap-Conditioned Readout", fill=p.cream_light)
  svg.line(336, 676, 336, 816, arrow=True)
  svg.line(336, 878, 336, 956, arrow=True)
  svg.text(336, 994, "Predicted matrix element  H_hat(I,J)", size=24, anchor="middle")

  svg.text(336, 88, "X_IJ,  R,  m,  S_IJ", size=20, weight="500", anchor="middle", italic=True)


def draw_panel_b(svg: Svg) -> None:
  p = svg.p
  svg.text(630, 48, "(b)", size=30, weight="700")

  x = 742
  y = 72
  w = 990
  h = 394
  svg.rect(x, y, w, h, fill=p.yellow_light, rx=0, sw=3.2)

  svg.text(840, 44, "token states  z^l", size=24, italic=True)
  svg.text(1492, 44, "pair relation  R", size=24, italic=True)

  token_box_y = 120
  small_box(svg, 820, token_box_y, 84, 50, "Q")
  small_box(svg, 928, token_box_y, 84, 50, "K")
  small_box(svg, 1036, token_box_y, 84, 50, "V")

  relation_box_x = 1484
  small_box(svg, relation_box_x, token_box_y, 106, 50, "Bias B(R)")
  small_box(svg, relation_box_x, 210, 106, 50, "Value U(R)")

  attn_x = 1170
  attn_y = 182
  svg.rect(attn_x, attn_y, 250, 74, fill=p.gray_light, rx=14, sw=3.0)
  svg.multiline(
      attn_x + 125,
      attn_y + 28,
      ["Attention weights", "softmax(QK^T / sqrt(d) + B(R))"],
      size=20,
      weight="600",
      line_gap=24,
  )

  svg.line(862, 88, 862, 120, arrow=True)
  svg.line(970, 88, 970, 120, arrow=True)
  svg.line(1078, 88, 1078, 120, arrow=True)
  svg.line(1537, 88, 1537, 120, arrow=True)
  svg.path("M 1537 88 L 1537 196", arrow=True)

  svg.path("M 862 170 L 862 220 L 1170 220", arrow=True)
  svg.path("M 970 170 L 970 220 L 1170 220", arrow=True)
  svg.path("M 1537 170 L 1537 220 L 1420 220", arrow=True)

  value_sum_y = 318
  svg.plus(1198, value_sum_y)
  svg.plus(1398, value_sum_y)
  svg.text(1198, value_sum_y + 7, "·", size=32, anchor="middle")
  svg.text(1398, value_sum_y + 7, "·", size=32, anchor="middle")

  svg.path("M 1295 256 L 1295 318 L 1214 318", arrow=True)
  svg.path("M 1295 256 L 1295 318 L 1382 318", arrow=True)
  svg.path("M 1078 170 L 1078 318 L 1182 318", arrow=True)
  svg.path("M 1537 260 L 1537 318 L 1414 318", arrow=True)

  small_box(svg, 1248, 350, 96, 46, "Sum", fill=p.paper)
  svg.path("M 1198 334 L 1198 373 L 1248 373", arrow=True)
  svg.path("M 1398 334 L 1398 373 L 1344 373", arrow=True)

  small_box(svg, 1140, 414, 312, 54, "Linear + residual + LayerNorm", fill=p.paper)
  svg.line(1296, 396, 1296, 414, arrow=True)
  svg.line(1296, 468, 1296, 524, arrow=True)
  svg.text(1296, 554, "updated token  z^(l+1)", size=26, anchor="middle", italic=True)

  svg.text(1458, 436, "Pair-Biased Attention", size=24)


def draw_panel_c(svg: Svg) -> None:
  p = svg.p
  svg.text(630, 572, "(c)", size=30, weight="700")

  x = 780
  top_y = 610

  small_box(svg, x + 254, top_y, 210, 54, "Masked Mean Pool", fill=p.paper)
  svg.text(x + 359, top_y - 18, "pooled token  z_bar", size=24, italic=True, anchor="middle")
  svg.line(x + 359, top_y - 6, x + 359, top_y, arrow=True)

  branch_y = 746
  left_x = x + 80
  right_x = x + 520
  small_box(svg, left_x, branch_y, 188, 56, "Linear + GELU", fill=p.navy_light)
  small_box(svg, right_x, branch_y, 188, 56, "Linear + GELU", fill=p.green_light)

  svg.path(f"M {x + 359} {top_y + 54} L {x + 359} 706 L {left_x + 94} 706 L {left_x + 94} {branch_y}", arrow=True)
  svg.text(right_x + 94, 698, "[S_IJ, |S_IJ|, sign(S_IJ)]", size=22, italic=True, anchor="middle")
  svg.line(right_x + 94, 710, right_x + 94, branch_y, arrow=True)

  small_box(svg, x + 250, 858, 220, 54, "Concat", fill=p.paper)
  svg.path(f"M {left_x + 94} {branch_y + 56} L {left_x + 94} 885 L {x + 250} 885", arrow=True)
  svg.path(f"M {right_x + 94} {branch_y + 56} L {right_x + 94} 885 L {x + 470} 885", arrow=True)

  svg.rect(x + 150, 950, 420, 78, fill=p.navy_light, rx=16, sw=3.0)
  svg.multiline(
      x + 360,
      979,
      ["Linear(128 -> 64) + GELU", "Linear(64 -> 1) + softplus"],
      size=22,
      weight="600",
      line_gap=26,
  )
  svg.line(x + 360, 912, x + 360, 950, arrow=True)

  svg.rect(x + 72, 1066, 576, 88, fill=p.cream_light, rx=18, sw=3.2)
  svg.multiline(
      x + 360,
      1098,
      ["H_hat(I,J) = - sign(S_IJ) * softplus(g_theta(z_bar, S_IJ))", "Overlap-conditioned scalar readout"],
      size=22,
      weight="600",
      line_gap=26,
  )
  svg.line(x + 360, 1028, x + 360, 1066, arrow=True)


def build_svg() -> str:
  svg = Svg(1760, 1180)
  draw_panel_a(svg)
  draw_panel_b(svg)
  draw_panel_c(svg)
  return svg.render()


def main() -> None:
  out = Path(__file__).with_name("deepvbh_model_iclr.svg")
  out.write_text(build_svg(), encoding="utf-8")
  print(out)


if __name__ == "__main__":
  main()
