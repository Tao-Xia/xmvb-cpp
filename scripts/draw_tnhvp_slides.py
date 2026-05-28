#!/usr/bin/env python3
"""Generate 3 PPT-quality diagrams for the TNHVP algorithm."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Global style ──────────────────────────────────────────────────────
BOX_ALPHA = 0.92
FONT_FAMILY = 'serif'
plt.rcParams.update({
    'font.family': FONT_FAMILY,
    'font.size': 13,
    'mathtext.fontset': 'cm',
})

# Color palette
C_INPUT    = '#E8F4FD'   # light blue  – input data
C_OP       = '#FFF3E0'   # light orange – operations
C_OUTPUT   = '#E8F5E9'   # light green – outputs
C_CORE     = '#FCE4EC'   # light red   – direct core
C_UPSTREAM = '#F3E5F5'   # light purple – upstream
C_OUTER    = '#E0F7FA'   # light cyan  – outer response
C_FORMULA  = '#FFFDE7'   # light yellow – formulas
C_ACCENT   = '#1565C0'   # dark blue   – arrows / emphasis
C_ITER     = '#FFF9C4'   # light yellow – iteration


def draw_box(ax, x, y, w, h, text, color, fontsize=11, bold=False):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor='#555555', linewidth=1.2, alpha=BOX_ALPHA,
        zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w / 2, y + h / 2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight=weight, zorder=3,
            wrap=True)
    return (x + w / 2, y, x + w / 2, y + h)  # (bot_cx, bot_y, top_cx, top_y)


def draw_arrow(ax, x1, y1, x2, y2, color=C_ACCENT, lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=4)


def draw_label(ax, x, y, text, fontsize=9, color='#333333'):
    ax.text(x, y, text, fontsize=fontsize, color=color,
            ha='center', va='center', zorder=5,
            fontstyle='italic')


# ======================================================================
#  Slide 1 – LOT Construction
# ======================================================================
def draw_slide1():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 6.5)
    ax.axis('off')
    fig.suptitle(r'$\mathbf{Slide\ 1}$: Local Orbital Tangent (LOT) Construction',
                 fontsize=16, fontweight='bold', y=0.97)

    # ── Top row: input → operations → output ──
    # (1) Input: reduced direction
    bx1 = draw_box(ax, 0.3, 4.5, 2.4, 1.2,
                    r'$\mathbf{d}_{\mathrm{reduced}}$' + '\n(reduced direction)',
                    C_INPUT, fontsize=12)

    # (2) expand_step
    bx2 = draw_box(ax, 4.0, 4.5, 2.6, 1.2,
                    'expand_step()\n' + r'$\mathbf{d}_{\mathrm{reduced}} \to \Delta\mathbf{C}_{\mathrm{packed}}$',
                    C_OP, fontsize=11)

    # (3) Build LOT context
    bx3 = draw_box(ax, 7.8, 4.5, 3.2, 1.2,
                    'build_dense_orbital\n_tangent_context()\n(dense LOT field)',
                    C_OP, fontsize=10)

    # (4) Output: directional result
    bx4 = draw_box(ax, 3.5, 2.2, 3.5, 1.3,
                    'orbital_preparation\n_directional_result\n'
                    r'$\delta A_{\mathrm{aux}},\ \delta D_{\mathrm{inactive}}$',
                    C_OUTPUT, fontsize=10)

    # Arrows
    draw_arrow(ax, 2.7, 5.1, 4.0, 5.1)
    draw_arrow(ax, 6.6, 5.1, 7.8, 5.1)
    draw_arrow(ax, 9.4, 4.5, 5.25, 3.5)

    # ── Bottom: key concepts ──
    # Nonredundant space note
    draw_box(ax, 0.2, 0.5, 3.5, 1.3,
             r'NonredundantOrbitalSpace' + '\n'
             r'$U_p$: tangent basis from $B_p$'
             '\n(no sphere constraint)',
             C_FORMULA, fontsize=10)

    # Update formula
    draw_box(ax, 5.5, 0.5, 5.5, 1.3,
             r'$\mathbf{C}_{\mathrm{new}} = \mathbf{C}_{\mathrm{old}} + \alpha \cdot \Delta\mathbf{C}$'
             '\n(Linear addition update)'
             '\n' + r'$\alpha$: step size (trust region / line search)',
             C_FORMULA, fontsize=11, bold=True)

    # Labels on arrows
    draw_label(ax, 3.35, 5.45, 'tangent basis $U_p$', fontsize=9)
    draw_label(ax, 7.2, 5.45, 'dense reshape', fontsize=9)
    draw_label(ax, 8.0, 3.7, 'directional\nquantities', fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig('/pool1/home/xiatao/project/xmvb-cpp/docs/tnhvp_slide1_lot.png', dpi=200, bbox_inches='tight')
    print("Slide 1 saved.")
    plt.close(fig)


# ======================================================================
#  Slide 2 – HVP Computation Pipeline
# ======================================================================
def draw_slide2():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-0.5, 8.5)
    ax.axis('off')
    fig.suptitle(r'$\mathbf{Slide\ 2}$: Hessian-Vector Product (HVP) Computation Pipeline',
                 fontsize=16, fontweight='bold', y=0.97)

    # ── Input at top ──
    draw_box(ax, 4.0, 7.2, 4.0, 0.9,
             'apply_reduced(' + r'$\mathbf{d}_{\mathrm{reduced}}$' + ')',
             C_INPUT, fontsize=12, bold=True)

    # ── Three branches ──
    branch_y = 4.0
    branch_h = 2.8
    branch_w = 3.5

    # Branch 1: Direct Core Response
    bx1_x = 0.2
    draw_box(ax, bx1_x, branch_y, branch_w, branch_h,
             '(1) Direct Core Response\n'
             '─────────────────\n'
             'AO eff. 1e directional op\n'
             r'$\delta F_{11} \cdot A_{\mathrm{act}}$'
             '\n'
             '2e adjoint HVP\n'
             r'$(\delta A^T G_{\mathrm{sso}} A)$'
             '\n'
             'backpropagate to\n'
             'orbital gradient',
             C_CORE, fontsize=9.5)

    # Branch 2: Fixed Upstream Pullback
    bx2_x = 4.5
    draw_box(ax, bx2_x, branch_y, branch_w, branch_h,
             '(2) Fixed Upstream\n'
             '   Pullback\n'
             '─────────────────\n'
             'apply_fixed_upstream\n'
             '_orbital_pullback\n'
             '_direction()\n'
             '\n'
             r'$\nabla_{\mathbf{C}} \langle g^{\mathrm{fix}}, \delta\mathbf{C}\rangle$',
             C_UPSTREAM, fontsize=9.5)

    # Branch 3: Outer Response
    bx3_x = 8.8
    draw_box(ax, bx3_x, branch_y, branch_w + 0.8, branch_h,
             '(3) Outer Response\n'
             '─────────────────\n'
             'directional integrals\n'
             '  → projected directional\n'
             '    structure matrices\n'
             '  → selected-state eigen\n'
             '    response\n'
             '  → active-space gradient\n'
             '  → orbital pullback',
             C_OUTER, fontsize=9.5)

    # ── Arrows from top to branches ──
    top_cx = 6.0
    draw_arrow(ax, top_cx, 7.2, bx1_x + branch_w / 2, branch_y + branch_h)
    draw_arrow(ax, top_cx, 7.2, bx2_x + branch_w / 2, branch_y + branch_h)
    draw_arrow(ax, top_cx, 7.2, bx3_x + (branch_w + 0.8) / 2, branch_y + branch_h)

    # ── Summation node ──
    sum_y = 1.5
    draw_box(ax, 3.5, sum_y, 5.5, 1.5,
             r'$\nabla_{\mathbf{C}}^{(1)} + \nabla_{\mathbf{C}}^{(2)} + \nabla_{\mathbf{C}}^{(3)}$'
             '\n→ combined_core_orbital_value_gradient',
             C_OUTPUT, fontsize=11, bold=True)

    # Arrows from branches to sum
    draw_arrow(ax, bx1_x + branch_w / 2, branch_y, 5.0, sum_y + 1.5)
    draw_arrow(ax, bx2_x + branch_w / 2, branch_y, 6.25, sum_y + 1.5)
    draw_arrow(ax, bx3_x + (branch_w + 0.8) / 2, branch_y, 7.5, sum_y + 1.5)

    # Labels
    draw_label(ax, 3.2, 6.6, 'core\nHessian', fontsize=8)
    draw_label(ax, 6.0, 6.6, 'retraction\neffect', fontsize=8)
    draw_label(ax, 9.8, 6.6, 'wavefunction\nresponse', fontsize=8)

    # ── Output arrow ──
    draw_box(ax, 4.0, 0.0, 4.5, 1.0,
             'gather_from_full + project_reduced_gradient\n'
             r'→ $\mathbf{r}_{\mathrm{reduced}} = H \cdot \mathbf{d}_{\mathrm{reduced}}$',
             C_FORMULA, fontsize=10)
    draw_arrow(ax, 6.25, sum_y, 6.25, 1.0)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig('/pool1/home/xiatao/project/xmvb-cpp/docs/tnhvp_slide2_hvp.png', dpi=200, bbox_inches='tight')
    print("Slide 2 saved.")
    plt.close(fig)


# ======================================================================
#  Slide 3 – Truncated Newton Solve & Orbital Update
# ======================================================================
def draw_slide3():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.3, 6.5)
    ax.axis('off')
    fig.suptitle(r'$\mathbf{Slide\ 3}$: Truncated Newton Solve & Orbital Update',
                 fontsize=16, fontweight='bold', y=0.97)

    # ── Outer: Truncated Newton loop ──
    # Draw a large rounded rectangle for the CG loop
    loop_box = FancyBboxPatch(
        (0.3, 0.5), 11.2, 4.8,
        boxstyle="round,pad=0.3",
        facecolor='#FAFAFA', edgecolor=C_ACCENT, linewidth=2.0,
        linestyle='--', alpha=0.5, zorder=1)
    ax.add_patch(loop_box)
    ax.text(0.8, 5.0, 'CG / Lanczos Krylov iteration',
            fontsize=11, color=C_ACCENT, fontstyle='italic', zorder=5)

    # Inside the loop:
    # Step 1: Apply HVP
    draw_box(ax, 0.8, 3.5, 3.2, 1.2,
             r'$v \mapsto Hv$' + '\napply_reduced()\n(HVP)',
             C_OP, fontsize=11, bold=True)

    # Step 2: CG update
    draw_box(ax, 4.8, 3.5, 2.8, 1.2,
             'CG step\n'
             r'$p_{k+1} = p_k + \alpha_k d_k$',
             C_ITER, fontsize=11)

    # Step 3: Convergence check
    draw_box(ax, 8.4, 3.5, 2.6, 1.2,
             'Converged?\n'
             r'$\|r_k\| < \epsilon$',
             C_INPUT, fontsize=11)

    # Arrows inside loop
    draw_arrow(ax, 4.0, 4.1, 4.8, 4.1)
    draw_arrow(ax, 7.6, 4.1, 8.4, 4.1)

    # Loop-back arrow
    draw_arrow(ax, 9.7, 3.5, 9.7, 1.8)
    draw_arrow(ax, 9.7, 1.8, 2.4, 1.8)
    draw_arrow(ax, 2.4, 1.8, 2.4, 3.5)
    draw_label(ax, 6.0, 1.5, 'No: continue iteration', fontsize=9, color='#B71C1C')

    # ── After loop: orbital update ──
    draw_box(ax, 1.0, 0.0, 3.5, 1.0,
             'expand_step(' + r'$\mathbf{p}^*$' + ')\n'
             r'$\Delta\mathbf{C} = U_p \cdot \mathbf{p}^*$',
             C_OP, fontsize=10)

    draw_box(ax, 5.2, 0.0, 3.5, 1.0,
             'retract_step()\n'
             r'$\mathbf{C}_{\mathrm{new}} = \mathbf{C}_{\mathrm{old}} + \alpha \cdot \Delta\mathbf{C}$',
             C_OUTPUT, fontsize=10, bold=True)

    draw_box(ax, 9.3, 0.0, 2.2, 1.0,
             'Next SCF\niteration',
             C_FORMULA, fontsize=10)

    # Arrows from convergence to update
    draw_arrow(ax, 9.7, 3.5, 2.75, 1.0, color='#2E7D32', lw=2.0)
    draw_label(ax, 5.5, 2.5, 'Yes: solve converged', fontsize=9, color='#2E7D32')

    draw_arrow(ax, 4.5, 0.5, 5.2, 0.5)
    draw_arrow(ax, 8.7, 0.5, 9.3, 0.5)

    # Key note at bottom
    ax.text(6.0, -0.15,
            r'Cache: accepted-point intermediates reused across all Krylov iterations (no recompute)',
            fontsize=9, color='#666666', ha='center', va='center',
            fontstyle='italic')

    fig.tight_layout(rect=[0, -0.05, 1, 0.94])
    fig.savefig('/pool1/home/xiatao/project/xmvb-cpp/docs/tnhvp_slide3_solve.png', dpi=200, bbox_inches='tight')
    print("Slide 3 saved.")
    plt.close(fig)


# ======================================================================
if __name__ == '__main__':
    draw_slide1()
    draw_slide2()
    draw_slide3()
    print("All 3 slides generated in docs/")
