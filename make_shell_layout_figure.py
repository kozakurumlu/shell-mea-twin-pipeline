"""Electrode-layout / orientation figure for the 16-channel folded shell MEA.

Two panels: (a) the unfolded device net in flat (u, v) coordinates, and
(b) the folded shell on the organoid sphere, both coloured by leaflet arm.
Pure plotting - geometry is imported from shell_mea_twin_pipeline so this
figure can never drift from the pipeline; no recording data is touched.

    .venv/bin/python make_shell_layout_figure.py

Writes paper/figs/shell_electrode_layout.pdf (vector) and .png.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from shell_mea_twin_pipeline import (_TWIN_TEX_RC, SHELL_R, SHELL_FLAT_COORDS,
                                     SHELL_ARM_OF, SHELL_ARM_COLORS,
                                     get_electrode_positions_3d)

ARM_AXIS = {'East': np.array([1.0, 0.0]), 'West': np.array([-1.0, 0.0]),
            'South': np.array([0.0, -1.0]), 'North': np.array([0.0, 1.0])}


def draw_net(ax):
    """(a) Unfolded net: four leaflet arms, four electrodes each."""
    for arm, axis in ARM_AXIS.items():
        color = SHELL_ARM_COLORS[arm]
        # thin centreline from the net origin out through the arm
        tip = 0.92 * axis
        ax.plot([0, tip[0]], [0, tip[1]], color=color, lw=0.9, alpha=0.55,
                zorder=1, solid_capstyle='round')
    for i in range(16):
        u, v = SHELL_FLAT_COORDS[i]
        arm = SHELL_ARM_OF[i]
        color = SHELL_ARM_COLORS[arm]
        ax.scatter(u, v, s=52, c=color, edgecolors='black', linewidths=0.7,
                   zorder=3)
        # label offset: push off-axis electrodes away from the centreline,
        # on-axis electrodes to a fixed perpendicular side
        axis = ARM_AXIS[arm]
        p = np.array([u, v])
        perp = p - np.dot(p, axis) * axis
        n = np.linalg.norm(perp)
        if n > 1e-9:
            off = 0.085 * perp / n
        else:
            off = 0.085 * np.array([-axis[1], axis[0]])
        ax.annotate(str(i), (u + off[0], v + off[1]), ha='center', va='center',
                    fontsize=8, zorder=4)
    ax.scatter(0, 0, s=12, c='0.35', zorder=2)  # net origin
    handles = [Line2D([], [], marker='o', ls='none', markersize=6,
                      markerfacecolor=SHELL_ARM_COLORS[a],
                      markeredgecolor='black', markeredgewidth=0.6, label=a)
               for a in ['East', 'West', 'South', 'North']]
    ax.legend(handles=handles, loc='lower left', fontsize=8, frameon=False,
              handletextpad=0.15, borderaxespad=0.0, labelspacing=0.25)
    ax.set_aspect('equal')
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.axis('off')
    ax.set_title('Unfolded shell net', fontsize=9)


# Screen-space label direction overrides for electrodes whose projection
# lands near the middle of the disc (radial push alone would collide).
LABEL_DIR_OVERRIDE = {8: (-0.55, 1.0), 15: (0.35, 1.0), 11: (0.15, -1.0),
                      14: (0.0, -1.0), 2: (1.0, -0.55), 5: (-1.0, 0.6),
                      13: (1.0, -0.05)}


def draw_folded(ax, elev=20.0, azim=-60.0):
    """(b) Folded shell: electrodes on the organoid sphere."""
    from mpl_toolkits.mplot3d import proj3d
    positions = get_electrode_positions_3d()

    uu = np.linspace(0, 2 * np.pi, 60)
    vv = np.linspace(0, np.pi, 40)
    xs = SHELL_R * np.outer(np.cos(uu), np.sin(vv))
    ys = SHELL_R * np.outer(np.sin(uu), np.sin(vv))
    zs = SHELL_R * np.outer(np.ones_like(uu), np.cos(vv))
    ax.plot_surface(xs, ys, zs, color='0.72', alpha=0.10, linewidth=0,
                    antialiased=True, shade=False)
    ax.plot_wireframe(xs, ys, zs, rstride=10, cstride=8, color='0.55',
                      alpha=0.08, linewidth=0.5)

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    lim = 1.28 * SHELL_R
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    # front/back split relative to the camera direction
    er, ar = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar),
                     np.sin(er)])
    M = ax.get_proj()
    cx, cy, _ = proj3d.proj_transform(0.0, 0.0, 0.0, M)
    for i in range(16):
        p = positions[i]['pos']
        color = positions[i]['color']
        front = float(np.dot(p, view)) > 0
        a = 1.0 if front else 0.35
        ax.scatter(*p, s=58, c=color, edgecolors='black', linewidths=0.7,
                   alpha=a, depthshade=False)
        # label: push outward from the disc centre in SCREEN space so no
        # label sits on the projected cluster; hand overrides for the few
        # electrodes that project near the centre
        px, py, _ = proj3d.proj_transform(*p, M)
        if i in LABEL_DIR_OVERRIDE:
            d = np.array(LABEL_DIR_OVERRIDE[i], dtype=float)
        else:
            d = np.array([px - cx, py - cy])
        n = np.linalg.norm(d)
        d = d / n if n > 1e-9 else np.array([1.0, 0.0])
        ax.annotate(str(i), xy=(px, py), xytext=(11.5 * d[0], 11.5 * d[1]),
                    textcoords='offset points', fontsize=8, ha='center',
                    va='center', alpha=min(1.0, a + 0.2))

    # small xyz triad so the orientation is unambiguous
    o = np.array([-2.45, -1.9, -2.3])
    for vec, name in [((1.0, 0, 0), 'x'), ((0, 1.0, 0), 'y'),
                      ((0, 0, 1.0), 'z')]:
        vec = np.array(vec)
        ax.quiver(*o, *vec, length=0.9, color='0.25', lw=0.9,
                  arrow_length_ratio=0.22)
        t = o + 1.3 * vec
        ax.text(t[0], t[1], t[2], f'${name}$', fontsize=8, ha='center',
                va='center', color='0.25')

    ax.set_axis_off()
    ax.set_title('Folded on the organoid', fontsize=9, pad=0)


def main():
    out_dir = os.path.join('paper', 'figs')
    os.makedirs(out_dir, exist_ok=True)
    with plt.rc_context(_TWIN_TEX_RC):
        fig = plt.figure(figsize=(7.0, 3.15))
        ax_a = fig.add_subplot(1, 2, 1)
        ax_b = fig.add_subplot(1, 2, 2, projection='3d')
        draw_net(ax_a)
        draw_folded(ax_b)
        for ax, letter in [(ax_a, '(a)'), (ax_b, '(b)')]:
            ax.text2D(0.0, 1.0, letter, transform=ax.transAxes, fontsize=10,
                      fontweight='bold', ha='left', va='top') \
                if hasattr(ax, 'text2D') else \
                ax.text(0.0, 1.0, letter, transform=ax.transAxes, fontsize=10,
                        fontweight='bold', ha='left', va='top')
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02,
                            wspace=0.05)
        for ext in ['pdf', 'png']:
            path = os.path.join(out_dir, f'shell_electrode_layout.{ext}')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f'Wrote {path} ({os.path.getsize(path)} bytes)')
        plt.close(fig)


if __name__ == '__main__':
    main()
