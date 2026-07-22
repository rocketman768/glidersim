#!/usr/bin/env python3

import sim
import math
from manim import *

class GliderThermalComparison(Scene):

    def construct(self):
        # Generate simulation datasets
        smoothPilot = sim.PILOT_CHEATER
        aggroPilot = sim.PILOT_SMOOTH
        smooth = sim.simulate(30.0, smoothPilot)
        aggro = sim.simulate(30.0, aggroPilot)

        x_min = 0
        x_max = max(smooth['x'][-1], aggro['x'][-1])

        z_min = 0
        z_max = 0
        for (z0, z1) in zip(smooth['z'], aggro['z']):
            z_min = min(min(z_min, z0), z1)
            z_max = max(max(z_max, z0), z1)
        
        e_min = smooth["E_h_detrended"][0]
        e_max = smooth["E_h_detrended"][0]
        for (e0, e1) in zip(smooth["E_h_detrended"], aggro["E_h_detrended"]):
            e_min = min(min(e_min, e0), e1)
            e_max = max(max(e_max, e0), e1)

        # ----------------------------------------------------------------------
        # Upper Axes: Trajectory (x vs z)
        # ----------------------------------------------------------------------
        ax_top = Axes(
            x_range=[0, x_max, 200],
            y_range=[z_min, z_max, 20],
            x_length=10,
            y_length=2.5,
            axis_config={"include_numbers": True, "font_size": 18},
        ).to_edge(UP, buff=0.6)

        label_top = ax_top.get_axis_labels(
            x_label=Tex("Distance (m)", font_size=20),
            y_label=Tex("Altitude (m)", font_size=20),
        )

        # ----------------------------------------------------------------------
        # Secondary Right Y-Axis & Thermal Profile w_allen(x)
        # ----------------------------------------------------------------------
        W_MAX = 5.0
        W_SCALE = z_max / W_MAX

        # Right Y-axis line at thermal center
        w_center = sim.thermalCenter
        right_axis_line = Line(
            ax_top.c2p(w_center, 0),
            ax_top.c2p(w_center, z_max),
            stroke_width=2,
            color=BLUE_B,
        )

        # Right Y-axis tick labels
        w_ticks = VGroup()
        for val in range(0, int(math.ceil(W_MAX)), 1):
            z_mapped = val * W_SCALE
            tick = Line(
                ax_top.c2p(w_center, z_mapped),
                ax_top.c2p(w_center, z_mapped) + RIGHT * 0.08,
                stroke_width=2,
                color=BLUE_B,
            )
            lbl = Text(f"{val}", font_size=14, color=BLUE_B).next_to(
                tick, RIGHT, buff=0.08
            )
            w_ticks.add(tick, lbl)

        label_top_w = Tex("$w_{\\text{therm}}$ (m/s)", font_size=18, color=BLUE_B)
        label_top_w.next_to(right_axis_line, UP + RIGHT, buff=0.05)

        # Generate w(x) points
        x_m_vals = np.linspace(sim.thermalCenter - 2.0 * sim.thermalWidth, sim.thermalCenter + 2.0 * sim.thermalWidth, 200)
        w_m_vals = [sim.w(xm) for xm in x_m_vals]

        w_pts = [ax_top.c2p(xf, wf * W_SCALE) for xf, wf in zip(x_m_vals, w_m_vals)]
        w_curve = VMobject(color=BLUE_B, stroke_width=2).set_points_smoothly(w_pts)

        # Fill under the updraft bell curve
        baseline_pts = [ax_top.c2p(xf, 0) for xf in reversed(w_m_vals)]
        w_fill = Polygon(
            *w_pts,
            *baseline_pts,
            fill_color=BLUE,
            fill_opacity=0.2,
            stroke_width=0,
        )

        # ----------------------------------------------------------------------
        # Lower Axes: Detrended Effective Energy Height
        # ----------------------------------------------------------------------
        e0 = aggro["E_h_detrended"][0]
        # e - e0
        ax_bot = Axes(
            x_range=[0, x_max, 200],
            y_range=[e_min - e0, e_max - e0, 1],
            x_length=10,
            y_length=2.8,
            axis_config={"include_numbers": True, "font_size": 18},
        ).to_edge(DOWN, buff=0.8)

        label_bot = ax_bot.get_axis_labels(
            x_label=Tex("Distance (m)", font_size=20),
            y_label=Tex("Racing Energy (m)", font_size=20),
        )

        # Title and Legends
        title = Text(
            "Dolphin Pull-Up",
            font_size=22,
            weight=BOLD,
        ).to_edge(UP, buff=0.15)

        leg_smooth = Line(ORIGIN, RIGHT * 0.4, color=TEAL, stroke_width=4)
        txt_smooth = Text(smoothPilot.name, font_size=14, color=TEAL)
        leg_aggro = Line(ORIGIN, RIGHT * 0.4, color=ORANGE, stroke_width=4)
        txt_aggro = Text(aggroPilot.name, font_size=14, color=ORANGE)

        legend = VGroup(
            leg_smooth, txt_smooth, leg_aggro, txt_aggro
        ).arrange_in_grid(rows=2, buff=0.2)
        legend.next_to(ax_top, LEFT, buff=-1.8).shift(DOWN * 0.7 + RIGHT * 0.7)

        self.add(
            title,
            ax_top,
            label_top,
            w_fill,
            w_curve,
            right_axis_line,
            w_ticks,
            label_top_w,
            ax_bot,
            label_bot,
            legend,
        )

        # ----------------------------------------------------------------------
        # Trajectory Curves
        # ----------------------------------------------------------------------
        pts_top_smooth = [ax_top.c2p(x, z) for x, z in zip(smooth["x"], smooth["z"])]
        pts_top_aggro = [ax_top.c2p(x, z) for x, z in zip(aggro["x"], aggro["z"])]

        pts_bot_smooth = [ax_bot.c2p(x, e - smooth["E_h_detrended"][0]) for x, e in zip(smooth["x"], smooth["E_h_detrended"])]
        pts_bot_aggro = [ax_bot.c2p(x, e - aggro["E_h_detrended"][0]) for x, e in zip(aggro["x"], aggro["E_h_detrended"])]

        path_top_smooth = VMobject(color=TEAL, stroke_width=3).set_points_smoothly(pts_top_smooth)
        path_top_aggro = VMobject(color=ORANGE, stroke_width=3).set_points_smoothly(pts_top_aggro)

        path_bot_smooth = VMobject(color=TEAL, stroke_width=3).set_points_smoothly(pts_bot_smooth)
        path_bot_aggro = VMobject(color=ORANGE, stroke_width=3).set_points_smoothly(pts_bot_aggro)

        max_idx = len(smooth["x"]) - 1
        n_tracker = ValueTracker(0)

        # 4. Vector attached to curve tip
        arrow = Arrow(buff=0, color=RED, stroke_width=6, tip_length=0.2, max_stroke_width_to_length_ratio=9999, max_tip_length_to_length_ratio=1)
        def update_arrow(mob):
            n = int(n_tracker.get_value())
            n = min(n, max_idx)

            tip_pos = path_top_smooth.get_end()

            # Convert scene position back to (x, z) axis coordinates
            x_val, z_val = ax_top.p2c(tip_pos)[:2]

            # Interpolate dz corresponding to current x_val
            #dz_val = np.interp(x_val, x, dz)
            dz_val = 20.0 * (smooth["n"][n] - 1.0)

            # Redraw arrow at tip
            mob.become(
                Arrow(
                    start=tip_pos,
                    end=ax_top.c2p(x_val, z_val + dz_val),
                    buff=0,
                    color=RED,
                    stroke_width=6, tip_length=0.2, max_stroke_width_to_length_ratio=9999, max_tip_length_to_length_ratio=1
                )
            )
        arrow.add_updater(update_arrow)
        self.add(arrow)

        # ----------------------------------------------------------------------
        # Animation Execution
        # ----------------------------------------------------------------------
        PLAYBACK_FACTOR = 4.0
        tEnd = max(smooth['t'][-1], aggro['t'][-1])
        self.play(
            Create(path_top_smooth),
            Create(path_top_aggro),
            Create(path_bot_smooth),
            Create(path_bot_aggro),
            n_tracker.animate.set_value(max_idx),
            run_time=tEnd / PLAYBACK_FACTOR,
            rate_func=linear,
        )
        arrow.remove_updater(update_arrow)

        final_e_smooth = smooth["E_h_detrended"][-1] - e0
        final_e_aggro = aggro["E_h_detrended"][-1] - e0
        diff_e = final_e_smooth - final_e_aggro

        dot_end_smooth = Dot(ax_bot.c2p(smooth["x"][-1], final_e_smooth), color=TEAL)
        dot_end_aggro = Dot(ax_bot.c2p(aggro["x"][-1], final_e_aggro), color=ORANGE)

        txt_delta = Tex(
            r"$\Delta E_{h,\text{eff}} \approx " + f"{diff_e:.1f}" + r" \text{ m}$",
            font_size=18,
            color=YELLOW,
        ).next_to(dot_end_smooth, RIGHT, buff=0.2)

        self.play(
            FadeIn(dot_end_smooth),
            FadeIn(dot_end_aggro),
            Write(txt_delta),
        )
        self.wait(2)