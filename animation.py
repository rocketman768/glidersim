#!/usr/bin/env python3

import sim
from manim import *

class GliderThermalComparison(Scene):

    def construct(self):
        # Generate simulation datasets
        smooth = sim.simulate(30.0, sim.PILOT_SMOOTH)
        aggro = sim.simulate(30.0, sim.PILOT_AGGRESSIVE)

        # ----------------------------------------------------------------------
        # Upper Axes: Trajectory (x vs z)
        # ----------------------------------------------------------------------
        ax_top = Axes(
            x_range=[0, 1150, 200],
            y_range=[-20, 60, 20],
            x_length=10,
            y_length=2.5,
            axis_config={"include_numbers": True, "font_size": 18},
        ).to_edge(UP, buff=0.6)

        label_top = ax_top.get_axis_labels(
            x_label=Tex("Distance $x$ (ft)", font_size=20),
            y_label=Tex("Altitude $z$ (ft)", font_size=20),
        )

        # Thermal updraft visualization shading
        thermal_rect = Rectangle(
            width=ax_top.x_axis.number_to_point(984)[0]
            - ax_top.x_axis.number_to_point(0)[0],
            height=2.3,
            fill_color=BLUE,
            fill_opacity=0.15,
            stroke_width=0,
        ).move_to(ax_top.c2p(492, 20))

        thermal_text = Text(
            "Thermal Updraft Region (Allen Model)", font_size=14, color=BLUE_B
        ).move_to(thermal_rect.get_top() + DOWN * 0.2)

        # ----------------------------------------------------------------------
        # Lower Axes: Detrended Effective Energy Height
        # ----------------------------------------------------------------------
        ax_bot = Axes(
            x_range=[0, 1150, 200],
            y_range=[-30, 450, 100],
            x_length=10,
            y_length=2.8,
            axis_config={"include_numbers": True, "font_size": 18},
        ).to_edge(DOWN, buff=0.8)

        label_bot = ax_bot.get_axis_labels(
            x_label=Tex("Distance $x$ (ft)", font_size=20),
            y_label=Tex("Detrended $E_{h,eff}$ (ft)", font_size=20),
        )

        # Title and Legends
        title = Text(
            "Thermal Entry Pull-Up: Smooth (1.2g) vs. Aggressive (2.0g)",
            font_size=22,
            weight=BOLD,
        ).to_edge(UP, buff=0.15)

        leg_smooth = Line(ORIGIN, RIGHT * 0.4, color=TEAL, stroke_width=4)
        txt_smooth = Text("1.2g Smooth Pull", font_size=14, color=TEAL)
        leg_aggro = Line(ORIGIN, RIGHT * 0.4, color=ORANGE, stroke_width=4)
        txt_aggro = Text("2.0g Aggressive Pull", font_size=14, color=ORANGE)

        legend = VGroup(
            leg_smooth, txt_smooth, leg_aggro, txt_aggro
        ).arrange_in_grid(rows=1, buff=0.2)
        legend.next_to(ax_top, RIGHT, buff=-1.8).shift(UP * 0.3)

        self.add(
            title,
            ax_top,
            label_top,
            thermal_rect,
            thermal_text,
            ax_bot,
            label_bot,
            legend,
        )

        # ----------------------------------------------------------------------
        # Trajectory Curves
        # ----------------------------------------------------------------------
        print(len(smooth['x']))
        print(len(smooth['z']))
        graph_top_smooth = ax_top.plot_line_graph(
            x_values=smooth["x"],
            y_values=smooth["z"],
            z_values=np.zeros(len(smooth["x"])),
            add_vertex_dots=False,
            line_color=TEAL,
            stroke_width=3,
        )
        path_top_smooth = graph_top_smooth["line_graph"]

        graph_top_aggro = ax_top.plot_line_graph(
            x_values=aggro["x"],
            y_values=aggro["z"],
            z_values=np.zeros(len(aggro["x"])),
            add_vertex_dots=False,
            line_color=ORANGE,
            stroke_width=3,
        )
        path_top_aggro = graph_top_aggro["line_graph"]

        graph_bot_smooth = ax_bot.plot_line_graph(
            x_values=smooth["x"],
            y_values=smooth["E_h_detrended"],
            z_values=np.zeros(len(smooth["x"])),
            add_vertex_dots=False,
            line_color=TEAL,
            stroke_width=3,
        )
        path_bot_smooth = graph_bot_smooth["line_graph"]

        graph_bot_aggro = ax_bot.plot_line_graph(
            x_values=aggro["x"],
            y_values=aggro["E_h_detrended"],
            z_values=np.zeros(len(aggro["x"])),
            add_vertex_dots=False,
            line_color=ORANGE,
            stroke_width=3,
        )
        path_bot_aggro = graph_bot_aggro["line_graph"]

        # ----------------------------------------------------------------------
        # Render Animation
        # ----------------------------------------------------------------------
        self.play(
            Create(path_top_smooth),
            Create(path_top_aggro),
            Create(path_bot_smooth),
            Create(path_bot_aggro),
            run_time=6.0,
            rate_func=linear,
        )

        final_e_smooth = smooth["E_h_detrended"][-1]
        final_e_aggro = aggro["E_h_detrended"][-1]

        dot_end_smooth = Dot(
            ax_bot.c2p(smooth["x"][-1], final_e_smooth), color=TEAL
        )
        dot_end_aggro = Dot(
            ax_bot.c2p(aggro["x"][-1], final_e_aggro), color=ORANGE
        )

        txt_delta = Tex(
            r"$\Delta E_{h,\text{eff}} \approx 2.1\text{ ft}$",
            font_size=18,
            color=YELLOW,
        ).next_to(dot_end_smooth, RIGHT, buff=0.2)

        self.play(
            FadeIn(dot_end_smooth),
            FadeIn(dot_end_aggro),
            Write(txt_delta),
        )
        self.wait(2)