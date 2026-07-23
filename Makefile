
all: previews

.PHONY: previews
previews:
	manim -pql animation.py GliderThermalComparison

