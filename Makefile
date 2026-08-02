
all: previews

.PHONY: previews
previews:
	manim -pql animation.py GliderThermalComparison

.PHONY: highres
highres:
	manim -pqh --resolution 3840,2160 animation.py GliderThermalComparison