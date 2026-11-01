"""Build laser_reservoir.html from template.html and core.js.

Run from the project folder:  python3 visualization/build.py
"""
import pathlib

here = pathlib.Path(__file__).parent
template = (here / "template.html").read_text()
core = (here / "core.js").read_text()

assert template.count("/*CORE*/") == 1
(here / "laser_reservoir.html").write_text(template.replace("/*CORE*/", core))
print("wrote", here / "laser_reservoir.html")
