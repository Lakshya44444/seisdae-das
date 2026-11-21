"""Render release notes for students from the same measurements as the report."""
from string import Template
from .report import TEMPLATES, evidence
from ..data import ROOT


def render(name):
    return Template((TEMPLATES / name).read_text()).substitute(evidence())


def readme():
    return render("readme.md.in")


def viva():
    return render("viva.md.in")


if __name__ == "__main__":
    (ROOT / "README.md").write_text(readme())
    (ROOT / "VIVA_NOTES.md").write_text(viva())
