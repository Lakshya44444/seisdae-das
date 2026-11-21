"""Execute notebooks in IPython when the host forbids Jupyter kernel sockets."""

import io
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import Image, display
from IPython.utils.capture import capture_output


def show(*args, **kwargs):
    for number in plt.get_fignums():
        fig = plt.figure(number)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        display(Image(data=buf.getvalue()))
        plt.close(fig)


def execute(path, book):
    shell = InteractiveShell.instance()
    shell.reset(new_session=True)
    plt.show = show
    for i, cell in enumerate((c for c in book.cells if c.cell_type == "code"), 1):
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        result.raise_error()
        cell.execution_count = i
        cell.outputs = []
        for stream in ("stdout", "stderr"):
            content = getattr(captured, stream)
            if content:
                cell.outputs.append(
                    nbformat.v4.new_output("stream", name=stream, text=content)
                )
        for obj in captured.outputs:
            cell.outputs.append(
                nbformat.v4.new_output(
                    "display_data", data=obj.data, metadata=obj.metadata
                )
            )
    nbformat.validate(book)
    nbformat.write(book, path)
    print(f"Executed {path.name}", flush=True)


if __name__ == "__main__":
    for filename in sys.argv[1:] or sorted(
        (Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb")
    ):
        path = Path(filename).resolve()
        book = nbformat.read(path, as_version=4)
        previous = Path.cwd()
        os.chdir(path.parent)
        try:
            execute(path, book)
        finally:
            os.chdir(previous)
