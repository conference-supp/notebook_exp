import re
import nbformat
import sys
import argparse
import shutil

TOC_TITLE = "Table of Contents"


def generate_toc_for_notebook(notebook_path, cmt_toc=True):
    # Make a backup of the current notebook
    backup_path = notebook_path + ".bak"
    shutil.copyfile(notebook_path, backup_path)

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as err:
        print(f"Error: {err}")
        sys.exit(1)

    toc = [
        f'<h1>{TOC_TITLE}<span class="tocSkip"></span></h1><ul>'
    ]  # Add a span element with class "tocSkip" to skip the TOC heading
    heading_counts = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
    }  # Assuming markdown headings levels 1-6

    if cmt_toc:
        # Check if the Table of Contents comment is present in any markdown cell
        for cell in nb["cells"]:
            if (
                cell["cell_type"] == "markdown"
                and cell["source"].strip() != ""
                and (
                    TOC_TITLE in cell["source"].strip().split("\n")[0]
                )  # Check the first line of the markdown cell
            ):
                # Convert the markdown cell to a code cell by adding a comment symbol
                cell["cell_type"] = "code"
                cell["source"] = "# " + cell["source"].replace("\n", "\n# ")
                cell["outputs"] = (
                    []
                )  # Add an empty outputs list to conform to code cell structure
                cell["execution_count"] = (
                    None  # Also set execution_count to None for completeness
                )

    # Add clickable links to headline titles in markdown cells
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            # print(type(cell))
            # print("=" * 20)
            # print(cell.keys())
            lines = cell["source"].split("\n")
            new_lines = []
            for line in lines:
                if not line.startswith("#"):
                    if not (
                        line.strip().startswith("<a id=")
                        and line.strip().startswith("</a>")
                    ):
                        new_lines.append(line)
                    continue

                level = line.count("#")
                title = line.strip("# ").strip().split("\n")[0]
                anchor_pos = re.search('<a id=".*"></a>', title)
                if anchor_pos:
                    title = re.findall(
                        r"^[0-9\.\s]*([^0-9]{1}.*)", title[: anchor_pos.start()]
                    )[0].strip()
                else:
                    title = re.findall(r"^[0-9\.\s]*([^0-9]{1}.*)", title)[0].strip()

                # Reset lower level counts when a higher level heading is encountered
                for l in range(level + 1, 7):
                    heading_counts[l] = 0
                heading_counts[level] += 1
                # Construct the numbering string
                numbering = (
                    ".".join(str(heading_counts[l]) for l in range(1, level + 1)) + "."
                )
                anchor = numbering + "-" + title.lower().replace(" ", "-")
                toc.append(
                    f'<li style="margin-left: {20 * (level - 1)}px;"><a href="#{anchor}">{numbering} {title}</a></li>'
                )

                # Add an anchor tag to the headline title
                new_lines.append(
                    '{} {} {}\n<a id="{}"></a>'.format(
                        "#" * level, numbering, title, anchor
                    )
                )
            cell["source"] = "\n".join(new_lines)

    toc.append("</ul>")

    toc_cell = nbformat.v4.new_markdown_cell("\n".join(toc))
    if "id" in toc_cell:
        # Avoid adding the "id" field to the cell, b/c jupyter notebook will raise an error
        del toc_cell["id"]
    # print(type(toc_cell))
    # print("=" * 20)
    # print(toc_cell.keys())
    nb["cells"].insert(0, toc_cell)

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Table of Contents added to {notebook_path}")


# # Example usage
# notebook_path = "<path_to_your_notebook>.ipynb"
# generate_toc_for_notebook(notebook_path)


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Generate Table of Contents for a Jupyter notebook"
    )

    # Add the notebook_path argument
    parser.add_argument("notebook_path", type=str, help="Path to the Jupyter notebook")

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the generate_toc_for_notebook function with the provided notebook_path
    generate_toc_for_notebook(args.notebook_path)
