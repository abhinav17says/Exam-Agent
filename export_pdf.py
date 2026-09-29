"""Export a PDF locally with the notebook's default Docling converter."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to the PDF filename with .md extension")
    args = parser.parse_args()
    output = args.output or args.pdf.with_suffix(".md")
    if output.exists():
        parser.error(f"Output already exists: {output}. Choose another --output path.")
    from exam_agent_backend import convert_pdf_to_markdown

    markdown = convert_pdf_to_markdown(args.pdf.read_bytes())
    if not markdown.strip():
        parser.error("Docling found no readable content in this PDF.")
    with output.open("x", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"Upload {output} to the hosted Exam Agent.")


if __name__ == "__main__":
    main()
