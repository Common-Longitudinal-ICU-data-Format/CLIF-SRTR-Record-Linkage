#!/usr/bin/env python
"""
Jupyter Notebook to Python Script Converter

This script converts Jupyter notebooks (.ipynb) to executable Python scripts (.py).
- Markdown cells are converted to multi-line comments
- Code cells are preserved as-is
- Cell outputs are ignored
- Headers are formatted nicely in comments

Usage:
    python notebook_to_script.py <notebook_file> [output_file]

Examples:
    python notebook_to_script.py 01_potential_donor_identifier.ipynb
    python notebook_to_script.py notebook.ipynb script.py
"""

import json
import sys
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any


def format_markdown_as_comment(text: str, comment_style: str = "block") -> str:
    """
    Convert markdown text to Python comments.

    Args:
        text: Markdown text to convert
        comment_style: Either "block" for multi-line comments or "hash" for # comments

    Returns:
        Formatted comment string
    """
    if not text.strip():
        return ""

    lines = text.split('\n')

    if comment_style == "block":
        # Use triple quotes for block comments
        # Escape any existing triple quotes in the text
        text_escaped = text.replace('"""', '\\"\\"\\"')
        return f'"""\n{text_escaped}\n"""'
    else:
        # Use # for line comments
        commented_lines = []
        for line in lines:
            if line.strip():
                commented_lines.append(f"# {line}")
            else:
                commented_lines.append("#")
        return '\n'.join(commented_lines)


def is_header_markdown(lines: List[str]) -> bool:
    """Check if markdown cell appears to be a header/section marker."""
    if not lines:
        return False

    # Check if first line is a markdown header
    first_line = lines[0].strip()
    if first_line.startswith('#'):
        return True

    # Check if it's a short cell (likely a header)
    if len(lines) <= 2 and all(len(line.strip()) < 80 for line in lines):
        return True

    return False


def convert_notebook_to_script(notebook_path: str, output_path: str = None,
                              add_shebang: bool = True,
                              add_encoding: bool = True) -> None:
    """
    Convert a Jupyter notebook to a Python script.

    Args:
        notebook_path: Path to the .ipynb file
        output_path: Optional path for output .py file (defaults to same name as notebook)
        add_shebang: Whether to add #!/usr/bin/env python at the top
        add_encoding: Whether to add encoding declaration
    """
    # Read the notebook
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Determine output path
    if output_path is None:
        output_path = notebook_path.with_suffix('.py')
    else:
        output_path = Path(output_path)

    # Build the script content
    script_lines = []

    # Add shebang and encoding if requested
    if add_shebang:
        script_lines.append("#!/usr/bin/env python")
    if add_encoding:
        script_lines.append("# -*- coding: utf-8 -*-")

    if add_shebang or add_encoding:
        script_lines.append("")

    # Add header docstring
    script_lines.append('"""')
    script_lines.append(f"Generated from: {notebook_path.name}")
    script_lines.append("")
    script_lines.append("This script was automatically generated from a Jupyter notebook.")
    script_lines.append("Markdown cells have been converted to comments.")
    script_lines.append('"""')
    script_lines.append("")

    # Process cells
    cells = notebook.get('cells', [])

    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type', '')
        source = cell.get('source', [])

        # Handle source as either list or string
        if isinstance(source, str):
            source_text = source
        else:
            source_text = ''.join(source)

        # Skip empty cells
        if not source_text.strip():
            continue

        if cell_type == 'markdown':
            # Convert markdown to comments
            lines = source_text.split('\n')

            # Check if this is a header/section markdown
            if is_header_markdown(lines):
                # Format as a section header with emphasis
                script_lines.append("")
                script_lines.append("#" * 80)
                for line in lines:
                    if line.strip():
                        # Remove markdown header symbols for cleaner look
                        clean_line = line.strip().lstrip('#').strip()
                        script_lines.append(f"# {clean_line}")
                script_lines.append("#" * 80)
                script_lines.append("")
            else:
                # Regular markdown - convert to block comment
                script_lines.append("")
                comment = format_markdown_as_comment(source_text, comment_style="block")
                if comment:
                    script_lines.append(comment)
                    script_lines.append("")

        elif cell_type == 'code':
            # Add code as-is
            # Remove any leading/trailing whitespace but preserve internal formatting
            code_lines = source_text.rstrip().split('\n')

            # Skip cells that are just comments or empty
            if all(line.strip().startswith('#') or not line.strip() for line in code_lines):
                continue

            # Add the code
            script_lines.extend(code_lines)
            script_lines.append("")  # Add blank line after code cell

    # Clean up multiple consecutive blank lines
    final_lines = []
    prev_blank = False
    for line in script_lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        final_lines.append(line)
        prev_blank = is_blank

    # Ensure file ends with a newline
    if final_lines and final_lines[-1] != "":
        final_lines.append("")

    # Write the output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_lines))

    print(f"✓ Converted: {notebook_path.name} -> {output_path.name}")
    print(f"  Output saved to: {output_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert Jupyter notebooks to Python scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s notebook.ipynb
      Convert notebook.ipynb to notebook.py

  %(prog)s notebook.ipynb output.py
      Convert notebook.ipynb to output.py

  %(prog)s *.ipynb --suffix _converted
      Convert all notebooks with suffix (e.g., notebook_converted.py)

  %(prog)s notebook.ipynb --no-shebang --no-encoding
      Convert without shebang and encoding declarations
        """
    )

    parser.add_argument(
        'notebook',
        nargs='+',
        help='Path to notebook file(s) to convert'
    )

    parser.add_argument(
        'output',
        nargs='?',
        help='Optional output file path (only valid with single notebook)'
    )

    parser.add_argument(
        '--suffix',
        default='',
        help='Suffix to add to output filename (e.g., "_converted")'
    )

    parser.add_argument(
        '--no-shebang',
        action='store_true',
        help='Do not add shebang line (#!/usr/bin/env python)'
    )

    parser.add_argument(
        '--no-encoding',
        action='store_true',
        help='Do not add encoding declaration'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Directory to save converted scripts (default: same as notebook)'
    )

    args = parser.parse_args()

    # Handle multiple notebooks
    notebooks = []
    for pattern in args.notebook:
        path = Path(pattern)
        if path.is_file():
            notebooks.append(path)
        else:
            # Try as glob pattern
            matched = list(Path('.').glob(pattern))
            notebooks.extend([p for p in matched if p.suffix == '.ipynb'])

    if not notebooks:
        print(f"Error: No notebooks found matching: {args.notebook}")
        sys.exit(1)

    # Validate arguments
    if args.output and len(notebooks) > 1:
        print("Error: Cannot specify output file when converting multiple notebooks")
        sys.exit(1)

    # Convert notebooks
    success_count = 0
    error_count = 0

    for notebook_path in notebooks:
        try:
            # Determine output path
            if args.output:
                output_path = Path(args.output)
            else:
                output_name = notebook_path.stem + args.suffix + '.py'
                if args.output_dir:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = args.output_dir / output_name
                else:
                    output_path = notebook_path.parent / output_name

            # Convert
            convert_notebook_to_script(
                notebook_path,
                output_path,
                add_shebang=not args.no_shebang,
                add_encoding=not args.no_encoding
            )
            success_count += 1

        except Exception as e:
            print(f"✗ Error converting {notebook_path}: {e}")
            error_count += 1

    # Print summary for multiple files
    if len(notebooks) > 1:
        print(f"\nSummary: {success_count} succeeded, {error_count} failed")

    sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()

## usage 
# cd code
# python notebook_to_script.py 01_potential_donor_identifier.ipynb