"""Export markdown documentation to HTML format.

Converts generated markdown files to HTML using the markdown library.
"""

import logging
from pathlib import Path

try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

from ..utils.config import load_config
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


def md_to_html(md_content: str, title: str = "Documentation") -> str:
    """Convert markdown content to HTML.

    Args:
        md_content: Markdown content string
        title: HTML page title

    Returns:
        Full HTML page as string
    """
    if HAS_MARKDOWN:
        # Convert markdown to HTML
        html_body = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code', 'codehilite']
        )
    else:
        # Fallback: wrap in <pre> tag
        logger.warning("markdown library not available, using <pre> fallback")
        html_body = f"<pre>{md_content}</pre>"

    # Wrap in full HTML
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            margin-top: 24px;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Courier New", monospace;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin-left: 0;
            color: #555;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""
    return html_template


def main():
    """Export all markdown docs to HTML."""
    setup_logging()
    logger.info("Exporting documentation to HTML...")

    config = load_config()
    artifacts_dir = Path(config.get("paths", {}).get("artifacts", "artifacts"))

    # Define documents to export
    docs = [
        ("MODEL_CARD.md", "Model Card - Water Quality SA Predictor"),
        ("BUSINESS_PLAN_SNAPSHOT.md", "Business Plan Snapshot"),
        ("VIDEO_SCRIPT.md", "Video Script - Water Quality SA Predictor"),
    ]

    for md_filename, title in docs:
        md_path = artifacts_dir / md_filename
        html_path = artifacts_dir / md_filename.replace(".md", ".html")

        if not md_path.exists():
            logger.warning(f"Markdown file not found: {md_path}")
            continue

        # Read markdown
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Convert to HTML
        html_content = md_to_html(md_content, title)

        # Save HTML
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Exported: {html_path}")

    logger.info("HTML export completed successfully")


if __name__ == "__main__":
    main()
