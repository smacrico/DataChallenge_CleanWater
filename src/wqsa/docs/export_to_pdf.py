"""Export markdown documentation to PDF format.

Converts generated markdown files to PDF using ReportLab.
"""

import logging
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..utils.config import load_config
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


def setup_pdf_styles():
    """Create PDF paragraph styles.

    Returns:
        Dictionary of styles
    """
    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(
        ParagraphStyle(
            name="CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=30,
        )
    )

    styles.add(
        ParagraphStyle(
            name="CustomHeading2",
            parent=styles["Heading2"],
            fontSize=18,
            textColor=colors.HexColor("#34495e"),
            spaceAfter=12,
            spaceBefore=12,
        )
    )

    styles.add(
        ParagraphStyle(
            name="CustomHeading3",
            parent=styles["Heading3"],
            fontSize=14,
            textColor=colors.HexColor("#7f8c8d"),
            spaceAfter=10,
            spaceBefore=10,
        )
    )

    return styles


def md_to_pdf_simple(md_content: str, output_path: Path, title: str = "Documentation"):
    """Convert markdown content to PDF (simplified approach).

    Args:
        md_content: Markdown content string
        output_path: Path to save PDF
        title: Document title
    """
    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    # Get styles
    styles = setup_pdf_styles()
    story = []

    # Add title
    story.append(Paragraph(title, styles["CustomTitle"]))
    story.append(Spacer(1, 12))

    # Simple markdown parsing (convert lines to paragraphs)
    lines = md_content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            story.append(Spacer(1, 6))
            i += 1
            continue

        # Heading level 1
        if line.startswith("# "):
            text = line[2:].strip()
            story.append(Paragraph(text, styles["CustomTitle"]))
            story.append(Spacer(1, 12))

        # Heading level 2
        elif line.startswith("## "):
            text = line[3:].strip()
            story.append(Paragraph(text, styles["CustomHeading2"]))
            story.append(Spacer(1, 8))

        # Heading level 3
        elif line.startswith("### "):
            text = line[4:].strip()
            story.append(Paragraph(text, styles["CustomHeading3"]))
            story.append(Spacer(1, 6))

        # Bullet points
        elif line.startswith("- ") or line.startswith("* "):
            text = "• " + line[2:].strip()
            story.append(Paragraph(text, styles["BodyText"]))

        # Horizontal rule
        elif line.startswith("---"):
            story.append(Spacer(1, 12))
            story.append(PageBreak())

        # Regular paragraph
        else:
            # Escape special characters for ReportLab
            text = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(text, styles["BodyText"]))

        i += 1

    # Build PDF
    doc.build(story)
    logger.info(f"PDF created: {output_path}")


def main():
    """Export all markdown docs to PDF."""
    setup_logging()
    logger.info("Exporting documentation to PDF...")

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
        pdf_path = artifacts_dir / md_filename.replace(".md", ".pdf")

        if not md_path.exists():
            logger.warning(f"Markdown file not found: {md_path}")
            continue

        # Read markdown
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Convert to PDF
        try:
            md_to_pdf_simple(md_content, pdf_path, title)
            logger.info(f"Exported: {pdf_path}")
        except Exception as e:
            logger.error(f"Failed to export {md_filename} to PDF: {e}")

    logger.info("PDF export completed")


if __name__ == "__main__":
    main()
