# notes_renderer.py
"""
Shared rendering layer for hierarchical notes.

Converts a validated HierarchicalNotes JSON dict into:
  - PDF  (via FPDF)
  - TXT  (numbered sections with bullets)
  - Markdown (# / ## / ### / - structure)

Both KG-based and non-KG notes pipelines use this renderer
after building their HierarchicalNotes dict.
"""

import re
from pathlib import Path
from typing import Dict, Optional, List


def _safe_latin1(s: str) -> str:
    """Encode to latin-1 safely for FPDF."""
    return s.encode("latin-1", "replace").decode("latin-1")


# ─── PDF Renderer (FPDF) ─────────────────────────────────────────────────────

def render_pdf(
    notes: Dict,
    output_path: Path,
    image_base_dir: Optional[Path] = None
) -> Path:
    """
    Render HierarchicalNotes dict to a highly-styled PDF with Topics Overview.
    """
    from fpdf import FPDF

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class CustomPDF(FPDF):
        def add_header_banner(self, is_toc=False):
            # Dark Blue Header Background
            self.set_fill_color(22, 60, 133) # 163C85 dark blue
            self.rect(15, 15, self.w - 30, 40, 'F')
            
            # Title
            self.set_y(25)
            self.set_font("Arial", "B", 24)
            self.set_text_color(255, 255, 255)
            self.cell(0, 10, _safe_latin1("Educational Notes"), ln=True, align="C")
            
            # Subtitle
            self.set_font("Arial", "I", 12)
            self.cell(0, 8, _safe_latin1("Generated from Knowledge Graph Analysis"), ln=True, align="C")
            self.set_text_color(0, 0, 0)
            self.set_y(65)

    pdf = CustomPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # ── Topics Overview Page ──
    pdf.add_page()
    pdf.add_header_banner(is_toc=True)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(22, 60, 133)
    pdf.cell(0, 10, "Topics Overview", ln=True)
    
    # TOC Header Row
    pdf.set_fill_color(49, 104, 187) # Brighter blue for table header
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(15, 8, "#", border=1, fill=True, align="C")
    pdf.cell(pdf.w - 45, 8, "Topic", border=1, fill=True, ln=True)
    
    # TOC Rows
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 10)
    
    # Create TOC Links
    toc_links = []
    
    for s_idx, section in enumerate(notes.get("sections", []), 1):
        heading = section.get("heading", "Section")
        link = pdf.add_link()
        toc_links.append(link)
        
        # Zebra striping
        fill = False
        if s_idx % 2 == 0:
            pdf.set_fill_color(245, 248, 252)
            fill = True
            
        pdf.cell(15, 8, str(s_idx), border=1, fill=fill, align="C")
        pdf.cell(pdf.w - 45, 8, _safe_latin1(heading), border=1, fill=fill, ln=True, link=link)

    # ── Sections Content ──
    pdf.add_page()
    
    for s_idx, section in enumerate(notes.get("sections", []), 1):
        # Set link target for TOC
        pdf.set_link(toc_links[s_idx-1])
        
        heading = section.get("heading", "Section")

        # Visual indicator (blue bar) next to heading
        pdf.set_x(15)
        pdf.set_fill_color(49, 104, 187)
        pdf.rect(15, pdf.get_y(), 3, 8, 'F')
        
        pdf.set_x(20)
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(10, 30, 80)
        pdf.cell(0, 8, _safe_latin1(f"{s_idx}. {heading}"), ln=True)
        pdf.ln(4)
        pdf.set_text_color(0, 0, 0)

        # ── Subsections ──
        for ss_idx, sub in enumerate(section.get("subsections", []), 1):
            ss_heading = sub.get("heading", "Subtopic")
            
            points = sub.get("points", [])
            if not points:
                continue

            if ss_heading.lower() == "description":
                # Special styling for "Description": Light blue box, italicized, no header text
                pdf.set_fill_color(237, 244, 252) # Light blue
                pdf.set_font("Arial", "I", 10)
                
                # Combine points for the description
                desc_text = " ".join([pt.get("text", "").strip() for pt in points if pt.get("text", "").strip()])
                if desc_text:
                    pdf.multi_cell(0, 6, _safe_latin1(desc_text), fill=True, border=1)
                    pdf.ln(6)
            else:
                # Regular subsection header
                pdf.set_font("Arial", "B", 11)
                pdf.set_text_color(49, 104, 187)
                pdf.cell(0, 6, _safe_latin1(f"- {ss_heading}"), ln=True)
                pdf.set_text_color(0, 0, 0)
                
                # Bullet points with inline bolding
                pdf.set_font("Arial", "", 10)
                for pt in points:
                    text = pt.get("text", "").strip()
                    if text:
                        # Add hanging indent for bullets
                        pdf.set_x(20)
                        
                        # Handle inline bolding if point is formatted like "Key Concept: Description"
                        if ":" in text:
                            parts = text.split(":", 1)
                            bold_part = parts[0] + ":"
                            rest_part = parts[1]
                            
                            pdf.set_font("Arial", "B", 10)
                            bold_w = pdf.get_string_width(bold_part) + 1
                            pdf.cell(bold_w, 5, _safe_latin1(bold_part))
                            
                            pdf.set_font("Arial", "", 10)
                            pdf.multi_cell(0, 5, _safe_latin1(rest_part))
                        else:
                            pdf.multi_cell(0, 5, _safe_latin1(f"* {text}"))
                        pdf.ln(2)
                pdf.ln(4)

        # ── Diagram (if present) ──
        diagram = section.get("diagram")
        if diagram:
            diag_path = diagram.get("path", "")
            caption = diagram.get("caption", "")

            # Resolve path
            if diag_path:
                p = Path(diag_path)
                if not p.is_absolute() and image_base_dir:
                    p = image_base_dir / diag_path

                if p.exists():
                    try:
                        pdf.set_font("Arial", "B", 10)
                        pdf.set_text_color(49, 104, 187)
                        pdf.cell(0, 6, "Visual Representation", ln=True)
                        pdf.set_text_color(0, 0, 0)
                        
                        from PIL import Image as PILImage
                        with PILImage.open(str(p)) as img:
                            w_px, h_px = img.size
                        max_w, max_h = 90, 90 # Constrain image size
                        aspect = h_px / w_px if w_px > 0 else 1
                        if w_px > h_px:
                            w_mm = min(max_w, w_px * 0.264583)
                            h_mm = w_mm * aspect
                        else:
                            h_mm = min(max_h, h_px * 0.264583)
                            w_mm = h_mm / aspect

                        # Center image
                        x_pos = (pdf.w - w_mm) / 2
                        pdf.image(str(p), x=x_pos, w=w_mm)
                        pdf.ln(1)
                    except Exception as e:
                        print(f"[Renderer] Failed to embed image {p}: {e}")

                    if caption:
                        pdf.set_font("Arial", "I", 9)
                        pdf.set_text_color(100, 100, 100)
                        pdf.multi_cell(0, 4, _safe_latin1(f"Figure {s_idx}: {caption}"), align="C")
                        pdf.set_text_color(0, 0, 0)
                        pdf.ln(6)

        # Separation line between main topics
        pdf.set_draw_color(200, 200, 200)
        pdf.line(15, pdf.get_y(), pdf.w - 15, pdf.get_y())
        pdf.ln(6)

    pdf.output(str(output_path))
    return output_path


# ─── TXT Renderer ─────────────────────────────────────────────────────────────

def render_txt(notes: Dict, output_path: Path) -> Path:
    """
    Render HierarchicalNotes dict to a plain text file.

    Returns:
        Path to the generated TXT file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    title = notes.get("title", "Lecture Notes")
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    for s_idx, section in enumerate(notes.get("sections", []), 1):
        heading = section.get("heading", "Section")
        lines.append(f"{s_idx}. {heading}")
        lines.append("-" * 40)

        for ss_idx, sub in enumerate(section.get("subsections", []), 1):
            ss_heading = sub.get("heading", "Subtopic")
            lines.append(f"  {s_idx}.{ss_idx} {ss_heading}")

            for pt in sub.get("points", []):
                text = pt.get("text", "")
                if text.strip():
                    lines.append(f"    - {text}")

            lines.append("")

        # Diagram reference
        diagram = section.get("diagram")
        if diagram and diagram.get("caption"):
            lines.append(f"  [Figure {s_idx}: {diagram['caption']}]")
            lines.append("")

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ─── Markdown Renderer ────────────────────────────────────────────────────────

def render_markdown(notes: Dict, output_path: Path) -> Path:
    """
    Render HierarchicalNotes dict to Markdown.

    Returns:
        Path to the generated .md file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    title = notes.get("title", "Lecture Notes")
    lines.append(f"# {title}")
    lines.append("")

    for s_idx, section in enumerate(notes.get("sections", []), 1):
        heading = section.get("heading", "Section")
        lines.append(f"## {s_idx}. {heading}")
        lines.append("")

        for ss_idx, sub in enumerate(section.get("subsections", []), 1):
            ss_heading = sub.get("heading", "Subtopic")
            lines.append(f"### {s_idx}.{ss_idx} {ss_heading}")

            for pt in sub.get("points", []):
                text = pt.get("text", "")
                if text.strip():
                    lines.append(f"- {text}")

            lines.append("")

        # Diagram
        diagram = section.get("diagram")
        if diagram:
            path = diagram.get("path", "")
            caption = diagram.get("caption", "")
            if path:
                lines.append(f"![{caption}]({path})")
                lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ─── Convenience: Render all formats ─────────────────────────────────────────

def render_all(
    notes: Dict,
    output_dir: Path,
    base_name: str = "lecture_notes",
    image_base_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Render notes to PDF + TXT + Markdown.

    Returns:
        Dict with keys 'pdf', 'txt', 'md' mapping to output paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    paths["pdf"] = render_pdf(notes, output_dir / f"{base_name}.pdf", image_base_dir)
    paths["txt"] = render_txt(notes, output_dir / f"{base_name}.txt")
    paths["md"] = render_markdown(notes, output_dir / f"{base_name}.md")

    return paths
