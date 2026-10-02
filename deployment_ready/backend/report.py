"""
Brain Tumor Detection — PDF Report Generator
Generates professional diagnostic reports using ReportLab with Calibrated Confidence, Tumor Characteristics, Recommended Medical Management, and XAI Insights.
"""

import io
import base64
import datetime
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String


DARK_BG     = colors.HexColor("#ffffff")
ACCENT      = colors.HexColor("#4f46e5")
ACCENT_SOFT = colors.HexColor("#4338ca")
SURFACE     = colors.HexColor("#f8fafc")
BAR_TRACK   = colors.HexColor("#e2e8f0")
BORDER      = colors.HexColor("#cbd5e1")
TEXT_MAIN   = colors.HexColor("#0f172a")
TEXT_MUTED  = colors.HexColor("#475569")
SUCCESS     = colors.HexColor("#16a34a")
WARNING     = colors.HexColor("#d97706")
DANGER      = colors.HexColor("#dc2626")
PURPLE      = colors.HexColor("#7c3aed")

SEVERITY_COLORS = {
    "None":   SUCCESS,
    "Low":    SUCCESS,
    "Medium": WARNING,
    "High":   DANGER,
}


def severity_color(severity):
    return SEVERITY_COLORS.get(severity, TEXT_MUTED)


def build_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["title"] = ParagraphStyle(
        "title",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=TEXT_MAIN,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    styles["section_header"] = ParagraphStyle(
        "section_header",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=ACCENT_SOFT,
        spaceAfter=6,
        spaceBefore=10,
    )
    styles["body"] = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=9,
        textColor=TEXT_MAIN,
        leading=14,
        spaceAfter=4,
    )
    styles["muted"] = ParagraphStyle(
        "muted",
        fontName="Helvetica",
        fontSize=8,
        textColor=TEXT_MUTED,
        leading=12,
    )
    styles["bold"] = ParagraphStyle(
        "bold",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=TEXT_MAIN,
    )
    styles["disclaimer"] = ParagraphStyle(
        "disclaimer",
        fontName="Helvetica-Oblique",
        fontSize=7,
        textColor=TEXT_MUTED,
        alignment=TA_CENTER,
        leading=10,
        spaceAfter=3,
    )
    styles["center"] = ParagraphStyle(
        "center",
        fontName="Helvetica",
        fontSize=9,
        textColor=TEXT_MAIN,
        alignment=TA_CENTER,
    )
    return styles


def b64_to_rl_image(b64_str, max_width, max_height):
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")

    w, h = pil_img.size
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=new_w, height=new_h)


def confidence_bar_table(scores, styles):
    CLASS_COLORS = {
        "glioma":     colors.HexColor("#ef4444"),
        "meningioma": colors.HexColor("#f59e0b"),
        "notumor":    colors.HexColor("#22c55e"),
        "pituitary":  colors.HexColor("#8b5cf6"),
    }
    rows = []
    for label, data in sorted(scores.items(), key=lambda x: -x[1]["calibrated_confidence"]):
        conf  = data["calibrated_confidence"]
        label_display = label.replace("notumor", "No Tumor").title()
        bar_width_max = 180
        bar_width     = max(4, int(bar_width_max * conf / 100))

        bar_color = CLASS_COLORS.get(label.lower(), ACCENT)

        bar_drawing = Drawing(bar_width_max + 60, 12)
        bar_drawing.add(Rect(0, 1, bar_width_max, 9, fillColor=BAR_TRACK, strokeColor=BORDER, strokeWidth=0.5))
        bar_drawing.add(Rect(0, 1, bar_width, 9, fillColor=bar_color, strokeColor=None))
        bar_drawing.add(String(bar_width_max + 5, 2, f"{conf:.1f}%",
                               fontName="Helvetica-Bold", fontSize=8, fillColor=TEXT_MAIN))

        rows.append([
            Paragraph(label_display, styles["body"]),
            bar_drawing,
        ])

    table = Table(rows, colWidths=[100, 250])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
    ]))
    return table


def generate_report(prediction_data: dict, patient_info: dict = None) -> bytes:
    buf    = io.BytesIO()
    styles = build_styles()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title="Brain Tumor Detection Report",
        author="NeuroScan AI",
    )

    patient  = patient_info or {}
    pid      = patient.get("patient_id", "PT-2026-EX")
    name     = patient.get("name",       "Anonymous Patient")
    age      = patient.get("age",        "N/A")
    gender   = patient.get("gender",     "N/A")
    doctor   = patient.get("referring_doctor", "Research AI System")
    now      = datetime.datetime.now()
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%H:%M:%S")

    pred      = prediction_data
    sev_color = severity_color(pred.get("severity", "None"))
    tumor_color = colors.HexColor(pred.get("color", "#6366f1"))

    story = []

    analysis_id = pred.get("analysis_id") or patient.get("analysis_id") or "N/A"

    story = []

    # 1. Header Banner
    header_data = [[
        Paragraph("🧠  NeuroScan AI — Research Diagnostic Report", ParagraphStyle(
            "hdr", fontName="Helvetica-Bold", fontSize=15, textColor=ACCENT, alignment=TA_LEFT)),
        Paragraph(f"Analysis ID: <b>{analysis_id}</b><br/>Date: {date_str}<br/>Time: {time_str}",
                  ParagraphStyle("hdr2", fontName="Helvetica", fontSize=8, textColor=TEXT_MUTED, alignment=TA_RIGHT)),
    ]]
    header_table = Table(header_data, colWidths=["60%", "40%"])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING",(0,0), (-1,-1), 10),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph("Brain Tumor MRI Diagnostic & Explainability Report", styles["title"]))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 6))

    # 2. AI Analysis Result Summary Section (Matching UI Card)
    story.append(Paragraph("📋  AI Analysis Result", styles["section_header"]))
    
    file_name = patient.get("file_name") or pred.get("file_name") or "MRI_Scan.jpg"
    is_no_tumor = (pred.get("class_id") == "notumor" or pred.get("prediction") == "No Tumor" or pred.get("prediction") == "notumor")
    tumor_status = "No Tumor Detected" if is_no_tumor else "Tumor Detected"
    model_name = pred.get("model_used") or "EfficientNetB0_Transfer"
    pred_display = pred.get("prediction") or pred.get("display_name") or "Unknown"
    cal_conf_val = pred.get('calibrated_confidence') if pred.get('calibrated_confidence') is not None else pred.get('confidence', 0)
    cal_conf_str = f"{cal_conf_val:.1f}%" if cal_conf_val is not None else "N/A"
    date_time_full = f"{date_str}, {time_str}"

    analysis_grid = [
        [
            Paragraph("Analysis ID", styles["muted"]), Paragraph(analysis_id, ParagraphStyle("aid", fontName="Helvetica-Bold", fontSize=9, textColor=ACCENT_SOFT)),
            Paragraph("Patient ID", styles["muted"]), Paragraph(pid, styles["bold"])
        ],
        [
            Paragraph("Patient Name", styles["muted"]), Paragraph(name, styles["bold"]),
            Paragraph("Age / Gender", styles["muted"]), Paragraph(f"{age} / {gender}", styles["bold"])
        ],
        [
            Paragraph("Referring Doctor", styles["muted"]), Paragraph(doctor, styles["bold"]),
            Paragraph("File Name", styles["muted"]), Paragraph(file_name, styles["bold"])
        ],
        [
            Paragraph("Prediction", styles["muted"]), Paragraph(pred_display, ParagraphStyle("p_disp", fontName="Helvetica-Bold", fontSize=9, textColor=tumor_color)),
            Paragraph("Tumor Status", styles["muted"]), Paragraph(tumor_status, ParagraphStyle("ts", fontName="Helvetica-Bold", fontSize=9, textColor=SUCCESS if is_no_tumor else DANGER))
        ],
        [
            Paragraph("Confidence Score", styles["muted"]), Paragraph(cal_conf_str, ParagraphStyle("c_disp", fontName="Helvetica-Bold", fontSize=9, textColor=ACCENT_SOFT)),
            Paragraph("Model Used", styles["muted"]), Paragraph(model_name, styles["bold"])
        ],
    ]

    analysis_table = Table(analysis_grid, colWidths=[90, 120, 90, 120])
    analysis_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(analysis_table)
    story.append(Spacer(1, 8))

    # 3. Diagnosis Summary & Calibration
    story.append(Paragraph("Diagnosis Detail & Calibration", styles["section_header"]))
    if pred.get("is_uncertain"):
        story.append(Paragraph(
            f"<b>⚠️ {pred.get('uncertainty_reason', 'UNCERTAIN PREDICTION')}</b>",
            ParagraphStyle("warn_box", fontName="Helvetica-Bold", fontSize=9, textColor=WARNING,
                           backColor=colors.HexColor("#fffbeb"), borderColor=WARNING, borderWidth=1,
                           borderPadding=6, spaceAfter=6, leading=12)
        ))

    diag_data = [[
        Paragraph(
            f"<font color='#{pred.get('color','6366f1')[1:]}' size='13'><b>{pred.get('display_name', 'Unknown')}</b></font>",
            ParagraphStyle("diag", fontName="Helvetica-Bold", fontSize=13, textColor=tumor_color, alignment=TA_CENTER)
        ),
        Table([
            [Paragraph("Raw Conf.", styles["muted"]), Paragraph(f"<b>{pred.get('confidence', 0):.1f}%</b>", styles["bold"])],
            [Paragraph("Calibrated Conf.", styles["muted"]), Paragraph(f"<b>{pred.get('calibrated_confidence', 0):.1f}%</b>", styles["bold"])],
            [Paragraph("Severity", styles["muted"]), Paragraph(f"<b>{pred.get('severity', 'N/A')}</b>", ParagraphStyle("sev", fontName="Helvetica-Bold", fontSize=9, textColor=sev_color))],
        ], colWidths=[80, 90]),
    ]]
    diag_table = Table(diag_data, colWidths=["50%", "50%"])
    diag_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 6))

    desc = pred.get("description", "")
    if desc:
        story.append(Paragraph(desc, styles["body"]))
    story.append(Spacer(1, 8))

    # 4. Tumor Characteristics Section
    characteristics = pred.get("characteristics", [])
    if characteristics:
        story.append(Paragraph("🧬  Tumor Characteristics", styles["section_header"]))
        char_rows = []
        for char in characteristics:
            char_p = Paragraph(f"• {char}", ParagraphStyle(
                "char_bullet",
                fontName="Helvetica",
                fontSize=9,
                textColor=TEXT_MAIN,
                leading=13,
            ))
            char_rows.append([char_p])
        
        char_table = Table(char_rows, colWidths=[420])
        char_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(char_table)
        story.append(Spacer(1, 8))

    # 5. Recommended Medical Management Section
    treatment = pred.get("treatment", "")
    if treatment:
        story.append(Paragraph("🩺  Recommended Medical Management", styles["section_header"]))
        treat_p = Paragraph(treatment, ParagraphStyle(
            "treat_body",
            fontName="Helvetica",
            fontSize=9,
            textColor=TEXT_MAIN,
            leading=14,
        ))
        treat_table = Table([[treat_p]], colWidths=[420])
        treat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(treat_table)
        story.append(Spacer(1, 8))

    # 6. Multi-XAI Analysis
    story.append(Paragraph("Model Attention Region & Explainability (Multi-XAI)", styles["section_header"]))
    img_cells = []
    captions = []

    if pred.get("original_b64"):
        try:
            img_cells.append(b64_to_rl_image(pred["original_b64"], 120, 120))
            captions.append(Paragraph("Original MRI", styles["center"]))
        except Exception:
            pass

    if pred.get("overlay_b64"):
        try:
            img_cells.append(b64_to_rl_image(pred["overlay_b64"], 120, 120))
            captions.append(Paragraph("Grad-CAM", styles["center"]))
        except Exception:
            pass

    if pred.get("ig_b64"):
        try:
            img_cells.append(b64_to_rl_image(pred["ig_b64"], 120, 120))
            captions.append(Paragraph("Integrated Gradients", styles["center"]))
        except Exception:
            pass

    if img_cells:
        img_table = Table([img_cells, captions], colWidths=[135] * len(img_cells))
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, 0), SURFACE),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING",(0,0), (-1,-1), 4),
            ("TOPPADDING",  (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ]))
        story.append(img_table)

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<i>Note: Grad-CAM and Integrated Gradients highlight feature regions that contributed to the model prediction.</i>",
        styles["muted"]
    ))
    story.append(Spacer(1, 8))

    # 7. Confidence Scores
    scores = pred.get("scores", {})
    if scores:
        story.append(KeepTogether([
            Paragraph("Classification Confidence Scores (Calibrated)", styles["section_header"]),
            Spacer(1, 2),
            confidence_bar_table(scores, styles)
        ]))

    story.append(Spacer(1, 8))

    # 8. Disclaimer
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "⚠️  DISCLAIMER: This system is intended for research and educational purposes only. "
        "It does NOT provide a medical diagnosis and should NOT replace evaluation by a qualified medical professional.",
        styles["disclaimer"]
    ))
    story.append(Paragraph(
        f"Generated by NeuroScan AI  •  {date_str} at {time_str}  •  Model: {pred.get('model_used', 'Multi-Model Research')}",
        styles["disclaimer"]
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
