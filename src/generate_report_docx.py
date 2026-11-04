"""
Script to generate Case_Study_Report_Template.docx
Student Register Number: 24BCE0551
Track: Multivariate and Time-Series Detection
Assigned Method: PCA-Based Multivariate Anomaly Detector
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def create_report_template():
    doc = Document()
    
    # Page setup: Standard Letter, 1-inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # Title Block
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("AI-Based Anomaly Detection and Intelligent Monitoring\nof Chiller Operating Data")
    run_title.font.name = 'Arial'
    run_title.font.size = Pt(20)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D) # Navy Blue

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(18)
    run_sub = p_sub.add_run("Academic Case Study Report Template — Multivariate and Time-Series Detection Track\nStudent Register Number: 24BCE0551 | Assigned Method: PCA-Based Detector")
    run_sub.font.size = Pt(11)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # Callout Box: Academic Integrity Notice
    t_box = doc.add_table(rows=1, cols=1)
    t_box.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = t_box.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    p_box = cell.paragraphs[0]
    p_box.paragraph_format.space_before = Pt(6)
    p_box.paragraph_format.space_after = Pt(6)
    r_box = p_box.add_run(
        "MANDATORY ACADEMIC INTEGRITY DIRECTIVE (INSTRUCTOR POLICY):\n"
        "• AI-Generated Content must be below 10% | Similarity must be below 5%.\n"
        "• This document is a structured template containing verified empirical results, data tables, and outline prompts.\n"
        "• You must author your own independent narrative, interpretations, and critical reflections in each section."
    )
    r_box.font.size = Pt(9.5)
    r_box.font.bold = True
    r_box.font.color.rgb = RGBColor(0x0C, 0x23, 0x40)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Helper function for section headings
    def add_section_header(title):
        h = doc.add_heading(level=1)
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(4)
        r = h.add_run(title)
        r.font.name = 'Arial'
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
        return h

    def add_guidance_box(content):
        t = doc.add_table(rows=1, cols=1)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        c = t.cell(0, 0)
        set_cell_background(c, "FDF6E2") # Soft amber
        p = c.paragraphs[0]
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(content)
        r.font.size = Pt(9.5)
        r.font.italic = True
        r.font.color.rgb = RGBColor(0x7A, 0x4B, 0x00)
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # 1. ABSTRACT
    add_section_header("1. Executive Summary & Abstract")
    add_guidance_box(
        "Recommended Scope (200–250 words):\n"
        "• Summarize the research question: Detecting anomalies from correlated temperature and pressure variables in a 5-chiller dual-compressor fleet.\n"
        "• Mention preprocessing rigor: Chronological splitting (Train: 358, Val: 92, Test: 67) preventing multi-machine data leakage.\n"
        "• State model architecture: Linear PCA (k = 12 PCs, 85.46% variance, 16 residual dimensions).\n"
        "• Highlight key results: 95th percentile validation threshold (tau = 9.2931) flagged 2.99% of clean test records; synthetic experiments achieved 100% event detection across sudden spikes, gradual drift, and relationship breaking.\n"
        "• Conclude on the physical validity of R134a vapor-pressure decoupling detection."
    )
    p_ph1 = doc.add_paragraph("[Student Independent Writing: Insert your synthesized abstract here...]")
    p_ph1.runs[0].font.italic = True

    # 2. INTRODUCTION & PROBLEM STATEMENT
    add_section_header("2. Introduction & Problem Statement")
    add_guidance_box(
        "Topics to Address Independently:\n"
        "• Industrial significance: Central chilled water plants consume up to 40% of building energy in commercial facilities.\n"
        "• Failure modes: Refrigerant leaks, condenser tube fouling, sensor calibration drift, expansion valve sticking.\n"
        "• Why univariate thresholding fails: A temperature of 45°C may be perfectly normal under high head pressure, but severely anomalous under low suction pressure. Multivariate correlation monitoring is required.\n"
        "• Assigned Research Question: Can anomalies be detected from correlated temperature and pressure variables?\n"
        "• Official raw dataset source: data/raw/official_dataset/Analyse Chillers_AI Course.xls."
    )
    p_ph2 = doc.add_paragraph("[Student Independent Writing: Formulate the background, thermodynamic motivation, and engineering context here...]")
    p_ph2.runs[0].font.italic = True

    # 3. DATASET UNDERSTANDING & DATA QUALITY (EDA)
    add_section_header("3. Dataset Understanding and Exploratory Data Analysis")
    add_guidance_box(
        "Empirical Evidence to Report:\n"
        "• Dimensions: 525 raw rows x 60 columns. Operating scope: N = 517 dual-compressor observations across Chillers 1, 2, 3, 4, and 5.\n"
        "• Data completeness: 0.0% missing cells across all 28 features (100% complete); 0 exact duplicate rows.\n"
        "• Sampling interval: 1.0 hour modal interval across 222 unique plant hours.\n"
        "• Mention industrial logging typos discovered in raw logs (10x decimal errors in motor currents and liquid temps; saturation temp logged as 322°C).\n"
        "• Figure Reference: Insert Figure 1 (results/figures/final/final_temperature_pressure_correlation.png).\n"
        "• Key finding: Strong physical correlation between Discharge Temp and Pressure Comp 1 (r = +0.9578) and Suction Sat Temp and Pressure (r = +0.8973)."
    )

    # Insert Table of Fleet Distribution
    doc.add_paragraph().add_run("Table 1: Operational Chiller Fleet Distribution (Dual-Compressor Regime)").bold = True
    t1 = doc.add_table(rows=6, cols=4)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Chiller Tag", "Raw Observations", "Operating Regime", "Sample Representation (%)"]
    for i, h_text in enumerate(headers):
        cell = t1.cell(0, i)
        set_cell_background(cell, "1B365D")
        r = cell.paragraphs[0].add_run(h_text)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.size = Pt(9.5)
    
    fleet_data = [
        ["Chiller - 1", "96", "Dual-Compressor Active", "18.57%"],
        ["Chiller - 2", "82", "Dual-Compressor Active", "15.86%"],
        ["Chiller - 3", "120", "Dual-Compressor Active", "23.21%"],
        ["Chiller - 4", "131", "Dual-Compressor Active", "25.34%"],
        ["Chiller - 5", "88", "Dual-Compressor Active", "17.02%"]
    ]
    for row_idx, data_row in enumerate(fleet_data, start=1):
        for col_idx, text_val in enumerate(data_row):
            cell = t1.cell(row_idx, col_idx)
            set_cell_background(cell, "F9FBFD" if row_idx % 2 == 1 else "FFFFFF")
            r = cell.paragraphs[0].add_run(text_val)
            r.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 4. PREPROCESSING & ANTI-DATA-LEAKAGE
    add_section_header("4. Data Preprocessing & Anti-Data-Leakage Methodology")
    add_guidance_box(
        "Key Methodological Principles:\n"
        "• Feature boundary: Exactly 28 thermodynamic variables (21 temperatures, 7 refrigerant pressures). Auxiliary electrical and pump pressures strictly excluded.\n"
        "• Chronological splitting protocol: Partitioned by unique plant timestamp hours (Train: 358, Val: 92, Test: 67) with zero temporal overlap.\n"
        "• Anti-leakage justification: Explain why random splitting or row slicing across synchronized chillers causes severe data leakage.\n"
        "• Training-only transformer fitting: SimpleImputer(median) and StandardScaler fitted exclusively on training data.\n"
        "• Reference results/metrics/preprocessing/preprocessing_validation.csv (18 passed checks)."
    )

    # Insert Table of Chronological Splits
    doc.add_paragraph().add_run("Table 2: Chronological Anti-Leakage Partitioning Summary").bold = True
    t2 = doc.add_table(rows=4, cols=5)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    h2 = ["Partition", "Unique Hours", "Date & Time Range", "Sample Count (N)", "Fleet Proportion (%)"]
    for i, h_text in enumerate(h2):
        cell = t2.cell(0, i)
        set_cell_background(cell, "1B365D")
        r = cell.paragraphs[0].add_run(h_text)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.size = Pt(9.5)
    
    split_data = [
        ["Training Set", "155", "2026-03-11 00:00 to 2026-03-17 10:00", "358", "69.25%"],
        ["Validation Set", "33", "2026-03-17 11:00 to 2026-03-18 19:00", "92", "17.79%"],
        ["Testing Set", "34", "2026-03-18 20:00 to 2026-03-20 08:00", "67", "12.96%"]
    ]
    for row_idx, data_row in enumerate(split_data, start=1):
        for col_idx, text_val in enumerate(data_row):
            cell = t2.cell(row_idx, col_idx)
            set_cell_background(cell, "F9FBFD" if row_idx % 2 == 1 else "FFFFFF")
            r = cell.paragraphs[0].add_run(text_val)
            r.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 5. PCA MODEL DEVELOPMENT
    add_section_header("5. PCA Model Development & Subspace Decomposition")
    add_guidance_box(
        "Mathematical and Conceptual Points:\n"
        "• Sample covariance eigendecomposition: Sigma = V Lambda V^T.\n"
        "• Eigenvalue spectrum: PC1 explains 23.77%, PC2 explains 13.64%.\n"
        "• Component selection justification: Why k = 12 was chosen (85.46% variance, 16 residual dimensions) over Kaiser rule (k=8, 72.35%).\n"
        "• Figure References: Insert Figure 2 (Scree plot, results/figures/final/final_pca_explained_variance.png) and Figure 3 (Loadings heatmap, results/figures/final/final_pca_component_loadings.png).\n"
        "• Physical interpretation: PC1 captures High-Side Lift; PC2 captures Low-Side Cooling Load."
    )
    p_ph5 = doc.add_paragraph("[Student Independent Writing: Provide mathematical formulations and physical component interpretations here...]")
    p_ph5.runs[0].font.italic = True

    # 6. RECONSTRUCTION ERROR & THRESHOLD CALIBRATION
    add_section_header("6. Reconstruction Error Derivation & Anomaly Detection")
    add_guidance_box(
        "Key Results to Discuss:\n"
        "• Squared Prediction Error formula: SPE_i = sum(e_ij^2) across 28 standardized features.\n"
        "• Threshold selection on validation set only: Why the 95th percentile (tau = 9.2931) was selected instead of the 99th percentile (tau = 1378.35, corrupted by operator typo).\n"
        "• Clean test evaluation (N = 67): 65 normal (97.01%), 2 baseline excursions (2.99%) on Chiller-4 (9.91) and Chiller-5 (13.60).\n"
        "• Figure References: Insert Figure 4 (Timeline, final_clean_test_reconstruction_error.png) and Figure 6 (Contributions, final_feature_residual_contributions.png).\n"
        "• Sensor contributions: Suction and lift temperatures account for > 73% of error in flagged points."
    )
    p_ph6 = doc.add_paragraph("[Student Independent Writing: Explain threshold calibration dynamics, order statistics, and clean test baseline behavior...]")
    p_ph6.runs[0].font.italic = True

    # 7. CONTROLLED SYNTHETIC ANOMALY EXPERIMENTS
    add_section_header("7. Controlled Synthetic Anomaly Experiments & Model Evaluation")
    add_guidance_box(
        "Experimental Evidence:\n"
        "• Why synthetic testing is necessary: Unlabelled operational data cannot evaluate Precision, Recall, or Latency without ground truth.\n"
        "• Scenario A (Sudden Spikes, +-4.0 sigma): 100% recall, 0 hours delay.\n"
        "• Scenario B (Gradual Drift, +0.7 to +3.5 sigma): 20% observation recall, 100% event detection, 4 hours delay.\n"
        "• Scenario C (Relationship Breaking, -3.5 sigma pressure decoupling): 100% recall, 0 hours delay.\n"
        "• Explain why FP = 2 across all scenarios (the 2 clean baseline excursions).\n"
        "• Figure References: Insert Figure 7 (Spikes), Figure 8 (Drift), Figure 9 (Rel Breaking), Figure 10 (Confusion matrices), Figure 11 (Metrics), Figure 12 (Event detection)."
    )

    # Insert Table of Consolidated Metrics
    doc.add_paragraph().add_run("Table 3: Consolidated Evaluation Performance Across Synthetic Anomaly Categories").bold = True
    t3 = doc.add_table(rows=4, cols=8)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    h3 = ["Scenario", "Injected Obs", "TP", "FP", "FN", "Precision", "Recall", "Event Recall"]
    for i, h_text in enumerate(h3):
        cell = t3.cell(0, i)
        set_cell_background(cell, "1B365D")
        r = cell.paragraphs[0].add_run(h_text)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.size = Pt(9.5)
    
    perf_data = [
        ["Scenario A: Sudden Spikes", "3", "3", "2", "0", "0.6000", "1.0000", "100.0%"],
        ["Scenario B: Gradual Drift", "5", "1", "2", "4", "0.3333", "0.2000", "100.0%"],
        ["Scenario C: Relationship Breaking", "3", "3", "2", "0", "0.6000", "1.0000", "100.0%"]
    ]
    for row_idx, data_row in enumerate(perf_data, start=1):
        for col_idx, text_val in enumerate(data_row):
            cell = t3.cell(row_idx, col_idx)
            set_cell_background(cell, "F9FBFD" if row_idx % 2 == 1 else "FFFFFF")
            r = cell.paragraphs[0].add_run(text_val)
            r.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 8. DISCUSSION & RESEARCH QUESTION EVIDENCE
    add_section_header("8. Discussion & Research Question Resolution")
    add_guidance_box(
        "Critical Reflection Points:\n"
        "• Directly address the research question: Yes, breaking temperature-pressure correlation elevates SPE into residual space and triggers alarm.\n"
        "• Contrast observation-level vs event-level performance in gradual drift.\n"
        "• Limitations: Small sample size (N_test = 67), linear PCA assumption, absence of real physical fault work orders.\n"
        "• Future improvements: Dynamic PCA (DPCA), non-linear autoencoders, multi-model chiller-specific thresholds."
    )
    p_ph8 = doc.add_paragraph("[Student Independent Writing: Formulate your discussion, research conclusions, and critical analysis here...]")
    p_ph8.runs[0].font.italic = True

    # 9. CONCLUSION & REFERENCES
    add_section_header("9. Conclusion & References")
    add_guidance_box(
        "Provide genuine academic citations in standard IEEE or Harvard format (e.g., Jolliffe on PCA, Jackson & Mudholkar on Q-statistic, ASHRAE handbook on chiller thermodynamics, Chiang et al. on fault diagnosis)."
    )
    p_ph9 = doc.add_paragraph(
        "1. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. Philosophical Transactions of the Royal Society A, 374(2065), 20150202.\n"
        "2. Jackson, J. E., & Mudholkar, G. S. (1979). Control procedures for residuals associated with principal component analysis. Technometrics, 21(3), 341-349.\n"
        "3. ASHRAE Handbook—HVAC Systems and Equipment. (2020). Liquid Chillers (Chapter 43). American Society of Heating, Refrigerating and Air-Conditioning Engineers.\n"
        "4. Chiang, L. H., Russell, E. L., & Braatz, R. D. (2001). Fault detection and diagnosis in industrial systems. Springer Science & Business Media."
    )

    doc_path = "Case_Study_Report_Template.docx"
    doc.save(doc_path)
    print(f"Successfully generated report template at: {doc_path}")

if __name__ == "__main__":
    create_report_template()

