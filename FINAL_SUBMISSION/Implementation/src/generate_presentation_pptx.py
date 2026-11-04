"""
Script to generate presentation/Chiller_Anomaly_Detection_Template.pptx
Student Register Number: 24BCE0551
Track: Multivariate and Time-Series Detection
Assigned Method: PCA-Based Multivariate Anomaly Detector
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def create_presentation_template():
    prs = Presentation()
    prs.slide_width = Inches(13.333) # 16:9 widescreen
    prs.slide_height = Inches(7.5)
    
    blank_layout = prs.slide_layouts[6]
    
    c_navy = RGBColor(0x1B, 0x36, 0x5D)
    c_dark = RGBColor(0x22, 0x22, 0x22)
    c_gray = RGBColor(0x55, 0x55, 0x55)
    c_amber = RGBColor(0xB8, 0x5D, 0x00)
    c_white = RGBColor(0xFF, 0xFF, 0xFF)
    
    def add_slide_header(slide, title_text, category_text="CHILLER MULTIVARIATE ANOMALY DETECTION"):
        # Category label
        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        p_cat = tb_cat.text_frame.paragraphs[0]
        r_cat = p_cat.add_run()
        r_cat.text = category_text.upper()
        r_cat.font.name = 'Arial'
        r_cat.font.size = Pt(10)
        r_cat.font.bold = True
        r_cat.font.color.rgb = c_amber
        
        # Title
        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        p_title = tb_title.text_frame.paragraphs[0]
        r_title = p_title.add_run()
        r_title.text = title_text
        r_title.font.name = 'Arial'
        r_title.font.size = Pt(22)
        r_title.font.bold = True
        r_title.font.color.rgb = c_navy

    # SLIDE 1: Title Slide
    s1 = prs.slides.add_slide(blank_layout)
    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(3.5))
    tf1 = tb1.text_frame
    
    p = tf1.paragraphs[0]
    r = p.add_run()
    r.text = "AI-Based Anomaly Detection and Intelligent Monitoring\nof Chiller Operating Data"
    r.font.name = 'Arial'
    r.font.size = Pt(32)
    r.font.bold = True
    r.font.color.rgb = c_navy
    
    p2 = tf1.add_paragraph()
    p2.space_before = Pt(16)
    r2 = p2.add_run()
    r2.text = "Track: Multivariate and Time-Series Detection | Assigned Method: PCA-Based Multivariate Detector\nStudent Register Number: 24BCE0551 | Course: Academic AI/ML Case Study"
    r2.font.size = Pt(15)
    r2.font.color.rgb = c_gray
    
    p3 = tf1.add_paragraph()
    p3.space_before = Pt(24)
    r3 = p3.add_run()
    r3.text = "[Student Oral Note: Introduce yourself, register number, and state your research goal clearly]"
    r3.font.size = Pt(12)
    r3.font.italic = True
    r3.font.color.rgb = c_amber

    # SLIDE 2: Problem Statement & Industrial Motivation
    s2 = prs.slides.add_slide(blank_layout)
    add_slide_header(s2, "Industrial Problem Statement & Motivation")
    tb2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.0))
    tf2 = tb2.text_frame
    bullets2 = [
        ("Central Chilled Water Plants: ", "Responsible for up to 40% of electricity usage in commercial facilities."),
        ("Physical Operational Faults: ", "Refrigerant charge loss, heat exchanger fouling, sensor calibration drift, expansion valve failures."),
        ("Failure of Univariate Thresholds: ", "A temperature reading of 48°C may be normal at full compression lift, but severely abnormal under partial load. Univariate alarms generate excessive false positives."),
        ("Multivariate Opportunity: ", "Centrifugal chillers operate under strict thermodynamic equilibrium (R134a vapor-pressure laws). Collinear sensor relationships provide early warning of developing failures."),
        ("Oral Defense Prompt: ", "[Explain why univariate alarms fail and why multivariate correlation is mandatory]")
    ]
    for b_title, b_desc in bullets2:
        p = tf2.add_paragraph()
        p.space_after = Pt(12)
        r_t = p.add_run()
        r_t.text = b_title
        r_t.font.bold = True
        r_t.font.size = Pt(14)
        r_t.font.color.rgb = c_navy if "Oral" not in b_title else c_amber
        r_d = p.add_run()
        r_d.text = b_desc
        r_d.font.size = Pt(14)
        r_d.font.italic = ("Oral" in b_title)

    # SLIDE 3: Research Question & Objectives
    s3 = prs.slides.add_slide(blank_layout)
    add_slide_header(s3, "Central Research Question & Technical Objectives")
    tb3 = s3.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.0))
    tf3 = tb3.text_frame
    bullets3 = [
        ("Central Research Question: ", "“Can anomalies be detected from correlated temperature and pressure variables?”"),
        ("Objective 1 (Collinearity Proof): ", "Quantify thermodynamic correlation between temperature and pressure loops (R134a vapor-pressure equilibrium)."),
        ("Objective 2 (PCA Subspace Decomposition): ", "Model normal operating variance in a reduced subspace (k = 12 PCs) while preserving a sensitive residual subspace (16 dimensions)."),
        ("Objective 3 (Threshold Calibration & Detection): ", "Calibrate an anti-leakage anomaly threshold (tau = 9.2931) on validation data and apply it to held-out test data."),
        ("Objective 4 (Controlled Synthetic Evaluation): ", "Benchmark detection sensitivity, precision, recall, and latency across sudden spikes, gradual drift, and relationship breaking."),
        ("Oral Defense Prompt: ", "[Emphasize that this case study directly investigates physical thermodynamic coupling]")
    ]
    for b_title, b_desc in bullets3:
        p = tf3.add_paragraph()
        p.space_after = Pt(10)
        r_t = p.add_run()
        r_t.text = b_title
        r_t.font.bold = True
        r_t.font.size = Pt(13.5)
        r_t.font.color.rgb = c_navy if "Oral" not in b_title else c_amber
        r_d = p.add_run()
        r_d.text = b_desc
        r_d.font.size = Pt(13.5)
        r_d.font.italic = ("Oral" in b_title)

    # SLIDE 4: Official Dataset & Data Quality (EDA)
    s4 = prs.slides.add_slide(blank_layout)
    add_slide_header(s4, "Official Dataset Inspection & Thermodynamic Collinearity")
    tb4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2))
    tf4 = tb4.text_frame
    b4 = [
        ("Official Dataset: ", "data/raw/official_dataset/Analyse Chillers_AI Course.xls"),
        ("Dimensions: ", "525 raw rows x 60 columns (Sheet: Log_Chiller_South)."),
        ("Dual-Compressor Scope: ", "N = 517 observations across Chillers 1, 2, 3, 4, 5."),
        ("Data Quality: ", "0.0% missing cells (100% complete); 0 duplicate rows."),
        ("Collinearity Evidence: ", "Discharge Temp vs Pressure Comp 1: r = +0.9578.\nSuction Sat Temp vs Pressure: r = +0.8973."),
        ("Oral Defense Prompt: ", "[Point to Figure 1 showing strong R134a vapor-pressure correlation]")
    ]
    for b_t, b_d in b4:
        p = tf4.add_paragraph()
        p.space_after = Pt(8)
        r_t = p.add_run()
        r_t.text = b_t
        r_t.font.bold = True
        r_t.font.size = Pt(12)
        r_t.font.color.rgb = c_navy if "Oral" not in b_t else c_amber
        r_d = p.add_run()
        r_d.text = b_d
        r_d.font.size = Pt(12)
        r_d.font.italic = ("Oral" in b_t)
        
    img4_path = "results/figures/final/final_temperature_pressure_correlation.png"
    if os.path.exists(img4_path):
        s4.shapes.add_picture(img4_path, Inches(6.8), Inches(1.6), width=Inches(5.7))

    # SLIDE 5: Preprocessing & Anti-Leakage Protocol
    s5 = prs.slides.add_slide(blank_layout)
    add_slide_header(s5, "Data Preprocessing & Anti-Data-Leakage Protocol")
    tb5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.0))
    tf5 = tb5.text_frame
    b5 = [
        ("28 Core Thermodynamic Features: ", "21 temperatures + 7 refrigerant pressures. Electrical loads, motor currents, run hours, and water pump pressures strictly excluded."),
        ("Chronological Splitting on Plant Hours: ", "Partitioned across 222 unique timestamp hours. Train = 358 (69.25%), Val = 92 (17.79%), Test = 67 (12.96%). Zero temporal overlap."),
        ("Anti-Leakage Safeguard: ", "Random splitting was strictly prohibited to prevent cross-chiller contemporaneous leakage."),
        ("Training-Only Transformer Fitting: ", "SimpleImputer(median) and StandardScaler fitted exclusively on training data. Standardized train mean = 0, std = 1."),
        ("Oral Defense Prompt: ", "[Explain how splitting on unique plant timestamps prevents cross-chiller leakage]")
    ]
    for b_t, b_d in b5:
        p = tf5.add_paragraph()
        p.space_after = Pt(12)
        r_t = p.add_run()
        r_t.text = b_t
        r_t.font.bold = True
        r_t.font.size = Pt(13.5)
        r_t.font.color.rgb = c_navy if "Oral" not in b_t else c_amber
        r_d = p.add_run()
        r_d.text = b_d
        r_d.font.size = Pt(13.5)
        r_d.font.italic = ("Oral" in b_t)

    # SLIDE 6: PCA Model Development & Component Selection
    s6 = prs.slides.add_slide(blank_layout)
    add_slide_header(s6, "PCA Architecture: Scree Plot & Variance Decomposition")
    tb6 = s6.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2))
    tf6 = tb6.text_frame
    b6 = [
        ("Input Dimension: ", "p = 28 standardized thermodynamic features."),
        ("Retained Subspace: ", "k = 12 principal components capturing 85.46% variance."),
        ("Residual Subspace: ", "16 dimensions (14.54% variance) preserved for SPE."),
        ("Kaiser Criterion Comparison: ", "Kaiser rule (lambda >= 1) retains 8 PCs (72.35%), which leaves 27.65% residual variance—too coarse for anomaly detection."),
        ("Mathematical Orthogonality: ", "Max deviation from identity = 1.998e-15."),
        ("Oral Defense Prompt: ", "[Explain why retaining 16 residual dimensions is critical for SPE detection]")
    ]
    for b_t, b_d in b6:
        p = tf6.add_paragraph()
        p.space_after = Pt(8)
        r_t = p.add_run()
        r_t.text = b_t
        r_t.font.bold = True
        r_t.font.size = Pt(12)
        r_t.font.color.rgb = c_navy if "Oral" not in b_t else c_amber
        r_d = p.add_run()
        r_d.text = b_d
        r_d.font.size = Pt(12)
        r_d.font.italic = ("Oral" in b_t)
        
    img6_path = "results/figures/final/final_pca_explained_variance.png"
    if os.path.exists(img6_path):
        s6.shapes.add_picture(img6_path, Inches(6.8), Inches(1.6), width=Inches(5.7))

    # SLIDE 7: Physical Meaning of Principal Components
    s7 = prs.slides.add_slide(blank_layout)
    add_slide_header(s7, "Physical Interpretation of Eigenvector Loadings")
    tb7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2))
    tf7 = tb7.text_frame
    b7 = [
        ("PC1 (23.77% Variance): ", "High-Side Condenser Compression Lift. Positive loadings on discharge saturation temps (+0.320), discharge pressures (+0.301), and lift temps (+0.285)."),
        ("PC2 (13.64% Variance): ", "Low-Side Evaporator Cooling Demand. Positive loadings on chilled water in/out temps (+0.384) and suction pressure (+0.244); negative loading on superheat (-0.231)."),
        ("Domain Alignment: ", "PCA axes naturally align with fundamental thermodynamic cycles rather than arbitrary mathematical directions."),
        ("Oral Defense Prompt: ", "[Walk the examiner through the physical meaning of PC1 vs PC2 loadings]")
    ]
    for b_t, b_d in b7:
        p = tf7.add_paragraph()
        p.space_after = Pt(10)
        r_t = p.add_run()
        r_t.text = b_t
        r_t.font.bold = True
        r_t.font.size = Pt(12)
        r_t.font.color.rgb = c_navy if "Oral" not in b_t else c_amber
        r_d = p.add_run()
        r_d.text = b_d
        r_d.font.size = Pt(12)
        r_d.font.italic = ("Oral" in b_t)
        
    img7_path = "results/figures/final/final_pca_component_loadings.png"
    if os.path.exists(img7_path):
        s7.shapes.add_picture(img7_path, Inches(6.8), Inches(1.5), width=Inches(5.7))

    # SLIDE 8: Clean Test Reconstruction Error & Threshold
    s8 = prs.slides.add_slide(blank_layout)
    add_slide_header(s8, "Reconstruction Error (SPE) & Threshold Calibration")
    tb8 = s8.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2))
    tf8 = tb8.text_frame
    b8 = [
        ("SPE Formula: ", "SPE_i = sum(e_ij^2) across 28 standardized variables."),
        ("Validation Threshold: ", "tau = 9.2931 (95th percentile of N_val = 92)."),
        ("Outlier Robustness: ", "The 99th percentile was corrupted (tau = 1378.35) by an operator typo (322°C saturation temp). 95th percentile resisted this outlier."),
        ("Clean Test Baseline: ", "65/67 (97.01%) normal; 2 natural baseline excursions on Chiller-4 (9.91) and Chiller-5 (13.60)."),
        ("Oral Defense Prompt: ", "[Explain why the 95th percentile was chosen over the corrupted 99th percentile]")
    ]
    for b_t, b_d in b8:
        p = tf8.add_paragraph()
        p.space_after = Pt(8)
        r_t = p.add_run()
        r_t.text = b_t
        r_t.font.bold = True
        r_t.font.size = Pt(12)
        r_t.font.color.rgb = c_navy if "Oral" not in b_t else c_amber
        r_d = p.add_run()
        r_d.text = b_d
        r_d.font.size = Pt(12)
        r_d.font.italic = ("Oral" in b_t)
        
    img8_path = "results/figures/final/final_clean_test_reconstruction_error.png"
    if os.path.exists(img8_path):
        s8.shapes.add_picture(img8_path, Inches(6.8), Inches(1.6), width=Inches(5.7))

    # SLIDE 9: Synthetic Anomaly Experiments Design
    s9 = prs.slides.add_slide(blank_layout)
    add_slide_header(s9, "Controlled Synthetic Anomaly Experiments Design")
    tb9 = s9.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.0))
    tf9 = tb9.text_frame
    b9 = [
        ("Why Synthetic Experiments? ", "Operational chiller logs lack verified ground-truth fault work orders. Supervised metrics (Precision, Recall, Latency) require controlled ground truth."),
        ("Scenario A (Sudden Sensor Spikes): ", "Injected +-4.0 sigma deviations into 3 discrete observations (Discharge Temp, Discharge Press, Suction Temp). Tests acute gross malfunction."),
        ("Scenario B (Gradual Sensor Drift): ", "Linear ramp from +0.7 to +3.5 sigma across 5 continuous hours in Chiller-1 (Evaporator Water Out Temp). Tests incipient boundary and detection latency."),
        ("Scenario C (Relationship Breaking): ", "Decoupled discharge pressure (-3.5 sigma) while leaving discharge temp normal over 3 hours. Directly targets the research question."),
        ("Oral Defense Prompt: ", "[Emphasize that all scenarios were derived independently from the clean reference]")
    ]
    for b_t, b_d in b9:
        p = tf9.add_paragraph()
        p.space_after = Pt(10)
        r_t = p.add_run()
        r_t.text = b_t
        r_t.font.bold = True
        r_t.font.size = Pt(13)
        r_t.font.color.rgb = c_navy if "Oral" not in b_t else c_amber
        r_d = p.add_run()
        r_d.text = b_d
        r_d.font.size = Pt(13)
        r_d.font.italic = ("Oral" in b_t)

    # SLIDE 10: Model Evaluation Results & Confusion Matrices
    s10 = prs.slides.add_slide(blank_layout)
    add_slide_header(s10, "Evaluation Metrics & Confusion Matrix Analysis")
    tb10 = s10.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2))
    tf10 = tb10.text_frame
    b10 = [
        ("Scenario A (Spikes): ", "Precision: 0.60 | Recall: 1.00 | F1: 0.75 | Event Recall: 100% | Delay: 0h."),
        ("Scenario B (Drift): ", "Precision: 0.33 | Recall: 0.20 | F1: 0.25 | Event Recall: 100% | Delay: 4h."),
        ("Scenario C (Rel Breaking): ", "Precision: 0.60 | Recall: 1.00 | F1: 0.75 | Event Recall: 100% | Delay: 0h."),
        ("Why FP = 2? ", "The 2 false positives match the 2 clean baseline excursions. Net synthetic false positives = ZERO."),
        ("Oral Defense Prompt: ", "[Explain the observation recall vs event recall trade-off in drift]")
    ]
    for b_t, b_d in b10:
        p = tf10.add_paragraph()
        p.space_after = Pt(8)
        r_t = p.add_run()
        r_t.text = b_t
        r_t.font.bold = True
        r_t.font.size = Pt(12)
        r_t.font.color.rgb = c_navy if "Oral" not in b_t else c_amber
        r_d = p.add_run()
        r_d.text = b_d
        r_d.font.size = Pt(12)
        r_d.font.italic = ("Oral" in b_t)
        
    img10_path = "results/figures/final/final_confusion_matrices.png"
    if os.path.exists(img10_path):
        s10.shapes.add_picture(img10_path, Inches(6.8), Inches(2.0), width=Inches(5.7))

    # SLIDE 11: Answering the Research Question
    s11 = prs.slides.add_slide(blank_layout)
    add_slide_header(s11, "Research Question Resolution: Relationship Breaking")
    tb11 = s11.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2))
    tf11 = tb11.text_frame
    b11 = [
        ("Direct Answer: ", "YES. Anomalies can be detected from correlated temperature and pressure variables with high sensitivity."),
        ("Thermodynamic Decoupling: ", "Perturbing discharge pressure (-3.5 sigma) while holding discharge temp normal violated R134a vapor-pressure equilibrium."),
        ("Residual Explosion: ", "SPE jumped from clean baseline ~1.65 to 9.54–9.74, breaching threshold across 100% of observations with 0 hours latency."),
        ("Sensor Attribution: ", "Residual contributions correctly identified the decoupled feature."),
        ("Oral Defense Prompt: ", "[Conclude firmly on how Scenario C proves the assigned research question]")
    ]
    for b_t, b_d in b11:
        p = tf11.add_paragraph()
        p.space_after = Pt(8)
        r_t = p.add_run()
        r_t.text = b_t
        r_t.font.bold = True
        r_t.font.size = Pt(12)
        r_t.font.color.rgb = c_navy if "Oral" not in b_t else c_amber
        r_d = p.add_run()
        r_d.text = b_d
        r_d.font.size = Pt(12)
        r_d.font.italic = ("Oral" in b_t)
        
    img11_path = "results/figures/final/final_relationship_breaking_analysis.png"
    if os.path.exists(img11_path):
        s11.shapes.add_picture(img11_path, Inches(6.8), Inches(1.6), width=Inches(5.7))

    # SLIDE 12: Academic Limitations & Conclusion
    s12 = prs.slides.add_slide(blank_layout)
    add_slide_header(s12, "Summary, Limitations & Future Engineering Work")
    tb12 = s12.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.0))
    tf12 = tb12.text_frame
    b12 = [
        ("Summary of Contributions: ", "Built end-to-end PCA detector on 517 dual-compressor observations; enforced zero data leakage via chronological hourly splitting; verified 85.46% variance across 12 PCs; achieved 100% event detection across all synthetic benchmarks."),
        ("Empirical Limitations: ", "Finite test sample (N_test = 67, 34 hours); static linear PCA assumption; absence of real physical fault maintenance logs."),
        ("What Cannot Be Claimed: ", "Synthetic detection does not prove real-world fault diagnosis without physical maintenance logs."),
        ("Future Research Extensions: ", "Dynamic PCA (DPCA) to incorporate time-lagged dynamics; non-linear autoencoders for broad seasonal transitions; chiller-specific multi-model baselines."),
        ("Oral Defense Prompt: ", "[Conclude with confidence, thanking the committee and welcoming viva questions]")
    ]
    for b_t, b_d in b12:
        p = tf12.add_paragraph()
        p.space_after = Pt(8)
        r_t = p.add_run()
        r_t.text = b_t
        r_t.font.bold = True
        r_t.font.size = Pt(12)
        r_t.font.color.rgb = c_navy if "Oral" not in b_t else c_amber
        r_d = p.add_run()
        r_d.text = b_d
        r_d.font.size = Pt(12)
        r_d.font.italic = ("Oral" in b_t)

    pptx_path = "presentation/Chiller_Anomaly_Detection_Template.pptx"
    os.makedirs("presentation", exist_ok=True)
    prs.save(pptx_path)
    print(f"Successfully generated 12-slide presentation template at: {pptx_path}")

if __name__ == "__main__":
    create_presentation_template()
