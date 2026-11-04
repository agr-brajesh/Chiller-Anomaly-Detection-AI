"""
Script to assemble the final submission directory and ZIP package.
Student Register Number: 24BCE0551
Track: Multivariate and Time-Series Detection
Assigned Method: PCA-Based Multivariate Anomaly Detector
"""

import os
import shutil
import zipfile

def assemble_submission():
    base_dir = "FINAL_SUBMISSION"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        
    os.makedirs(f"{base_dir}/Implementation/src", exist_ok=True)
    os.makedirs(f"{base_dir}/Documentation", exist_ok=True)
    os.makedirs(f"{base_dir}/Visualizations", exist_ok=True)
    os.makedirs(f"{base_dir}/Presentation", exist_ok=True)
    
    # 1. Implementation
    shutil.copy("notebooks/Chiller_PCA_Anomaly_Detection.ipynb", f"{base_dir}/Implementation/")
    shutil.copy("requirements.txt", f"{base_dir}/Implementation/")
    shutil.copy("README.md", f"{base_dir}/Implementation/")
    
    src_files = [
        "src/data_inspection.py",
        "src/verify_modeling_scope.py",
        "src/eda_analysis.py",
        "src/preprocessing.py",
        "src/pca_detector.py",
        "src/anomaly_detection_phase5.py",
        "src/synthetic_experiments_phase6.py",
        "src/final_results_phase7.py",
        "src/build_notebook.py",
        "src/generate_report_docx.py",
        "src/generate_presentation_pptx.py"
    ]
    for sf in src_files:
        if os.path.exists(sf):
            shutil.copy(sf, f"{base_dir}/Implementation/src/")
            
    # 2. Documentation
    doc_files = [
        "documentation/preprocessing_notes.md",
        "documentation/model_configuration.md",
        "documentation/evaluation_and_interpretation.md",
        "documentation/ai_assistance_log.md",
        "documentation/viva_preparation.md",
        "documentation/phase7_visualization_index.md",
        "documentation/phase7_results_analysis_working.md"
    ]
    for df in doc_files:
        if os.path.exists(df):
            shutil.copy(df, f"{base_dir}/Documentation/")
            
    # 3. Visualizations (12 final figures)
    fig_dir = "results/figures/final"
    for fig_file in os.listdir(fig_dir):
        if fig_file.endswith(".png"):
            shutil.copy(os.path.join(fig_dir, fig_file), f"{base_dir}/Visualizations/")
            
    # 4. Report Template
    shutil.copy("Case_Study_Report_Template.docx", f"{base_dir}/")
    
    # 5. Presentation
    shutil.copy("presentation/Chiller_Anomaly_Detection_Template.pptx", f"{base_dir}/Presentation/")
    
    print(f"Assembled complete folder hierarchy in {base_dir}/")
    
    # 6. Create ZIP archive
    zip_name = "FINAL_SUBMISSION_24BCE0551.zip"
    if os.path.exists(zip_name):
        os.remove(zip_name)
        
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                abs_file = os.path.join(root, file)
                rel_path = os.path.relpath(abs_file, base_dir)
                zipf.write(abs_file, arcname=f"FINAL_SUBMISSION_24BCE0551/{rel_path}")
                
    zip_size = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"Created clean submission archive: {zip_name} ({zip_size:.2f} MB)")

if __name__ == "__main__":
    assemble_submission()

