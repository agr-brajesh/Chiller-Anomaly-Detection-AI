import re
import os

docs = [
    "documentation/dataset_profile_working.md",
    "documentation/phase1_decision_log.md",
    "documentation/phase2_eda_working.md",
    "documentation/phase3_preprocessing_working.md",
    "documentation/phase4_pca_working.md",
    "README.md"
]

print("CHECKING FILE REFERENCES IN DOCUMENTATION:")
for doc in docs:
    if not os.path.exists(doc):
        print(f"Missing doc: {doc}")
        continue
    with open(doc, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Find all results/ or data/ paths
    matches = re.findall(r'(?:data|results|documentation)/[a-zA-Z0-9_\-/\.]+', text)
    matches = set(matches)
    broken = []
    for m in matches:
        # Strip trailing punctuation
        m_clean = m.rstrip(".,;:)`*\"'")
        if not os.path.exists(m_clean):
            broken.append(m_clean)
    print(f"\n{doc}: Total paths found: {len(matches)}, Broken paths: {len(broken)}")
    for b in broken:
        print(f"  [BROKEN]: {b}")

