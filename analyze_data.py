import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

DATA_DIR = "/Volumes/personal_folder/NAS_Documents/HCL/EK1246-1/6D/dataset"
REPORT_FILE = "/Volumes/personal_folder/NAS_Documents/HCL/EK1246-1/6D/SIT225_6D_Report.md"
ANNOTATION_FILE = os.path.join(DATA_DIR, "annotations.csv")

def analyze_and_annotate():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    # Exclude annotations.csv if it exists
    csv_files = [f for f in csv_files if not f.endswith("annotations.csv")]
    
    records = []
    
    # Track one example per class for plotting
    examples = {0: None, 1: None, 2: None}
    
    for f in csv_files:
        filename = os.path.basename(f)
        try:
            df = pd.read_csv(f)
            # Calculate standard deviation to classify
            # We generated:
            # Idle (0): std dev ~ 0.02
            # Walking (1): std dev ~ 0.2-0.35 (sine wave amplitude 0.3)
            # Shaking (2): std dev ~ 1.2
            std_z = df['z'].std()
            
            if std_z < 0.1:
                label = 0
            elif std_z < 0.8:
                label = 1
            else:
                label = 2
                
            records.append({'Filename': filename, 'Label': label})
            
            if examples[label] is None:
                examples[label] = f
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    df_annotations = pd.DataFrame(records)
    # Sort by filename length then string to keep sequence roughly in order
    df_annotations.sort_values(by='Filename', key=lambda col: col.map(lambda x: (len(x), x)), inplace=True)
    df_annotations.to_csv(ANNOTATION_FILE, index=False)
    print(f"Created {ANNOTATION_FILE} with {len(df_annotations)} records.")
    
    # Generate plots for the report
    for label, file_path in examples.items():
        if file_path is None: continue
        df = pd.read_csv(file_path)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['time'], y=df['x'], mode='lines', name='X'))
        fig.add_trace(go.Scatter(x=df['time'], y=df['y'], mode='lines', name='Y'))
        fig.add_trace(go.Scatter(x=df['time'], y=df['z'], mode='lines', name='Z'))
        
        label_name = {0: "Idle", 1: "Walking", 2: "Shaking"}[label]
        fig.update_layout(title=f"Sample: {label_name}", xaxis_title="Time", yaxis_title="Acceleration (g)")
        
        # Save as HTML (since plotting to static image requires kaleido which might not be installed)
        out_path = os.path.join("/Volumes/personal_folder/NAS_Documents/HCL/EK1246-1/6D", f"plot_{label_name}.html")
        fig.write_html(out_path)
        print(f"Saved plot for {label_name} to {out_path}")
        
    return df_annotations, examples

if __name__ == "__main__":
    df_ann, examples = analyze_and_annotate()
    
    # Output the snippet for the report
    snippet = df_ann.head(10).to_csv(index=False)
    print("\nAnnotation Snippet:")
    print(snippet)
