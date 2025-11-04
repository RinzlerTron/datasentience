import os
import pandas as pd
import json
from src.vector_store import optimized_vector_store

def index_all_data():
    """Load and index all data into vector store"""
    
    print("\n" + "="*60)
    print("INDEXING DATA INTO VECTOR STORE")
    print("="*60 + "\n")
    
    texts = []
    metadata = []
    
    # Index telemetry (summarize by hour)
    print("[1/3] Indexing telemetry data...")
    df = pd.read_csv("data/telemetry.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.floor("h")  # Use 'h' instead of deprecated 'H'
    
    hourly = df.groupby("hour").agg({
        "temp_avg": "mean",
        "temp_rack_47": "mean",
        "power_it_kw": "mean",
        "power_cooling_kw": "mean",
        "pue": "mean",
        "crac_12_fan_speed_pct": "mean",
        "chiller_2_cop": "mean",
        "cooling_mode": "first",
        "workload_cluster_b_pct": "mean"
    }).reset_index()
    
    for _, row in hourly.iterrows():
        text = "At {}: Avg temp {:.1f}C, Rack 47 temp {:.1f}C, PUE {:.2f}, CRAC-12 fan {:.1f}%, Chiller-2 COP {:.2f}, Cluster B load {:.1f}%, cooling mode: {}".format(
            row["hour"],
            row["temp_avg"],
            row["temp_rack_47"],
            row["pue"],
            row["crac_12_fan_speed_pct"],
            row["chiller_2_cop"],
            row["workload_cluster_b_pct"],
            row["cooling_mode"]
        )
        texts.append(text)
        metadata.append({"type": "telemetry", "timestamp": str(row["hour"])})
    
    print("  Added " + str(len(hourly)) + " telemetry summaries")
    
    # Index logs (optional - skip if file doesn't exist)
    print("\n[2/3] Indexing system logs...")
    try:
        log_path = "data/logs/system.log"
        if os.path.exists(log_path):
            with open(log_path) as f:
                for line in f:
                    log = json.loads(line)
                    text = "[{}] {}: {}".format(log["level"], log["time"], log["message"])
                    texts.append(text)
                    metadata.append({"type": "log", "timestamp": log["time"]})
            print("  Added system log entries")
        else:
            print("  System log file not found, skipping log indexing")
    except Exception as e:
        print(f"  Warning: Could not index system logs: {e}")
    
    # Index manual
    print("\n[3/4] Indexing equipment manuals...")
    with open("data/manuals/datacenter_operations.md") as f:
        manual = f.read()
        # Split into sections
        sections = manual.split("\n## ")
        for section in sections:
            if section.strip():
                texts.append(section)
                metadata.append({"type": "manual"})
    
    print("  Added manual sections")
    
    # Index workload history
    print("\n[4/4] Indexing workload deployment history...")
    with open("data/workload_history.json") as f:
        deployments = json.load(f)
        for dep in deployments:
            text = "Deployment on {}: {} to {} by {} - {} cores, auto_terminate: {}. Notes: {}".format(
                dep["timestamp"],
                dep["workload"],
                dep["cluster"],
                dep["requested_by"],
                dep["cpu_cores"],
                dep["auto_terminate"],
                dep.get("notes", "")
            )
            texts.append(text)
            metadata.append({"type": "deployment", "timestamp": dep["timestamp"]})
    
    print("  Added deployment records")
    
    # Index everything
    print("\n" + "-"*60)
    optimized_vector_store.add_documents(texts, metadata)
    
    print("\n" + "="*60)
    print("✓ INDEXING COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    index_all_data()