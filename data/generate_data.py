import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

def generate_telemetry(days=14, interval_minutes=5):
    """Generate dramatic data center telemetry with 4 compelling scenarios"""
    
    print("Generating compelling telemetry scenarios...")
    
    start_time = datetime.now() - timedelta(days=days)
    timestamps = [start_time + timedelta(minutes=i*interval_minutes) 
                  for i in range(days * 24 * (60//interval_minutes))]
    
    data = []
    for ts in timestamps:
        day_offset = (ts - start_time).days
        hour = ts.hour
        
        # Baseline normal operation
        temp_rack_47 = 24.0 + np.random.normal(0, 0.3)
        temp_avg = 24.5 + np.random.normal(0, 0.4)
        power_it = 520 + np.random.normal(0, 15)
        power_cooling = 220 + np.random.normal(0, 8)
        crac_12_fan_speed = 90.0 + np.random.normal(0, 1)
        chiller_2_cop = 3.5 + np.random.normal(0, 0.1)
        cooling_mode = "auto"
        workload_cluster_b = 45 + np.random.normal(0, 5)  # percent utilization
        
        # SCENARIO 1: Equipment Failure Prevention (Days 8-13)
        # CRAC-12 fan degrading, causing Rack 47 temperature rise
        if 8 <= day_offset <= 13:
            degradation = (day_offset - 8) / 5.0  # 0 to 1
            crac_12_fan_speed = 90 - (degradation * 8)  # 90% -> 82%
            temp_rack_47 = 24.0 + (degradation * 3.5)  # 24C -> 27.5C
            
            # Critical alert threshold on day 13
            if day_offset == 13 and 14 <= hour <= 16:
                temp_rack_47 = 27.8 + np.random.normal(0, 0.2)
                crac_12_fan_speed = 82 + np.random.normal(0, 0.5)
        
        # SCENARIO 2: Mystery Cost Spike (Days 3-10)
        # Dev cluster left running at full load
        if 3 <= day_offset <= 10:
            workload_cluster_b = 95 + np.random.normal(0, 2)  # Stuck at 95%
            power_it = 520 + 180  # +35% power consumption
            power_cooling = 220 + 60  # Cooling works harder
        
        # SCENARIO 3: Proactive Optimization Opportunity
        # Nighttime overcooling (every night)
        if 22 <= hour or hour <= 6:  # 10PM - 6AM
            workload_cluster_b = 35 + np.random.normal(0, 3)  # Low load at night
            power_it = 520 - 180  # Low IT load
            # But cooling still runs at full power (inefficient)
            power_cooling = 220 + np.random.normal(0, 8)  # Should be lower
        
        # SCENARIO 4: Chiller Degradation (Days 9-14)
        # Chiller-2 efficiency dropping (early warning)
        if 9 <= day_offset <= 14:
            degradation = (day_offset - 9) / 5.0
            chiller_2_cop = 3.5 - (degradation * 0.45)  # 3.5 -> 3.05 COP
            # Subtle power increase as chiller works harder
            power_cooling = power_cooling + (degradation * 25)
        
        # Calculate PUE
        pue = (power_it + power_cooling) / power_it if power_it > 0 else 1.4
        
        data.append({
            "timestamp": ts.isoformat(),
            "temp_rack_47": round(temp_rack_47, 2),
            "temp_avg": round(temp_avg, 2),
            "power_it_kw": round(power_it, 2),
            "power_cooling_kw": round(power_cooling, 2),
            "pue": round(pue, 3),
            "crac_12_fan_speed_pct": round(crac_12_fan_speed, 1),
            "chiller_2_cop": round(chiller_2_cop, 2),
            "cooling_mode": cooling_mode,
            "workload_cluster_b_pct": round(workload_cluster_b, 1)
        })
    
    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/telemetry.csv", index=False)
    
    print("✓ Generated " + str(len(df)) + " telemetry records with 4 scenarios")
    print("  - Scenario 1: CRAC-12 fan degradation (Days 8-13)")
    print("  - Scenario 2: Mystery cost spike (Days 3-10)")
    print("  - Scenario 3: Night overcooling (Daily)")
    print("  - Scenario 4: Chiller-2 degradation (Days 9-14)")
    
    return df

def generate_logs():
    """Generate dramatic system logs matching the scenarios"""
    
    print("\nGenerating system logs...")
    
    now = datetime.now()
    
    # Helper to format dates
    def days_ago(days, hour=12, minute=0):
        return (now - timedelta(days=days)).replace(hour=hour, minute=minute).strftime("%Y-%m-%dT%H:%M:%S")
    
    logs = [
        # Normal operations
        {"time": days_ago(14, 9, 15), "level": "INFO", "message": "System initialization complete. All systems nominal."},
        {"time": days_ago(13, 10, 30), "level": "INFO", "message": "Scheduled maintenance completed on CRAC units 1-11"},
        
        # SCENARIO 2: Dev cluster anomaly
        {"time": days_ago(10, 16, 45), "level": "INFO", "message": "Cluster B: ML training job started by user dev-team@company.com"},
        {"time": days_ago(10, 16, 46), "level": "INFO", "message": "Cluster B workload increased to 95% utilization"},
        {"time": days_ago(9, 2, 15), "level": "WARN", "message": "Cluster B has been at >90% utilization for 24 hours"},
        {"time": days_ago(5, 8, 30), "level": "WARN", "message": "Power consumption 22% above baseline for 5 consecutive days"},
        {"time": days_ago(4, 14, 0), "level": "INFO", "message": "Finance team requested power usage investigation"},
        
        # SCENARIO 4: Chiller degradation
        {"time": days_ago(9, 11, 20), "level": "WARN", "message": "Chiller-2 COP dropped below 3.4 (normal: 3.5)"},
        {"time": days_ago(8, 15, 45), "level": "INFO", "message": "Maintenance checked Chiller-2, no obvious issues found"},
        {"time": days_ago(7, 9, 10), "level": "WARN", "message": "Chiller-2 COP continues declining, now at 3.3"},
        {"time": days_ago(5, 13, 30), "level": "WARN", "message": "Slight condensation detected in Chiller-2 coolant pipes"},
        {"time": days_ago(3, 10, 15), "level": "INFO", "message": "Historical analysis: Similar pattern preceded Chiller-1 failure 6 months ago"},
        
        # SCENARIO 1: CRAC-12 fan degradation (CRITICAL)
        {"time": days_ago(8, 14, 0), "level": "INFO", "message": "CRAC-12 fan speed 90%, within normal range"},
        {"time": days_ago(7, 10, 30), "level": "WARN", "message": "CRAC-12 fan speed decreased to 88%"},
        {"time": days_ago(6, 15, 20), "level": "WARN", "message": "CRAC-12 fan speed 86%, trending downward"},
        {"time": days_ago(5, 9, 45), "level": "WARN", "message": "Rack 47 inlet temperature 25.2C (threshold: 27C)"},
        {"time": days_ago(4, 11, 10), "level": "WARN", "message": "CRAC-12 fan speed 84%, possible bearing wear"},
        {"time": days_ago(3, 16, 30), "level": "ERROR", "message": "CRAC-12 fan speed 82%, Rack 47 temperature 26.8C"},
        {"time": days_ago(2, 14, 15), "level": "ERROR", "message": "Rack 47 temperature 27.4C - approaching critical threshold"},
        {"time": days_ago(1, 14, 45), "level": "CRITICAL", "message": "URGENT: Rack 47 temperature 27.8C, CRAC-12 fan at 82%. Failure imminent."},
        {"time": days_ago(1, 15, 0), "level": "CRITICAL", "message": "Automatic alert sent to facilities team"},
        {"time": days_ago(1, 15, 30), "level": "INFO", "message": "Emergency maintenance dispatched to CRAC-12"},
        
        # SCENARIO 3: Efficiency observation
        {"time": days_ago(12, 3, 0), "level": "INFO", "message": "Night shift: IT load at 35%, cooling at 100% capacity"},
        {"time": days_ago(6, 2, 30), "level": "INFO", "message": "Energy audit: Cooling efficiency low during night hours (10PM-6AM)"},
    ]
    
    os.makedirs("data/logs", exist_ok=True)
    
    with open("data/logs/system.log", "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")
    
    print("✓ Generated " + str(len(logs)) + " log entries with critical alerts")

def generate_manuals():
    """Generate comprehensive equipment documentation"""
    
    print("\nGenerating equipment manuals...")
    
    manual = """# Data Center Operations Manual

## Equipment Specifications

### CRAC Units (Computer Room Air Conditioning)
- Model: Liebert DS 100kW
- Optimal fan speed: 85-95%
- Temperature setpoint: 18-22°C
- **CRITICAL**: Fan speed below 85% indicates bearing failure risk
- **CRITICAL**: Inlet temperature above 27°C risks equipment damage

### Chiller System
- Model: Carrier 30RB400
- Optimal COP (Coefficient of Performance): 3.4-3.6
- **WARNING**: COP below 3.2 indicates efficiency loss
- **PATTERN**: COP degradation often precedes mechanical failure (30-60 days)
- Maintenance interval: Every 6 months

## Optimal Operating Parameters

### Temperature Management
- Inlet air temperature: 24-26°C (optimal balance)
- Hot aisle: <32°C
- Cold aisle: 18-22°C
- **Every 1°C increase above 24°C saves 3-5% cooling energy**
- **Above 27°C risks equipment lifespan reduction**

### Cooling Efficiency
- PUE target: <1.4 (world-class: 1.2-1.3)
- Cooling mode: AUTO (manual mode reduces efficiency 10-15%)
- Night operations: Consider raising setpoint 2°C when load <50%

### Workload Management
- Cluster utilization >90% sustained: Check for runaway jobs
- Night shift typical load: 30-40%
- Peak hours: 9AM-6PM (60-75% average)

## Common Issues and Solutions

### High PUE (>1.5)
**Causes:**
- Cooling in manual mode
- Chiller setpoint too low (overcooling)
- Workload imbalance across racks
- CRAC unit malfunction

**Solutions:**
- Return to automatic cooling mode
- Increase setpoint during low-load periods
- Redistribute workloads
- Inspect CRAC units for degradation

### Rack Temperature Anomalies
**Investigation steps:**
1. Check CRAC unit serving that zone
2. Verify fan speed and bearing condition (listen for noise)
3. Inspect air filters for blockage
4. Check workload distribution

**Critical actions:**
- Temperature >27°C: Immediate investigation required
- Temperature >29°C: Emergency response (risk of hardware damage)

### Cost Anomalies
**Common causes:**
- Forgotten dev/test clusters at full load
- Cooling system inefficiency
- Workload migrations without capacity planning
- Seasonal HVAC changes

**Analysis approach:**
1. Compare power consumption to baseline
2. Cross-reference with workload logs
3. Identify deployment/config changes
4. Calculate cost per workload segment

## Preventive Maintenance

### Weekly Checks
- CRAC fan speeds and temperatures
- Chiller COP and pressure readings
- Unusual log patterns or alerts

### Monthly Analysis
- PUE trends
- Cost per workload analysis
- Equipment degradation patterns

### Proactive Indicators
**CRAC Failure Warning Signs:**
- Fan speed declining >5% over 3 days
- Temperature rising in served zone
- Unusual vibration or noise

**Chiller Failure Warning Signs:**
- COP declining >10% from baseline
- Condensation in coolant pipes
- Pressure fluctuations

## Energy Optimization Opportunities

### Time-Based Optimization
- **Night hours (10PM-6AM)**: Raise setpoint 2°C when load <50%
- **Potential savings**: 12-15% cooling energy = $10K-15K monthly

### Workload Optimization
- Auto-scale dev clusters during off-hours
- Schedule batch jobs during cooler ambient temperatures
- Consolidate VMs to reduce sprawl

### Seasonal Adjustments
- **Summer**: Consider pre-cooling during night
- **Winter**: Utilize free cooling (economizer mode)
- **Spring/Fall**: Optimal efficiency period

## Historical Incidents

### Chiller-1 Failure (6 months ago)
- **Warning pattern**: COP declined from 3.5 to 3.1 over 4 weeks
- **Ignored signs**: Condensation in coolant pipes
- **Result**: Emergency replacement $150K + 8hr downtime
- **Lesson**: COP <3.2 requires immediate investigation

### Rack 23 Overheating (1 year ago)
- **Cause**: CRAC-7 fan bearing failure
- **Warning**: Fan speed declined 90% to 78% over 5 days
- **Result**: 3 servers damaged ($25K hardware loss)
- **Lesson**: Fan speed <85% = emergency maintenance

## Emergency Response Procedures

### Critical Temperature Alert (>27°C)
1. Identify affected racks and CRAC unit
2. Check CRAC fan speed and status
3. If CRAC malfunction: Migrate workloads immediately
4. Dispatch maintenance within 1 hour
5. Document incident and response time

### Equipment Failure Prediction
- **Use historical patterns to predict failures 1-2 weeks early**
- **Schedule maintenance during planned windows**
- **Avoid costly emergency responses**

## Cost Savings Opportunities

### Identified Optimizations
1. **Night cooling reduction**: $12K-15K monthly
2. **Workload auto-scaling**: $10K-20K monthly  
3. **Predictive maintenance**: Avoid $100K-200K emergency costs annually
4. **PUE improvement 1.45→1.30**: $180K annually

### ROI Calculations
- **Downtime cost**: $8,000-12,000 per hour
- **Emergency maintenance**: 3-5x normal cost
- **Equipment replacement**: Plan vs Emergency = 40% cost difference
"""
    
    os.makedirs("data/manuals", exist_ok=True)
    
    with open("data/manuals/datacenter_operations.md", "w", encoding="utf-8") as f:
        f.write(manual)
    
    print("✓ Generated comprehensive operations manual with cost analysis")

def generate_workload_history():
    """Generate workload deployment history for cost investigation"""
    
    print("\nGenerating workload history...")
    
    now = datetime.now()
    
    deployments = [
        {
            "timestamp": (now - timedelta(days=10, hours=16, minutes=45)).isoformat(),
            "cluster": "Cluster B",
            "action": "deploy",
            "workload": "ml-training-resnet50",
            "requested_by": "dev-team@company.com",
            "cpu_cores": 256,
            "expected_duration": "4 hours",
            "auto_terminate": False,
            "notes": "Experimental ML training job"
        },
        {
            "timestamp": (now - timedelta(days=14, hours=9)).isoformat(),
            "cluster": "Cluster A",
            "action": "deploy",
            "workload": "web-service-prod",
            "requested_by": "ops-team@company.com",
            "cpu_cores": 128,
            "expected_duration": "continuous",
            "auto_terminate": False,
            "notes": "Production web services"
        }
    ]
    
    with open("data/workload_history.json", "w", encoding="utf-8") as f:
        json.dump(deployments, f, indent=2)
    
    print("✓ Generated workload deployment history")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("IMPRESSIVE DATA CENTER SCENARIO GENERATOR")
    print("="*60 + "\n")
    
    generate_telemetry()
    generate_logs()
    generate_manuals()
    generate_workload_history()
    
    print("\n" + "="*60)
    print("✓ ALL DATA GENERATED - READY FOR DEMO!")
    print("="*60)
    print("\nScenarios created:")
    print("1. Equipment failure prevention (CRAC-12)")
    print("2. Mystery cost spike investigation (Cluster B)")
    print("3. Proactive optimization (Night overcooling)")
    print("4. Cascade failure prediction (Chiller-2)")
    print("\nNext: Run indexer to load into vector store")
    print("="*60 + "\n")