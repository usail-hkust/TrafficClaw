import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
from pathlib import Path

def load_sumo_flow_data(file_path):
    """Load traffic flow data from SUMO XML file."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    vehicles = []
    # Parse vehicle elements
    for vehicle in root.findall('.//vehicle'):
        depart_time = float(vehicle.get('depart', 0))
        vehicles.append({
            'startTime': depart_time  # Convert to seconds
        })

    return vehicles

def analyze_time_span(flow_data):
    """Analyze the time span of the traffic data."""
    start_times = [vehicle['startTime'] for vehicle in flow_data]
    if not start_times:
        return 0, 0, 0

    min_time = min(start_times)
    max_time = max(start_times)
    total_hours = (max_time - min_time) / 3600  # Convert seconds to hours

    return min_time, max_time, total_hours

def calculate_hourly_flows(flow_data, min_time, max_time):
    """Calculate average hourly flows."""
    # Initialize counters for each hour (0-23)
    hourly_counts = np.zeros(24)
    days_count = np.zeros(24)  # Track number of days for each hour

    # Count vehicles for each hour
    for vehicle in flow_data:
        start_time = vehicle['startTime']
        # Convert seconds to hours since start
        hour = int((start_time - min_time) / 3600) % 24
        day = int((start_time - min_time) / (3600 * 24))

        hourly_counts[hour] += 1
        days_count[hour] = max(days_count[hour], day + 1)

    # Calculate averages
    hourly_averages = np.zeros(24)
    for hour in range(24):
        if days_count[hour] > 0:
            hourly_averages[hour] = hourly_counts[hour] / days_count[hour]
        else:
            hourly_averages[hour] = hourly_counts[hour]

    return hourly_averages

def plot_hourly_flows(hourly_flows, output_path=None, title="Average Hourly Traffic Flow"):
    """Create visualization of hourly flows."""
    plt.figure(figsize=(12, 6))
    hours = range(24)

    plt.bar(hours, hourly_flows, color='skyblue', alpha=0.7)
    plt.plot(hours, hourly_flows, 'r-', linewidth=2)
    plt.scatter(hours, hourly_flows, color='red', zorder=5)

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Number of Vehicles', fontsize=12)
    plt.ylim(0, max(hourly_flows)*1.5)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(hours)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add value labels on top of each bar
    for i, v in enumerate(hourly_flows):
        plt.text(i, v + max(hourly_flows)*0.02, f'{v:.1f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # if output_path:
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #     print(f"Plot saved to: {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize SUMO traffic flow data')
    parser.add_argument('--input_file', type=str, default="../sumo_config_highway/Southern_Bronx/routes.background.rou.xml",
                      help='Path to the input SUMO route/trip XML file')
    parser.add_argument('--output', default="../../LLMTSCS/figures/mzwmap_0905_0102.png", type=str, help='Path to save the output plot (optional)')
    
    args = parser.parse_args()
    
    # Load and process data
    flow_data = load_sumo_flow_data(args.input_file)
    min_time, max_time, total_hours = analyze_time_span(flow_data)
    
    if not flow_data:
        print("No vehicle data found in the input file.")
        return
    
    # Calculate hourly flows
    hourly_flows = calculate_hourly_flows(flow_data, min_time, max_time)
    
    # Determine if data spans multiple days
    days_span = total_hours / 24
    title = f"SUMO Traffic Flow Analysis"
    if days_span > 1:
        title += f" (Averaged over {int(days_span)} days)"
    
    # Create visualization
    output_path = args.output if args.output else None
    plot_hourly_flows(hourly_flows, output_path, title)
    
    # Print summary statistics
    print(f"\nTraffic Flow Analysis Summary:")
    print(f"Total time span: {total_hours:.2f} hours ({days_span:.1f} days)")
    print(f"Total vehicles: {len(flow_data)}")
    print(f"Peak hour: {np.argmax(hourly_flows):02d}:00 with average {np.max(hourly_flows):.1f} vehicles")
    print(f"Minimum flow hour: {np.argmin(hourly_flows):02d}:00 with average {np.min(hourly_flows):.1f} vehicles")
    print(f"Average vehicles per hour: {np.mean(hourly_flows):.1f}")

if __name__ == "__main__":
    main()
