#!/usr/bin/env python3
"""
Script to add bus flow definitions to ptflows.rou.xml.

This script reads the existing ptflows.rou.xml file and adds <flow> elements
for bus routes that are missing flow definitions. This is necessary because
passengers are generated with specific intended vehicle IDs (e.g., bus_M60_SBS:1.1)
that expect buses to be dispatched at regular intervals.

Usage:
    python add_bus_flows_to_ptflows.py /path/to/ptflows.rou.xml [--period 600] [--output /path/to/output.xml]
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_existing_flows(root):
    """Parse existing flow definitions to avoid duplicates."""
    existing_flows = set()
    for flow in root.findall('.//flow'):
        flow_id = flow.get('id')
        if flow_id:
            existing_flows.add(flow_id)
    return existing_flows


def parse_bus_routes(root):
    """Parse bus route definitions that need flow elements."""
    bus_routes = []
    for route in root.findall('.//route'):
        route_id = route.get('id')
        if route_id and route_id.startswith('bus_'):
            bus_routes.append({
                'id': route_id,
                'element': route
            })
    return bus_routes


def extract_line_name(route_id):
    """Extract line name from route ID.

    Examples:
        bus_M60_SBS:1 -> M60_SBS:1
        bus_Bx19:0 -> Bx19:0
    """
    if route_id.startswith('bus_'):
        return route_id[4:]  # Remove 'bus_' prefix
    return route_id


def add_bus_flows(input_file, output_file=None, default_period=600, begin=0.0, end=86400.0):
    """
    Add bus flow definitions to ptflows.rou.xml.

    Args:
        input_file: Path to input ptflows.rou.xml
        output_file: Path to output file (default: overwrite input)
        default_period: Default period between buses in seconds (default: 600 = 10 minutes)
        begin: Simulation begin time
        end: Simulation end time
    """
    input_path = Path(input_file)
    if output_file is None:
        output_file = input_path
    output_path = Path(output_file)

    print(f"Reading {input_path}...")

    # Parse XML
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Get existing flows
    existing_flows = parse_existing_flows(root)
    print(f"Found {len(existing_flows)} existing flow definitions")

    # Get bus routes
    bus_routes = parse_bus_routes(root)
    print(f"Found {len(bus_routes)} bus route definitions")

    # Track routes that already have flows
    routes_with_flows = set()
    for flow_id in existing_flows:
        # Flow ID often matches route ID
        routes_with_flows.add(flow_id)

    # Add flow definitions for bus routes that don't have them
    added_count = 0
    for bus_route in bus_routes:
        route_id = bus_route['id']
        route_element = bus_route['element']

        if route_id in routes_with_flows:
            print(f"  Skipping {route_id}: flow already exists")
            continue

        # Create flow element
        line_name = extract_line_name(route_id)

        # Calculate begin offset to stagger bus departures
        # This helps avoid all buses on the same route starting at the same time
        begin_offset = added_count * 11.0  # 11 second offset per route
        flow_begin = begin + begin_offset
        flow_end = end + begin_offset

        flow_attrib = {
            'id': route_id,
            'type': 'bus',
            'route': route_id,
            'begin': f"{flow_begin:.1f}",
            'end': f"{flow_end:.1f}",
            'period': str(default_period),
            'line': line_name
        }

        # Create flow element with stops from route
        flow_elem = ET.Element('flow', flow_attrib)

        # Copy stop elements from route to flow
        for stop in route_element.findall('stop'):
            flow_elem.append(stop)

        # Insert flow after all route definitions (before closing tag)
        # Find position after all routes
        root.append(flow_elem)

        added_count += 1
        print(f"  Added flow for {route_id} (line={line_name}, period={default_period}s)")

    if added_count > 0:
        # Pretty print: add indentation
        indent_xml(root)

        # Write output
        tree.write(output_path, encoding='UTF-8', xml_declaration=True)
        print(f"\nWrote {output_path} with {added_count} new bus flow definitions")
    else:
        print("\nNo new bus flows added (all routes already have flows)")

    return added_count


def indent_xml(elem, level=0):
    """Add indentation to XML elements for pretty printing."""
    indent = "\n" + "    " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def main():
    parser = argparse.ArgumentParser(
        description='Add bus flow definitions to ptflows.rou.xml'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input ptflows.rou.xml file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to output file (default: overwrite input)'
    )
    parser.add_argument(
        '--period', '-p',
        type=int,
        default=600,
        help='Default period between buses in seconds (default: 600 = 10 minutes)'
    )
    parser.add_argument(
        '--begin', '-b',
        type=float,
        default=0.0,
        help='Simulation begin time (default: 0.0)'
    )
    parser.add_argument(
        '--end', '-e',
        type=float,
        default=86400.0,
        help='Simulation end time (default: 86400.0 = 24 hours)'
    )

    args = parser.parse_args()

    add_bus_flows(
        input_file=args.input_file,
        output_file=args.output,
        default_period=args.period,
        begin=args.begin,
        end=args.end
    )


if __name__ == '__main__':
    main()
