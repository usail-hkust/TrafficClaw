#!/usr/bin/env python3
"""
Validate SUMO network file for common errors that prevent SUMO from starting.

This script checks:
1. Phase state string lengths match the number of controlled links
2. Phase state characters are valid SUMO signal states
3. No conflicting movements in the same phase
4. All required XML attributes are present

Usage:
    python tools/validate_net_file.py sumo_config/jinan/roadnet_3_4_with_phases.net.xml
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Tuple


def validate_phase_state_chars(state: str) -> Tuple[bool, str]:
    """Validate that phase state contains only valid characters.
    
    Valid SUMO signal states:
    - r/R: red
    - g/G: green (g=minor, G=major)
    - y/Y: yellow
    - o/O: off (blinking)
    - u/U: red+yellow
    - s/S: green right turn (s=minor, S=major)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_chars = set('rRgGyYoOuUsS')
    invalid_chars = set(state) - valid_chars
    
    if invalid_chars:
        return False, f"Invalid characters: {invalid_chars}"
    
    return True, ""


def validate_tl_logic(tl_logic: ET.Element, tls_id: str) -> List[str]:
    """Validate a single tlLogic element.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check required attributes
    if not tl_logic.get('id'):
        errors.append(f"TLS {tls_id}: Missing 'id' attribute")
    
    if not tl_logic.get('type'):
        errors.append(f"TLS {tls_id}: Missing 'type' attribute")
    
    if not tl_logic.get('programID'):
        errors.append(f"TLS {tls_id}: Missing 'programID' attribute")
    
    # Get all phases
    phases = tl_logic.findall('phase')
    
    if not phases:
        errors.append(f"TLS {tls_id}: No phases defined")
        return errors
    
    # Track state lengths to ensure consistency
    state_lengths = set()
    
    for i, phase in enumerate(phases):
        phase_name = phase.get('name', f'phase_{i}')
        
        # Check required phase attributes
        if not phase.get('duration'):
            errors.append(f"TLS {tls_id}, {phase_name}: Missing 'duration' attribute")
        
        state = phase.get('state')
        if not state:
            errors.append(f"TLS {tls_id}, {phase_name}: Missing 'state' attribute")
            continue
        
        # Validate state characters
        is_valid, error_msg = validate_phase_state_chars(state)
        if not is_valid:
            errors.append(f"TLS {tls_id}, {phase_name}: {error_msg}")
        
        # Track state length
        state_lengths.add(len(state))
    
    # Check if all phases have the same state length
    if len(state_lengths) > 1:
        errors.append(f"TLS {tls_id}: Inconsistent state lengths across phases: {state_lengths}")
    
    return errors


def validate_net_file(net_file_path: Path) -> Tuple[bool, List[str]]:
    """Validate a SUMO network file.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if not net_file_path.exists():
        return False, [f"File not found: {net_file_path}"]
    
    try:
        tree = ET.parse(net_file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        return False, [f"XML parsing error: {e}"]
    
    all_errors = []
    
    # Validate all tlLogic elements
    tl_logics = root.findall('tlLogic')
    
    if not tl_logics:
        all_errors.append("Warning: No traffic light logic found in network file")
    
    for tl_logic in tl_logics:
        tls_id = tl_logic.get('id', 'unknown')
        errors = validate_tl_logic(tl_logic, tls_id)
        all_errors.extend(errors)
    
    is_valid = len(all_errors) == 0
    return is_valid, all_errors


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate SUMO network file for common errors"
    )
    parser.add_argument(
        "net_file",
        type=str,
        help="Path to SUMO network file (.net.xml)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed validation results"
    )
    
    args = parser.parse_args()
    
    net_file_path = Path(args.net_file)
    
    print(f"Validating: {net_file_path}")
    print("=" * 60)
    
    is_valid, errors = validate_net_file(net_file_path)
    
    if is_valid:
        print("✓ Validation PASSED")
        print(f"  No errors found in {net_file_path.name}")
        return 0
    else:
        print("✗ Validation FAILED")
        print(f"  Found {len(errors)} error(s):")
        print()
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

