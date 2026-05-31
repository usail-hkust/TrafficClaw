"""
Code Sandbox for executing LLM-generated Python code in a safe, isolated environment.
Provides access to multi-modal transportation data and graph structures while maintaining security.
Supports: traffic signals, subway scheduling (TODO), bus scheduling (TODO).
"""

import os
import sys
import json
import traceback
import tempfile
import subprocess
import uuid
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
from utils.id_utils import generate_simulation_identifiers, generate_file_prefix as _generate_file_prefix


def execute_code(code: str,
                 context: Dict[str, Any],
                 timeout: int = 60,
                 verbose: bool = True) -> Dict[str, Any]:
    """
    Execute Python code in a sandboxed environment with access to multi-modal transportation data.
    
    Args:
        code: Python code to execute
        context: Context dictionary containing:
            - lane_graph: NetworkX graph (optional)
            - lane_inter_graph: NetworkX graph (optional)
            - intersection_graph: NetworkX graph (optional)
            - highway_graph: NetworkX graph for highway roads (optional)
            - highway_segment_graph: NetworkX DiGraph connecting highway segments (optional)
            - ramp_lane_graph: NetworkX DiGraph connecting ramps to controlled lanes and their 2-hop neighbors (optional)
            - lane_dict: Dictionary mapping lane_id to metadata (optional)
            - highway_segment_dict: Dictionary mapping road_id to highway road information (optional)
            - current_configs: Dictionary mapping module names to their configurations (optional)
                Format: {"signal_timing": {...}, "subway_scheduling": {...}, "bus_scheduling": {...}, ...}
            - subway_network: NetworkX graph for subway (optional, TODO)
            - cache_dir: Directory path for caching analysis results (optional, will be created if not exists)
        timeout: Execution timeout in seconds (default: 60)
        verbose: Whether to print execution details
        
    Returns:
        Dictionary containing:
            - success: Whether execution succeeded
            - output: Captured stdout/stderr
            - return_value: Returned value from code (if any)
            - error: Error message (if failed)
    """
    if verbose:
        print(f"Executing code in sandbox...")

    # Get workspace root
    current_file = Path(__file__).resolve()
    workspace_root = current_file.parent.parent

    # Generate simulation_id and file_prefix using unified function
    simulation_id, file_prefix = generate_simulation_identifiers(
        config_name=context.get("config_name"),
        llm_name=context.get("llm_name"),
        control_modules=context.get("control_modules"),
        simulation_id=context.get(
            "simulation_id")  # Use existing simulation_id if provided
    )

    # Generate unique sandbox_id using UUID
    # This ensures absolute uniqueness even when Ray reuses worker processes
    sandbox_id = str(uuid.uuid4())

    # Create sandbox identifier: simulation_id + sandbox_id
    sandbox_identifier = f"{simulation_id}_sandbox{sandbox_id}"

    # Get or create cache directory for this session
    cache_dir = context.get("cache_dir")
    if cache_dir is None:
        # Create cache directory in records/sandbox/cache/
        cache_base_dir = workspace_root / "records" / "sandbox" / "cache"
        cache_base_dir.mkdir(parents=True, exist_ok=True)

        # Use session_id from context if available, otherwise use sandbox_identifier
        session_id = context.get("session_id", sandbox_identifier)
        # Include prefix in cache directory name
        cache_dir = str(cache_base_dir / f"{file_prefix}_{session_id}")
        os.makedirs(cache_dir, exist_ok=True)
        context["cache_dir"] = cache_dir  # Store in context for reuse

    # Create temporary directory for this execution (for return value file)
    sandbox_base_dir = workspace_root / "records" / "sandbox"
    sandbox_base_dir.mkdir(parents=True, exist_ok=True)

    # Create a unique subdirectory for this sandbox session using sandbox_identifier
    temp_dir = str(sandbox_base_dir / f"{file_prefix}_{sandbox_identifier}")
    os.makedirs(temp_dir, exist_ok=True)

    # Store temp_dir in context for cleanup
    context["_temp_dir"] = temp_dir

    # Create temporary file for code execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        temp_file = f.name

        # Prepare sandbox environment
        sandbox_code = _prepare_sandbox_code(code,
                                             context,
                                             workspace_root,
                                             cache_dir,
                                             temp_dir,
                                             verbose=verbose)
        f.write(sandbox_code)

    try:
        # Execute code in subprocess for isolation
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8'
        )
        output = result.stdout
        error_output = result.stderr

        if result.returncode == 0:
            # Try to parse return value from file or output
            return_value = _extract_return_value(output, temp_dir)

            if verbose:
                print(f"Code executed successfully.")
                if output:
                    print(f"Output: {output[:500]}")  # Limit output length

            return {
                "success": True,
                "output": output,
                "return_value": return_value,
                "error": None
            }
        else:
            if verbose:
                print(
                    f"Code execution failed with return code {result.returncode}"
                )
                print(f"Error: {error_output}")

            return {
                "success": False,
                "output": output,
                "return_value": None,
                "error": error_output
            }

    except subprocess.TimeoutExpired:
        error_msg = f"Code execution timed out ({timeout}s limit)"
        if verbose:
            print(f"Error: {error_msg}")

        return {
            "success": False,
            "output": "",
            "return_value": None,
            "error": error_msg
        }

    except Exception as e:
        error_msg = f"Code execution error: {str(e)}"
        if verbose:
            print(f"Error: {error_msg}")
            traceback.print_exc()

        return {
            "success": False,
            "output": "",
            "return_value": None,
            "error": error_msg
        }

    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except:
            pass


def _prepare_sandbox_code(code: str,
                          context: Dict[str, Any],
                          workspace_root: Path,
                          cache_dir: str,
                          temp_dir: str,
                          verbose: bool = True) -> str:
    """
    Prepare code for sandbox execution by injecting context and utilities.
    
    Args:
        code: User's Python code
        context: Context dictionary
        workspace_root: Path to workspace root directory
        cache_dir: Cache directory path
        temp_dir: Temporary directory path for this execution
        
    Returns:
        Complete Python code ready for execution
    """
    # Serialize context data
    traffic_states_filepath = context.get("traffic_states_filepath", "")

    # Generate simulation_id and file_prefix using unified function
    simulation_id, file_prefix = generate_simulation_identifiers(
        config_name=context.get("config_name"),
        llm_name=context.get("llm_name"),
        control_modules=context.get("control_modules"),
        simulation_id=context.get(
            "simulation_id")  # Use existing simulation_id if provided
    )

    # Store file_prefix and simulation_id in sandbox code so traffic state reading functions can use them
    sandbox_file_prefix = file_prefix
    sandbox_simulation_id = simulation_id

    # Save graphs to temporary files
    graph_files = {}
    for graph_name in [
            "lane_graph", "lane_inter_graph", "intersection_graph",
            "highway_graph", "highway_segment_graph", "ramp_lane_graph",
            "zone_graph", "transit_graph"
    ]:
        if graph_name in context and context[graph_name] is not None:
            graph_file = os.path.join(temp_dir, f"{graph_name}.gml")
            try:
                _save_graph_to_gml(context[graph_name], graph_file)
                graph_files[graph_name] = graph_file
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to save {graph_name}: {e}")

    # Save lane_dict
    lane_dict_file = os.path.join(temp_dir, "lane_dict.json")
    if "lane_dict" in context:
        try:
            with open(lane_dict_file, 'w', encoding='utf-8') as f:
                json.dump(context["lane_dict"], f)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to save lane_dict: {e}")

    # Save highway_segment_dict
    highway_segment_dict_file = os.path.join(temp_dir,
                                             "highway_segment_dict.json")
    if "highway_segment_dict" in context:
        try:
            with open(highway_segment_dict_file, 'w', encoding='utf-8') as f:
                json.dump(context["highway_segment_dict"], f)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to save highway_segment_dict: {e}")

    # Save zone_dict
    zone_dict_file = os.path.join(temp_dir, "zone_dict.json")
    if "zone_dict" in context and context["zone_dict"]:
        try:
            with open(zone_dict_file, 'w', encoding='utf-8') as f:
                json.dump(context["zone_dict"], f)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to save zone_dict: {e}")

    # Save bus_route_info
    bus_route_info_file = os.path.join(temp_dir, "bus_route_info.json")
    if "bus_route_info" in context and context["bus_route_info"]:
        try:
            with open(bus_route_info_file, 'w', encoding='utf-8') as f:
                json.dump(context["bus_route_info"], f)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to save bus_route_info: {e}")

    # Save current_configs dictionary using unified naming: module_name_config.json
    configs_to_save = context.get("current_configs", {}) or {}
    
    # Save each module's config to its corresponding file
    known_modules = ['signal_timing', 'highway_speed_limit', 'ramp_metering', 'subway_scheduling', 'bus_scheduling', 'taxi_scheduling']
    config_file_paths = {}
    
    for module_name in known_modules:
        config_file = os.path.join(temp_dir, f"{module_name}_config.json")
        config_file_paths[module_name] = config_file
        
        # Get config data for this module
        config_data = configs_to_save.get(module_name)
        
        if config_data is not None:
            try:
                # ✅ 如果是 {module, config} 结构，只提取 config 部分
                if isinstance(config_data, dict) and 'config' in config_data:
                    config_to_save = config_data['config']
                else:
                    config_to_save = config_data
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_to_save, f)
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to save {module_name} config: {e}")
        else:
            # Create empty file if config is not provided
            try:
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to create empty {module_name}_config.json file: {e}")
    
    # Set file path variables for generated code
    signal_config_file = config_file_paths['signal_timing']
    highway_config_file = config_file_paths['highway_speed_limit']
    ramp_config_file = config_file_paths['ramp_metering']
    subway_schedule_file = config_file_paths['subway_scheduling']
    bus_schedule_file = config_file_paths['bus_scheduling']
    current_taxi_config_file = config_file_paths['taxi_scheduling']

    # Save taxi_fleet_state
    taxi_fleet_state_file = os.path.join(temp_dir, "taxi_fleet_state.json")
    if "taxi_fleet_state" in context and context["taxi_fleet_state"] is not None:
        try:
            with open(taxi_fleet_state_file, 'w', encoding='utf-8') as f:
                json.dump(context["taxi_fleet_state"], f, default=str)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to save taxi_fleet_state: {e}")

    # Save pending_reservations
    pending_reservations_file = os.path.join(temp_dir, "pending_reservations.json")
    if "pending_reservations" in context and context["pending_reservations"] is not None:
        try:
            with open(pending_reservations_file, 'w', encoding='utf-8') as f:
                json.dump(context["pending_reservations"], f, default=str)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to save pending_reservations: {e}")

    # Save taz_stats
    taz_stats_file = os.path.join(temp_dir, "taz_stats.json")
    if "taz_stats" in context and context["taz_stats"] is not None:
        try:
            with open(taz_stats_file, 'w', encoding='utf-8') as f:
                json.dump(context["taz_stats"], f, default=str)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to save taz_stats: {e}")

    # Convert paths to use forward slashes for cross-platform compatibility
    workspace_root_str = str(workspace_root).replace('\\', '/')
    cache_dir_str = cache_dir.replace('\\', '/')
    temp_dir_str = temp_dir.replace('\\', '/')

    # Convert all file paths to use forward slashes (critical for Windows compatibility in generated code)
    lane_dict_file = lane_dict_file.replace('\\', '/')
    highway_segment_dict_file = highway_segment_dict_file.replace('\\', '/')
    zone_dict_file = zone_dict_file.replace('\\', '/')
    bus_route_info_file = bus_route_info_file.replace('\\', '/')
    signal_config_file = signal_config_file.replace('\\', '/')
    highway_config_file = highway_config_file.replace('\\', '/')
    ramp_config_file = ramp_config_file.replace('\\', '/')
    taxi_fleet_state_file = taxi_fleet_state_file.replace('\\', '/')
    pending_reservations_file = pending_reservations_file.replace('\\', '/')
    taz_stats_file = taz_stats_file.replace('\\', '/')
    current_taxi_config_file = current_taxi_config_file.replace('\\', '/')
    subway_schedule_file = subway_schedule_file.replace('\\', '/')
    bus_schedule_file = bus_schedule_file.replace('\\', '/')
    # Prepare sandbox code
    sandbox_code = f"""# -*- coding: utf-8 -*-
import sys
import json
import os
import pickle
import time
import numpy as np
from pathlib import Path

# Add workspace to path
workspace_root = Path(r"{workspace_root_str}")
sys.path.insert(0, str(workspace_root))

# Import required modules
try:
    import networkx as nx
except ImportError:
    nx = None
    print("Warning: networkx not available")

from utils.traffic_state_collector import read_lane_traffic_states as _read_lane_traffic_states, read_highway_traffic_states as _read_highway_traffic_states, read_ramp_lane_traffic_states as _read_ramp_lane_traffic_states, _read_traffic_states_file as _read_traffic_states_file
from utils.traffic_prediction import predict_arima as _predict_arima

# Temp directory for this sandbox session
temp_dir = r"{temp_dir_str}"

# Cache directory for storing analysis results
_cache_dir = Path(r"{cache_dir_str}")
_cache_dir.mkdir(parents=True, exist_ok=True)
_cache_index_file = _cache_dir / "_cache_index.json"

# Initialize cache index if not exists
if not _cache_index_file.exists():
    _cache_index = {{}}
    with open(_cache_index_file, 'w', encoding='utf-8') as f:
        json.dump(_cache_index, f)
else:
    try:
        with open(_cache_index_file, 'r', encoding='utf-8') as f:
            _cache_index = json.load(f)
    except:
        _cache_index = {{}}

def _convert_for_pickle(obj):
    \"\"\"
    Recursively convert objects that can't be pickled (like defaultdict with lambda) to pickleable formats.
    \"\"\"
    from collections import defaultdict
    
    if isinstance(obj, defaultdict):
        # Convert defaultdict to regular dict
        result = dict(obj)
        # Recursively convert nested defaultdicts
        for k, v in result.items():
            result[k] = _convert_for_pickle(v)
        return result
    elif isinstance(obj, dict):
        # Recursively convert nested dicts that might contain defaultdicts
        result = {{}}
        for k, v in obj.items():
            result[k] = _convert_for_pickle(v)
        return result
    elif isinstance(obj, (list, tuple)):
        # Recursively convert lists/tuples that might contain defaultdicts
        if isinstance(obj, tuple):
            return tuple(_convert_for_pickle(item) for item in obj)
        else:
            return [_convert_for_pickle(item) for item in obj]
    elif isinstance(obj, set):
        # Convert sets that might contain unhashable defaultdicts
        # Note: sets can't contain dicts, but we handle it for completeness
        try:
            return {{_convert_for_pickle(item) if isinstance(item, (dict, defaultdict)) else item for item in obj}}
        except TypeError:
            # If set contains unhashable types, convert to list
            return [_convert_for_pickle(item) for item in obj]
    else:
        # For other types, return as-is
        return obj

def save_cache(cache_dict: dict):
    \"\"\"
    Save values to cache for later use in DATA_ANALYSIS or POLICY_PLANNING.
    
    Args:
        cache_dict: Dictionary where keys are cache names and values are dicts with:
            - "value": The value to cache (can be any Python object, including non-serializable ones)
            - "description": Optional description of what is cached
    
    Example:
        save_cache({{
            "bottleneck_intersections": {{
                "value": bottleneck_list,
                "description": "Top 20 bottleneck intersections"
            }},
            "traffic_data": {{
                "value": traffic_states,
                "description": "Historical traffic states for analysis"
            }}
        }})
    
    Note: Automatically converts defaultdict and lambda functions to pickleable formats.
    \"\"\"
    if not isinstance(cache_dict, dict):
        print(f"Warning: save_cache expects a dict, got {{type(cache_dict).__name__}}")
        return
    
    saved_keys = []
    for key, cache_info in cache_dict.items():
        if not isinstance(cache_info, dict):
            print(f"Warning: Cache entry '{{key}}' must be a dict with 'value' and optional 'description', skipping")
            continue
        
        value = cache_info.get("value")
        description = cache_info.get("description", "")
        
        # Convert value to pickleable format (e.g., defaultdict -> dict)
        try:
            value = _convert_for_pickle(value)
        except Exception as e:
            print(f"Warning: Failed to convert value for '{{key}}' to pickleable format: {{e}}")
            continue
        
        cache_file = _cache_dir / f"{{key}}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            _cache_index[key] = {{
                "description": description,
                "timestamp": time.time()
            }}
            saved_keys.append(key)
            if description:
                print(f"Cached '{{key}}': {{description}}")
            else:
                print(f"Cached '{{key}}'")
        except Exception as e:
            print(f"Warning: Failed to save cache '{{key}}': {{e}}")
    
    # Update cache index file after all saves
    if saved_keys:
        try:
            with open(_cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(_cache_index, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to update cache index: {{e}}")

def load_cache(key: str):
    \"\"\"
    Load a value from cache.
    
    Args:
        key: Cache key (string identifier)
    
    Returns:
        Cached value directly (the same object that was saved), or None if key doesn't exist
        Note: This returns the value directly, NOT a dict with "value" key.
        If you saved with save_cache({{"key": {{"value": data, "description": "desc"}}}}),
        load_cache("key") returns data directly, not {{"value": data, "description": "desc"}}
    
    Example:
        # Correct usage:
        bottleneck_list = load_cache("bottleneck_intersections")
        traffic_data = load_cache("traffic_data")
        
        # Wrong usage (will cause KeyError):
        # bottleneck_list = load_cache("bottleneck_intersections")["value"]  # DON'T DO THIS
    \"\"\"
    cache_file = _cache_dir / f"{{key}}.pkl"
    if not cache_file.exists():
        print(f"Warning: Cache key '{{key}}' not found")
        return None
    try:
        with open(cache_file, 'rb') as f:
            value = pickle.load(f)
        cache_info = _cache_index.get(key, {{}})
        description = cache_info.get("description", "")
        if description:
            print(f"Loaded cache '{{key}}': {{description}}")
        else:
            print(f"Loaded cache '{{key}}'")
        return value
    except Exception as e:
        print(f"Warning: Failed to load cache '{{key}}': {{e}}")
        return None

def list_cache() -> dict:
    \"\"\"
    List all cached keys and their descriptions.
    
    Returns:
        Dictionary mapping cache keys to their metadata (description, timestamp)
    
    Example:
        cache_list = list_cache()
        for key, info in cache_list.items():
            print(f"{{key}}: {{info['description']}}")
    \"\"\"
    return _cache_index.copy()

def clear_cache(key: str = None):
    \"\"\"
    Clear cache entry(ies).
    
    Args:
        key: Cache key to clear. If None, clears all cache.
    
    Example:
        clear_cache("bottleneck_intersections")  # Clear specific key
        clear_cache()  # Clear all cache
    \"\"\"
    if key is None:
        # Clear all cache
        for cache_key in list(_cache_index.keys()):
            cache_file = _cache_dir / f"{{cache_key}}.pkl"
            if cache_file.exists():
                try:
                    os.remove(cache_file)
                except:
                    pass
        _cache_index.clear()
        with open(_cache_index_file, 'w', encoding='utf-8') as f:
            json.dump(_cache_index, f)
        print("Cleared all cache")
    else:
        # Clear specific key
        if key in _cache_index:
            cache_file = _cache_dir / f"{{key}}.pkl"
            if cache_file.exists():
                try:
                    os.remove(cache_file)
                except:
                    pass
            del _cache_index[key]
            with open(_cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(_cache_index, f)
            print(f"Cleared cache '{{key}}'")
        else:
            print(f"Warning: Cache key '{{key}}' not found")

# Load graphs
lane_graph = None
lane_inter_graph = None
intersection_graph = None
highway_graph = None
highway_segment_graph = None
zone_graph = None
transit_graph = None
"""

    # Add graph loading code
    if "lane_graph" in graph_files:
        lane_graph_path = graph_files["lane_graph"].replace('\\', '/')
        sandbox_code += f"""
try:
    lane_graph = nx.read_gml(r"{lane_graph_path}")
except Exception as e:
    print(f"Warning: Failed to load lane_graph: {{e}}")
"""

    if "lane_inter_graph" in graph_files:
        lane_inter_graph_path = graph_files["lane_inter_graph"].replace('\\', '/')
        sandbox_code += f"""
try:
    lane_inter_graph = nx.read_gml(r"{lane_inter_graph_path}")
except Exception as e:
    print(f"Warning: Failed to load lane_inter_graph: {{e}}")
"""

    if "intersection_graph" in graph_files:
        intersection_graph_path = graph_files["intersection_graph"].replace('\\', '/')
        sandbox_code += f"""
try:
    intersection_graph = nx.read_gml(r"{intersection_graph_path}")
except Exception as e:
    print(f"Warning: Failed to load intersection_graph: {{e}}")
"""

    if "highway_graph" in graph_files:
        highway_graph_path = graph_files["highway_graph"].replace('\\', '/')
        sandbox_code += f"""
try:
    highway_graph = nx.read_gml(r"{highway_graph_path}")
except Exception as e:
    print(f"Warning: Failed to load highway_graph: {{e}}")
"""

    if "highway_segment_graph" in graph_files:
        sandbox_code += f"""
try:
    highway_segment_graph = nx.read_gml("{graph_files["highway_segment_graph"]}")
except Exception as e:
    print(f"Warning: Failed to load highway_segment_graph: {{e}}")
    highway_segment_graph = None
"""

    if "ramp_lane_graph" in graph_files:
        sandbox_code += f"""
try:
    ramp_lane_graph = nx.read_gml("{graph_files["ramp_lane_graph"]}")
except Exception as e:
    print(f"Warning: Failed to load ramp_lane_graph: {{e}}")
    ramp_lane_graph = None
"""
    else:
        sandbox_code += """
ramp_lane_graph = None
"""

    # Add zone_graph loading
    if "zone_graph" in graph_files:
        sandbox_code += f"""
try:
    zone_graph = nx.read_gml("{graph_files["zone_graph"]}")
except Exception as e:
    print(f"Warning: Failed to load zone_graph: {{e}}")
    zone_graph = None
"""

    # Add transit_graph loading
    if "transit_graph" in graph_files:
        sandbox_code += f"""
try:
    transit_graph = nx.read_gml("{graph_files["transit_graph"]}")
except Exception as e:
    print(f"Warning: Failed to load transit_graph: {{e}}")
    transit_graph = None
"""

    # Add lane_dict loading
    sandbox_code += f"""
# Load lane_dict
lane_dict = {{}}
try:
    if os.path.exists("{lane_dict_file}"):
        with open("{lane_dict_file}", 'r', encoding='utf-8') as f:
            lane_dict = json.load(f)
except Exception as e:
    print(f"Warning: Failed to load lane_dict: {{e}}")
    lane_dict = {{}}

# Load highway_segment_dict
highway_segment_dict = {{}}
try:
    if os.path.exists("{highway_segment_dict_file}"):
        with open("{highway_segment_dict_file}", 'r', encoding='utf-8') as f:
            highway_segment_dict = json.load(f)
except Exception as e:
    print(f"Warning: Failed to load highway_segment_dict: {{e}}")
    highway_segment_dict = {{}}

# Load zone_dict
zone_dict = {{}}
try:
    if os.path.exists("{zone_dict_file}"):
        with open("{zone_dict_file}", 'r', encoding='utf-8') as f:
            zone_dict = json.load(f)
except Exception as e:
    print(f"Warning: Failed to load zone_dict: {{e}}")
    zone_dict = {{}}

# Load bus_route_info
bus_route_info = {{}}
try:
    if os.path.exists("{bus_route_info_file}"):
        with open("{bus_route_info_file}", 'r', encoding='utf-8') as f:
            bus_route_info = json.load(f)
except Exception as e:
    print(f"Warning: Failed to load bus_route_info: {{e}}")
    bus_route_info = {{}}

# Load current_configs dictionary (new unified format)
# Use unified naming: module_name_config.json
current_configs = {{}}
known_modules = ['signal_timing', 'highway_speed_limit', 'ramp_metering', 'subway_scheduling', 'bus_scheduling', 'taxi_scheduling']

# Load each module's config and add to current_configs dictionary
for module_name in known_modules:
    config_file = os.path.join(r"{temp_dir}", f"{{module_name}}_config.json")
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                if isinstance(config_data, dict):
                    current_configs[module_name] = config_data
                else:
                    current_configs[module_name] = {{}}
        else:
            current_configs[module_name] = {{}}
    except Exception as e:
        print(f"Warning: Failed to load {{module_name}} config: {{e}}")
        current_configs[module_name] = {{}}

# Create individual variables from current_configs for use in sandbox code
current_signal_config = current_configs.get('signal_timing', {{}})
current_highway_speed_limit_config = current_configs.get('highway_speed_limit', {{}})
current_ramp_metering_config = current_configs.get('ramp_metering', {{}})
current_subway_schedule = current_configs.get('subway_scheduling', {{}})
current_bus_schedule = current_configs.get('bus_scheduling', {{}})
current_taxi_config = current_configs.get('taxi_scheduling', {{}})

# Load taxi_fleet_state
taxi_fleet_state = {{}}
try:
    if os.path.exists("{taxi_fleet_state_file}"):
        with open("{taxi_fleet_state_file}", 'r', encoding='utf-8') as f:
            taxi_fleet_state = json.load(f)
        if not isinstance(taxi_fleet_state, dict):
            taxi_fleet_state = {{}}
except Exception as e:
    print(f"Warning: Failed to load taxi_fleet_state: {{e}}")
    taxi_fleet_state = {{}}

# Load pending_reservations
pending_reservations = {{}}
try:
    if os.path.exists("{pending_reservations_file}"):
        with open("{pending_reservations_file}", 'r', encoding='utf-8') as f:
            pending_reservations = json.load(f)
        if not isinstance(pending_reservations, dict):
            pending_reservations = {{}}
except Exception as e:
    print(f"Warning: Failed to load pending_reservations: {{e}}")
    pending_reservations = {{}}

# Load taz_stats
taz_stats = {{}}
try:
    if os.path.exists("{taz_stats_file}"):
        with open("{taz_stats_file}", 'r', encoding='utf-8') as f:
            taz_stats = json.load(f)
        if not isinstance(taz_stats, dict):
            taz_stats = {{}}
except Exception as e:
    print(f"Warning: Failed to load taz_stats: {{e}}")
    taz_stats = {{}}

def dispatch_taxi(taxi_id, reservation_ids):
    \"\"\"Build a dispatch decision entry for taxi scheduling config.\"\"\"
    if reservation_ids is None:
        reservation_ids = []
    return {{
        "taxi_id": str(taxi_id),
        "reservation_ids": [str(res_id) for res_id in reservation_ids]
    }}

def reposition_taxi(taxi_id, target_edge=None, target_taz=None):
    \"\"\"Build a reposition decision entry for taxi scheduling config.\"\"\"
    entry = {{"taxi_id": str(taxi_id)}}
    if target_edge is not None:
        entry["target_edge"] = str(target_edge)
    if target_taz is not None:
        entry["target_taz"] = str(target_taz)
    return entry

def _find_reservation_by_id(reservation_id):
    if reservation_id is None:
        return None
    for res in pending_reservations.get("reservations", []):
        if str(res.get("id")) == str(reservation_id):
            return res
    return None

def rank_idle_taxis_by_distance(reservation_id=None, reservation=None, max_candidates=10):
    \"\"\"Rank idle taxis by Euclidean distance to a reservation pickup_position.\"\"\"
    res = reservation or _find_reservation_by_id(reservation_id)
    if not isinstance(res, dict):
        return []
    pickup_position = res.get("pickup_position")
    if not pickup_position or len(pickup_position) < 2:
        return []

    idle_taxis = taxi_fleet_state.get("idle_taxis", [])
    taxi_details = taxi_fleet_state.get("taxi_details", {{}})
    candidates = []
    for taxi_id in idle_taxis:
        details = taxi_details.get(taxi_id, {{}})
        pos = details.get("position")
        if not pos or len(pos) < 2:
            continue
        distance = math.hypot(pos[0] - pickup_position[0], pos[1] - pickup_position[1])
        candidates.append({{
            "taxi_id": taxi_id,
            "distance": distance,
            "current_edge": details.get("current_edge"),
            "position": pos
        }})

    candidates.sort(key=lambda item: item.get("distance", float("inf")))
    if max_candidates is None:
        return candidates
    return candidates[:max_candidates]

# TODO: Load subway network and schedule
subway_network = None

# Default file_prefix and simulation_id from context (automatically used if not specified)
# These are set from the simulation context to ensure only current simulation data is read
_default_file_prefix = {repr(file_prefix) if file_prefix else "None"}
_default_simulation_id = {repr(sandbox_simulation_id) if sandbox_simulation_id else "None"}

# Define read_lane_traffic_states() function (for lane/intersection traffic data)
def read_lane_traffic_states(start_time=None, end_time=None, exact_time=None, max_snapshots=None):
    \"\"\"Query historical lane traffic conditions from the current simulation's traffic states file.
    
    Automatically uses the current simulation's file_prefix and simulation_id from context.
    This ensures that only traffic states from the current simulation are read.
    LLM agent does not need to pass file_prefix or simulation_id - they are automatically set.
    
    Args:
        start_time: Minimum simulation time to include (None for no lower bound)
        end_time: Maximum simulation time to include (None for no upper bound)
        exact_time: Exact simulation time to retrieve (takes precedence over range)
        max_snapshots: Maximum number of snapshots to read (None for all)
    \"\"\"
    # Automatically use default file_prefix and simulation_id from context
    file_prefix = _default_file_prefix if _default_file_prefix != "None" else None
    simulation_id = _default_simulation_id if _default_simulation_id != "None" else None
    return _read_lane_traffic_states(
        max_snapshots=max_snapshots,
        start_time=start_time,
        end_time=end_time,
        exact_time=exact_time,
        file_prefix=file_prefix,
        simulation_id=simulation_id
    )

# Define read_highway_traffic_states() function (for highway traffic data)
def read_highway_traffic_states(start_time=None, end_time=None, exact_time=None, max_snapshots=None):
    \"\"\"Query historical highway traffic conditions from the current simulation's traffic states file.
    
    Automatically uses the current simulation's file_prefix and simulation_id from context.
    This ensures that only traffic states from the current simulation are read.
    LLM agent does not need to pass file_prefix or simulation_id - they are automatically set.
    
    Args:
        start_time: Minimum simulation time to include (None for no lower bound)
        end_time: Maximum simulation time to include (None for no upper bound)
        exact_time: Exact simulation time to retrieve (takes precedence over range)
        max_snapshots: Maximum number of snapshots to read (None for all)
    \"\"\"
    # Automatically use default file_prefix and simulation_id from context
    file_prefix = _default_file_prefix if _default_file_prefix != "None" else None
    simulation_id = _default_simulation_id if _default_simulation_id != "None" else None
    return _read_highway_traffic_states(
        max_snapshots=max_snapshots,
        start_time=start_time,
        end_time=end_time,
        exact_time=exact_time,
        file_prefix=file_prefix,
        simulation_id=simulation_id
    )

# Define read_ramp_lane_traffic_states() function (for ramp lane traffic data)
def read_ramp_lane_traffic_states(start_time=None, end_time=None, exact_time=None, max_snapshots=None):
    \"\"\"Query historical ramp lane traffic conditions from the current simulation's traffic states file.
    
    Reads lane-level data from ramp_lane_graph (controlled_lane, upstream_lane, downstream_lane nodes).
    Automatically uses the current simulation's file_prefix and simulation_id from context.
    This ensures that only traffic states from the current simulation are read.
    LLM agent does not need to pass file_prefix or simulation_id - they are automatically set.
    
    Args:
        start_time: Minimum simulation time to include (None for no lower bound)
        end_time: Maximum simulation time to include (None for no upper bound)
        exact_time: Exact simulation time to retrieve (takes precedence over range)
        max_snapshots: Maximum number of snapshots to read (None for all)
    \"\"\"
    # Automatically use default file_prefix and simulation_id from context
    file_prefix = _default_file_prefix if _default_file_prefix != "None" else None
    simulation_id = _default_simulation_id if _default_simulation_id != "None" else None
    return _read_ramp_lane_traffic_states(
        max_snapshots=max_snapshots,
        start_time=start_time,
        end_time=end_time,
        exact_time=exact_time,
        file_prefix=file_prefix,
        simulation_id=simulation_id
    )

# Define read_taxi_traffic_states() function (for taxi fleet data)
def read_taxi_traffic_states(start_time=None, end_time=None, exact_time=None, max_snapshots=None):
    \"\"\"Query historical taxi fleet states from the current simulation's traffic states file.
    
    Returns empty snapshots if taxi states are not collected in the traffic states file.
    \"\"\"
    # Automatically use default file_prefix and simulation_id from context
    file_prefix = _default_file_prefix if _default_file_prefix != "None" else None
    simulation_id = _default_simulation_id if _default_simulation_id != "None" else None
    raw_result = _read_traffic_states_file(
        max_snapshots=max_snapshots,
        start_time=start_time,
        end_time=end_time,
        exact_time=exact_time,
        file_prefix=file_prefix,
        simulation_id=simulation_id
    )

    if "error" in raw_result:
        return raw_result

    result = {{
        "metadata": raw_result.get("metadata"),
        "snapshots": [],
        "total_snapshots": raw_result.get("total_snapshots", 0),
        "filtered_snapshots": 0
    }}

    for snapshot in raw_result.get("snapshots", []):
        traffic_states = snapshot.get("traffic_states", {{}})
        taxi_states = traffic_states.get("taxi_states")
        if taxi_states is None:
            continue
        if isinstance(taxi_states, dict):
            taxi_snapshot = dict(taxi_states)
        else:
            taxi_snapshot = {{"taxi_states": taxi_states}}
        taxi_snapshot["simulation_time"] = snapshot.get("simulation_time")
        result["snapshots"].append(taxi_snapshot)

    result["filtered_snapshots"] = len(result["snapshots"])
    return result

# Helper function to extract current speed limit from schedule list
def get_current_speed_limit(seg_id, config=None, default=65):
    \"\"\"
    Extract current speed limit (at time=0) from highway speed limit configuration.
    
    Args:
        seg_id: Highway segment ID (str)
        config: Speed limit configuration dict (default: current_highway_speed_limit_config)
        default: Default speed limit to return if not found (default: 65 mph)
    
    Returns:
        Current speed limit in mph (int)
    
    Example:
        current_limit = get_current_speed_limit('highway_segment_0')
        # Returns the speed_limit from the first entry (time=0) in the schedule list
    \"\"\"
    if config is None:
        config = current_highway_speed_limit_config
    
    schedule = config.get(seg_id)
    if schedule and isinstance(schedule, list) and len(schedule) > 0:
        first_entry = schedule[0]
        if isinstance(first_entry, dict) and 'speed_limit' in first_entry:
            return first_entry['speed_limit']
    return default

# Define read_subway_states function
def read_subway_states(start_time=None, end_time=None, exact_time=None, line_id=None, max_snapshots=None):
    \"\"\"Query historical subway operation data from the current simulation's traffic states file.

    Automatically uses the current simulation's file_prefix and simulation_id from context.
    This ensures that only traffic states from the current simulation are read.
    LLM agent does not need to pass file_prefix or simulation_id - they are automatically set.

    Args:
        start_time: Minimum simulation time to include (None for no lower bound)
        end_time: Maximum simulation time to include (None for no upper bound)
        exact_time: Exact simulation time to retrieve (takes precedence over range)
        line_id: Optional line ID to filter results (e.g., \"subway_line_1\")
        max_snapshots: Maximum number of snapshots to read (None for all)
    \"\"\"
    # Automatically use default file_prefix and simulation_id from context
    file_prefix = _default_file_prefix if _default_file_prefix != \"None\" else None
    simulation_id = _default_simulation_id if _default_simulation_id != \"None\" else None
    from utils.traffic_state_collector import read_subway_traffic_states
    return read_subway_traffic_states(
        start_time=start_time,
        end_time=end_time,
        exact_time=exact_time,
        line_id=line_id,
        max_snapshots=max_snapshots,
        file_prefix=file_prefix,
        simulation_id=simulation_id
    )

# Define read_bus_states function
def read_bus_states(start_time=None, end_time=None, exact_time=None, line_id=None, max_snapshots=None):
    \"\"\"Query historical bus operation data from the current simulation's traffic states file.

    Automatically uses the current simulation's file_prefix and simulation_id from context.
    This ensures that only traffic states from the current simulation are read.
    LLM agent does not need to pass file_prefix or simulation_id - they are automatically set.

    Args:
        start_time: Minimum simulation time to include (None for no lower bound)
        end_time: Maximum simulation time to include (None for no upper bound)
        exact_time: Exact simulation time to retrieve (takes precedence over range)
        line_id: Optional line ID to filter results (e.g., \"route_bus1_eastbound\")
        max_snapshots: Maximum number of snapshots to read (None for all)
    \"\"\"
    # Automatically use default file_prefix and simulation_id from context
    file_prefix = _default_file_prefix if _default_file_prefix != \"None\" else None
    simulation_id = _default_simulation_id if _default_simulation_id != \"None\" else None
    from utils.traffic_state_collector import read_bus_traffic_states
    return read_bus_traffic_states(
        start_time=start_time,
        end_time=end_time,
        exact_time=exact_time,
        line_id=line_id,
        max_snapshots=max_snapshots,
        file_prefix=file_prefix,
        simulation_id=simulation_id
    )

# Define predict_arima function (generic ARIMA time series prediction)
# This is a wrapper that calls the generic prediction function from utils.traffic_prediction
def predict_arima(
    time_series,
    history_window,
    prediction_window,
    forecast_interval=1
):
    \"\"\"
    Generic ARIMA time series prediction function.
    
    Args:
        time_series: List of time series values (float)
        history_window: Number of time steps to use as history (int)
        prediction_window: Number of time steps to predict (int)
        forecast_interval: Interval between prediction points (int, default: 1)
                         If prediction_window=10 and forecast_interval=2, predicts 5 points
    
    Returns:
        Dictionary containing:
            - predicted_values: List of predicted values
            - confidence_lower: List of lower confidence bounds (95% CI)
            - confidence_upper: List of upper confidence bounds (95% CI)
            - model_info: Dictionary with model parameters and statistics
            - error: Error message if prediction failed (None if successful)
    \"\"\"
    return _predict_arima(
        time_series=time_series,
        history_window=history_window,
        prediction_window=prediction_window,
        forecast_interval=forecast_interval
    )

# User code starts here
# ============================================
{code}
# ============================================

# Helper function to clean NaN/Infinity values for JSON serialization
def _clean_for_json(obj):
    \"\"\"Recursively clean NaN/Infinity values from dict/list for JSON serialization.\"\"\"
    import math
    if isinstance(obj, dict):
        return {{k: _clean_for_json(v) for k, v in obj.items()}}
    elif isinstance(obj, list):
        return [_clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None  # Replace NaN/Infinity with None
        return obj
    elif hasattr(obj, 'item'):  # numpy scalar
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    elif hasattr(obj, 'tolist'):  # numpy array
        return _clean_for_json(obj.tolist())
    else:
        return obj

# If code defines a result variable, save it to a file instead of printing
# This avoids long stdout output and provides a cleaner way to pass data between processes
# Support both legacy format (signal_config) and new format (config dict with module names)
_result_file = Path("{temp_dir}") / "_return_value.json"
if 'config' in locals():
    try:
        with open(_result_file, 'w') as _f:
            json.dump(_clean_for_json(config), _f, indent=2)
        print("__RETURN_VALUE_SAVED__")
    except Exception as _e:
        print(f"Warning: Failed to save return value: {{_e}}")
elif 'result' in locals():
    try:
        with open(_result_file, 'w') as _f:
            json.dump(_clean_for_json(result), _f, indent=2)
        print("__RETURN_VALUE_SAVED__")
    except Exception as _e:
        print(f"Warning: Failed to save return value: {{_e}}")
elif 'final_configuration' in locals():
    try:
        with open(_result_file, 'w') as _f:
            json.dump(_clean_for_json(final_configuration), _f, indent=2)
        print("__RETURN_VALUE_SAVED__")
    except Exception as _e:
        print(f"Warning: Failed to save return value: {{_e}}")
elif 'final_config' in locals():
    try:
        with open(_result_file, 'w') as _f:
            json.dump(_clean_for_json(final_config), _f, indent=2)
        print("__RETURN_VALUE_SAVED__")
    except Exception as _e:
        print(f"Warning: Failed to save return value: {{_e}}")
elif 'control_configs' in locals():
    try:
        with open(_result_file, 'w') as _f:
            json.dump(_clean_for_json(control_configs), _f, indent=2)
        print("__RETURN_VALUE_SAVED__")
    except Exception as _e:
        print(f"Warning: Failed to save return value: {{_e}}")
"""

    # Normalize indentation from literal tabs to avoid unexpected indent errors.
    sandbox_code = "\n".join(
        line.lstrip("\t") for line in sandbox_code.splitlines())
    return sandbox_code


def _save_graph_to_gml(graph, filepath: str):
    """
    Save NetworkX graph to GML file format.
    
    Args:
        graph: NetworkX graph object
        filepath: Path to save the graph
    """
    try:
        import networkx as nx
        # Use GML format for better compatibility
        nx.write_gml(graph, filepath)
    except Exception as e:
        raise Exception(f"Failed to save graph to {filepath}: {e}")


def _extract_return_value(output: str, temp_dir: str) -> Any:
    """
    Extract return value from file or code output.
    First tries to read from _return_value.json file (preferred method).
    Falls back to parsing output for __RETURN_VALUE__ marker (legacy method).
    
    Args:
        output: Code execution output string
        temp_dir: Temporary directory where return value file is stored
        
    Returns:
        Parsed return value (dict, list, etc.) or None if not found
    """
    # Method 1: Try to read from file (preferred - avoids long stdout)
    return_value_file = Path(temp_dir) / "_return_value.json"
    if return_value_file.exists():
        try:
            with open(return_value_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read return value from file: {e}")

    # Method 2: Fall back to parsing output (legacy method)
    lines = output.split('\n')

    # Look for __RETURN_VALUE_SAVED__ marker (new method)
    for line in lines:
        if line.strip() == "__RETURN_VALUE_SAVED__":
            # Already handled by file reading above
            return None

    # Look for __RETURN_VALUE__ marker (legacy method)
    for i, line in enumerate(lines):
        if line.strip() == "__RETURN_VALUE__":
            if i + 1 < len(lines):
                try:
                    return json.loads(lines[i + 1])
                except json.JSONDecodeError:
                    # If JSON parsing fails, return as string
                    return lines[i + 1]

    return None
