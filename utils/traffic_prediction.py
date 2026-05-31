"""
Traffic flow prediction utilities using ARIMA time series models.
Provides generic time series prediction functionality.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


def predict_arima(
    time_series: List[float],
    history_window: int,
    prediction_window: int,
    forecast_interval: int = 1
) -> Dict[str, Any]:
    """
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
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        import warnings
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
    except ImportError as e:
        return {
            "error": f"Required library not available: {e}. Please install statsmodels: pip install statsmodels",
            "predicted_values": [],
            "confidence_lower": [],
            "confidence_upper": [],
            "model_info": {}
        }
    
    # Validate inputs
    if not isinstance(time_series, (list, np.ndarray)):
        return {
            "error": "time_series must be a list or numpy array",
            "predicted_values": [],
            "confidence_lower": [],
            "confidence_upper": [],
            "model_info": {}
        }
    
    if len(time_series) < 10:
        return {
            "error": f"Insufficient data: {len(time_series)} points (need at least 10)",
            "predicted_values": [],
            "confidence_lower": [],
            "confidence_upper": [],
            "model_info": {}
        }
    
    # Convert to numpy array
    values = np.array(time_series)
    
    # Use last history_window points if history_window is specified and valid
    if history_window > 0 and history_window < len(values):
        values = values[-history_window:]
    
    if len(values) < 10:
        return {
            "error": f"Insufficient data in history window: {len(values)} points (need at least 10)",
            "predicted_values": [],
            "confidence_lower": [],
            "confidence_upper": [],
            "model_info": {}
        }
    
    # Calculate statistics
    mean_value = float(np.mean(values))
    std_value = float(np.std(values))
    
    # Fit ARIMA model - try different orders and select best based on AIC
    best_aic = float('inf')
    best_model = None
    best_order = None
    
    # Try common ARIMA orders
    orders_to_try = [
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
        (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1),
        (0, 1, 1), (0, 1, 2)
    ]
    
    for order in orders_to_try:
        try:
            model = ARIMA(values, order=order)
            fitted_model = model.fit()
            if fitted_model.aic < best_aic:
                best_aic = fitted_model.aic
                best_model = fitted_model
                best_order = order
        except:
            continue
    
    if best_model is None:
        # Fallback to simple model
        try:
            model = ARIMA(values, order=(1, 0, 0))
            best_model = model.fit()
            best_order = (1, 0, 0)
            best_aic = best_model.aic
        except Exception as e:
            return {
                "error": f"Failed to fit ARIMA model: {e}",
                "predicted_values": [],
                "confidence_lower": [],
                "confidence_upper": [],
                "model_info": {}
            }
    
    # Calculate number of forecast steps
    num_steps = int(prediction_window / forecast_interval)
    if num_steps <= 0:
        num_steps = 1
    
    # Generate forecast
    try:
        forecast_result = best_model.forecast(steps=num_steps)
        forecast_values = forecast_result
        
        # Get confidence intervals (approximate using standard error)
        # For simplicity, use ±1.96 * std of residuals for 95% CI
        residuals_std = np.std(best_model.resid)
        confidence_lower = forecast_values - 1.96 * residuals_std
        confidence_upper = forecast_values + 1.96 * residuals_std
        
        # Convert to lists
        predicted_values = forecast_values.tolist()
        confidence_lower = confidence_lower.tolist()
        confidence_upper = confidence_upper.tolist()
        
    except Exception as e:
        return {
            "error": f"Failed to generate forecast: {e}",
            "predicted_values": [],
            "confidence_lower": [],
            "confidence_upper": [],
            "model_info": {}
        }
    
    return {
        "error": None,
        "predicted_values": predicted_values,
        "confidence_lower": confidence_lower,
        "confidence_upper": confidence_upper,
        "model_info": {
            "model_type": "ARIMA",
            "order": best_order,
            "aic": float(best_aic),
            "data_points": len(values),
            "mean": mean_value,
            "std": std_value
        }
    }


def predict_traffic_flow_arima(
    highway_segment_id: str,
    metric: str = 'occupancy',
    forecast_horizon: int = 1800,
    forecast_interval: int = 300,
    historical_window: int = 3600,
    traffic_states_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Predict future traffic flow patterns for a highway segment using ARIMA.
    This is a convenience wrapper around predict_arima() that handles traffic data extraction.
    
    Args:
        highway_segment_id: Highway segment ID to predict (str)
        metric: Traffic metric to predict: 'speed', 'occupancy', 'density', or 'throughput' (str)
        forecast_horizon: Number of seconds to forecast into the future (int)
        forecast_interval: Time interval between forecast points in seconds (int)
        historical_window: Number of seconds of historical data to use for training (int)
        traffic_states_data: Optional pre-loaded traffic states data. If None, will read from file.
    
    Returns:
        Dictionary with forecast results and analysis
    """
    # Import here to avoid circular imports
    from utils.traffic_state_collector import read_highway_traffic_states as _read_highway_traffic_states
    
    # Read historical traffic states if not provided
    if traffic_states_data is None:
        try:
            data = _read_highway_traffic_states()
        except Exception as e:
            return {
                "error": f"Failed to read traffic states: {e}",
                "highway_segment_id": highway_segment_id,
                "metric": metric
            }
    else:
        data = traffic_states_data
    
    if not data or 'snapshots' not in data or len(data['snapshots']) == 0:
        return {
            "error": "No historical traffic data available",
            "highway_segment_id": highway_segment_id,
            "metric": metric
        }
    
    # Extract time series data for the specified segment and metric
    times = []
    values = []
    
    for snapshot in data['snapshots']:
        sim_time = snapshot.get('simulation_time', 0)
        highway_states = snapshot.get('highway_states', {})
        
        if highway_segment_id not in highway_states:
            continue
        
        segment_data = highway_states[highway_segment_id]
        
        # Extract the requested metric
        if metric == 'speed':
            value = segment_data.get('segment_speed', 0.0)
        elif metric == 'occupancy':
            value = segment_data.get('segment_occupancy', 0.0)
        elif metric == 'density':
            value = segment_data.get('segment_density', 0.0)
        elif metric == 'throughput':
            # Calculate throughput as speed * density
            speed = segment_data.get('segment_speed', 0.0)
            density = segment_data.get('segment_density', 0.0)
            value = speed * density
        else:
            return {
                "error": f"Unknown metric: {metric}. Use 'speed', 'occupancy', 'density', or 'throughput'",
                "highway_segment_id": highway_segment_id,
                "metric": metric
            }
        
        times.append(sim_time)
        values.append(value)
    
    if len(values) < 10:
        return {
            "error": f"Insufficient historical data: {len(values)} points (need at least 10)",
            "highway_segment_id": highway_segment_id,
            "metric": metric,
            "available_data_points": len(values)
        }
    
    # Filter to historical window
    if len(times) > 0:
        latest_time = times[-1]
        start_time = latest_time - historical_window
        
        filtered_times = []
        filtered_values = []
        for t, v in zip(times, values):
            if t >= start_time:
                filtered_times.append(t)
                filtered_values.append(v)
        
        times = filtered_times
        values = filtered_values
    
    if len(values) < 10:
        return {
            "error": f"Insufficient data in historical window: {len(values)} points (need at least 10)",
            "highway_segment_id": highway_segment_id,
            "metric": metric,
            "historical_window": historical_window
        }
    
    # Calculate number of time steps for prediction
    # Convert forecast_horizon (seconds) to time steps based on data sampling rate
    if len(times) >= 2:
        avg_time_step = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else 1.0
        prediction_window_steps = int(forecast_horizon / avg_time_step)
        history_window_steps = int(historical_window / avg_time_step) if historical_window > 0 else len(values)
        forecast_interval_steps = int(forecast_interval / avg_time_step) if avg_time_step > 0 else 1
    else:
        avg_time_step = 60.0  # Assume 60s intervals
        prediction_window_steps = int(forecast_horizon / avg_time_step)
        history_window_steps = len(values)
        forecast_interval_steps = int(forecast_interval / avg_time_step)
    
    # Use generic ARIMA prediction function
    prediction_result = predict_arima(
        time_series=values,
        history_window=history_window_steps,
        prediction_window=prediction_window_steps,
        forecast_interval=forecast_interval_steps
    )
    
    # Check for errors
    if prediction_result.get("error"):
        return {
            "error": prediction_result["error"],
            "highway_segment_id": highway_segment_id,
            "metric": metric
        }
    
    predicted_values = prediction_result["predicted_values"]
    confidence_lower = prediction_result["confidence_lower"]
    confidence_upper = prediction_result["confidence_upper"]
    model_info = prediction_result["model_info"]
    
    # Generate forecast time points (relative to latest time)
    latest_time = times[-1] if times else 0
    forecast_times = [forecast_interval * (i + 1) for i in range(len(predicted_values))]
    
    # Analyze trend and congestion forecast
    mean_value = model_info.get("mean", np.mean(values))
    std_value = model_info.get("std", np.std(values))
    
    if len(predicted_values) >= 2:
        trend_slope = (predicted_values[-1] - predicted_values[0]) / len(predicted_values)
        if trend_slope > std_value * 0.1:
            trend = "increasing"
        elif trend_slope < -std_value * 0.1:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "stable"
    
    # Determine congestion forecast based on metric
    if metric in ['occupancy', 'density']:
        # Higher values = worse congestion
        if trend == "increasing":
            congestion_forecast = "worsening"
            recommended_action = "Apply early speed reduction on upstream segments to prevent congestion buildup"
        elif trend == "decreasing":
            congestion_forecast = "improving"
            recommended_action = "Consider gradual speed limit increase in later part of cycle"
        else:
            congestion_forecast = "stable"
            recommended_action = "Maintain current speed limit strategy"
    elif metric == 'speed':
        # Lower values = worse congestion
        if trend == "decreasing":
            congestion_forecast = "worsening"
            recommended_action = "Apply early speed reduction on upstream segments to prevent congestion buildup"
        elif trend == "increasing":
            congestion_forecast = "improving"
            recommended_action = "Consider gradual speed limit increase in later part of cycle"
        else:
            congestion_forecast = "stable"
            recommended_action = "Maintain current speed limit strategy"
    else:  # throughput
        # Lower values = worse congestion
        if trend == "decreasing":
            congestion_forecast = "worsening"
            recommended_action = "Apply early speed reduction on upstream segments to prevent congestion buildup"
        elif trend == "increasing":
            congestion_forecast = "improving"
            recommended_action = "Consider gradual speed limit increase in later part of cycle"
        else:
            congestion_forecast = "stable"
            recommended_action = "Maintain current speed limit strategy"
    
    # Find peak time and value
    peak_idx = np.argmax(np.abs(np.array(predicted_values) - mean_value))
    peak_time = forecast_times[peak_idx] if forecast_times else 0
    peak_value = predicted_values[peak_idx] if predicted_values else 0
    
    return {
        "highway_segment_id": highway_segment_id,
        "metric": metric,
        "forecast_horizon": forecast_horizon,
        "forecast_interval": forecast_interval,
        "historical_data": {
            "times": times,
            "values": values,
            "mean": float(mean_value),
            "std": float(std_value)
        },
        "forecast": {
            "times": forecast_times,
            "predicted_values": predicted_values,
            "confidence_lower": confidence_lower,
            "confidence_upper": confidence_upper
        },
        "analysis": {
            "trend": trend,
            "congestion_forecast": congestion_forecast,
            "peak_time": float(peak_time),
            "peak_value": float(peak_value),
            "recommended_action": recommended_action
        },
        "model_info": model_info
    }
