import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional, Any, Union

def rotate_point(x: float, y: float, angle_degrees: float) -> Tuple[float, float]:
    """
    Rotate a point (x, y) by a given angle in degrees
    
    Args:
        x: x-coordinate of the point
        y: y-coordinate of the point
        angle_degrees: rotation angle in degrees
    
    Returns:
        Tuple of (rotated_x, rotated_y)
    """
    angle_rad = np.radians(angle_degrees)
    rotated_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    rotated_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return rotated_x, rotated_y

def format_duration(milliseconds: int) -> str:
    """
    Format duration in milliseconds to MM:SS format
    
    Args:
        milliseconds: Duration in milliseconds
        
    Returns:
        Formatted duration string (MM:SS)
    """
    seconds = milliseconds / 1000
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default if denominator is zero
    
    Args:
        numerator: Division numerator
        denominator: Division denominator
        default: Default value to return if division is unsafe
        
    Returns:
        Result of division or default value
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default

def calculate_rolling_average(values: List[float], window_size: int = 5) -> List[float]:
    """
    Calculate rolling average of a list of values
    
    Args:
        values: List of values to average
        window_size: Size of the rolling window
        
    Returns:
        List of rolling averages
    """
    if not values:
        return []
        
    # Ensure window size is valid
    window_size = min(window_size, len(values))
    window_size = max(1, window_size)
    
    # Calculate rolling average
    result = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        window = values[start_idx:i+1]
        result.append(sum(window) / len(window))
    
    return result

def get_standard_features() -> List[str]:
    """
    Get standard features for summary statistics
    
    Returns:
        List of feature names
    """
    return [
        'impact_region', 'rotated_x', 'modified_rotated_y',
        'power', 'racket_speed', 'ball_spin', 'backswing_time'
    ]

def get_color_options() -> List[str]:
    """
    Get list of options for coloring data points
    
    Returns:
        List of color options
    """
    return [
        "None", "stroke", "impact_region", "power", "ball_spin", 
        "racket_speed", "impact_position_x", "impact_position_y"
    ]

def calculate_shot_breakdown(session_data: Dict) -> Optional[Dict[str, Any]]:
    """
    Calculate shot type breakdown from session data
    
    Args:
        session_data: Session data dictionary
        
    Returns:
        Dictionary of shot breakdown stats or None if invalid data
    """
    try:
        if 'swings' not in session_data:
            return None
            
        swings = session_data['swings']
        shots = {
            'serve': swings.get('serve', {}).get('serve_swings', 0),
            'forehand': sum(swings.get('forehand', {}).get(f'{type}_swings', 0) 
                          for type in ['flat', 'slice', 'topspin']),
            'backhand': sum(swings.get('backhand', {}).get(f'{type}_swings', 0) 
                          for type in ['flat', 'slice', 'topspin']),
            'volley': 0,  # Default to 0 for these types
            'smash': 0
        }
        
        # Calculate total and handle division by zero
        shots['total'] = sum(shots.values())
        if shots['total'] == 0:
            return None
            
        # Add percentages
        result = {
            'Serve': shots['serve'],
            'Forehand': shots['forehand'],
            'Backhand': shots['backhand'],
            'Volley': shots['volley'],
            'Smash': shots['smash'],
            'Total': shots['total'],
        }
        
        # Calculate percentages
        for shot_type in ['Serve', 'Forehand', 'Backhand', 'Volley', 'Smash']:
            result[f'{shot_type}_pct'] = (result[shot_type] / shots['total'] * 100) 
            
        return result
    except Exception:
        return None

def get_feature_options() -> List[str]:
    """
    Get standard feature options for visualizations
    
    Returns:
        List of feature names
    """
    return [
        'impact_region', 'rotated_x', 'rotated_y', 'modified_rotated_y',
        'power', 'racket_speed', 'ball_spin', 'backswing_time',
        'impact_position_x', 'impact_position_y'
    ]
