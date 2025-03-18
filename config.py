from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

@dataclass
class Config:
    """Configuration settings for the Tennis Analytics Dashboards"""
    
    # Database paths
    DB_PATH: Path = Path('./data/ztennis.db')
    
    # Time and timezone settings
    TIMEZONE: str = 'America/Phoenix'
    
    # Rotation settings for impact visualization
    ROTATION_ANGLE: float = 333.5
    
    # Visualization defaults
    DEFAULT_PLOT_TYPE: str = "2D"
    DEFAULT_JITTER: bool = True
    DEFAULT_COLOR_BY: str = "impact_region"
    DEFAULT_JITTER_AMOUNT: float = 0.05
    POINT_SIZE: int = 6
    
    # Animation settings
    ANIMATION_SPEED_MS: int = 500
    ANIMATION_TRAIL_LENGTH: int = 20
    
    # Conversion factors and thresholds
    SPEED_CONVERSION_FACTOR: float = 2.25  # m/s to mph
    DEFAULT_ROLLING_WINDOW: int = 5
    MIN_SPEED_THRESHOLD: float = 50.0
    
    # Plot colors
    PLOT_COLORS: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived attributes after initialization"""
        # Set default colors if not provided
        if not self.PLOT_COLORS:
            self.PLOT_COLORS = {
                # Stroke types
                'SLICEFH': '#1f77b4',
                'TOPSPINBH': '#ff7f0e',
                'FLATFH': '#2ca02c',
                'FLATBH': '#d62728',
                'SERVEFH': '#9467bd',
                'TOPSPINFH': '#8c564b',
                'SLICEBH': '#e377c2',
                
                # Metrics
                'best_piq': '#2ecc71',
                'avg_piq': '#3498db',
                'forehand': '#e67e22',
                'backhand': '#9b59b6',
                'serve': '#e74c3c',
                'consistency': '#1f77b4',
                'power': '#ff7f0e',
                'intensity': '#2ca02c',
                'overall': '#d62728'
            }
