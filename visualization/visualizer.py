import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from typing import List, Dict, Optional, Tuple, Any, Union

class Visualizer:
    """Base class for visualization utilities"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
    
    def create_bar_chart(
        self, 
        data: pd.DataFrame, 
        x_col: str, 
        y_col: str, 
        title: str,
        color_col: Optional[str] = None,
        orientation: str = 'v',
        barmode: str = 'group'
    ) -> Figure:
        """
        Create a bar chart
        
        Args:
            data: DataFrame with the data
            x_col: Column to use for x-axis
            y_col: Column to use for y-axis
            title: Chart title
            color_col: Column to use for color coding
            orientation: Bar orientation ('v' for vertical, 'h' for horizontal)
            barmode: Bar mode ('group', 'stack', etc.)
            
        Returns:
            Plotly figure object
        """
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            orientation=orientation,
            barmode=barmode
        )
        return fig
    
    def create_pie_chart(
        self, 
        values: List[float], 
        names: List[str], 
        title: str
    ) -> Figure:
        """
        Create a pie chart
        
        Args:
            values: List of values
            names: List of category names
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = px.pie(
            values=values,
            names=names,
            title=title
        )
        return fig
    
    def create_time_series(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        title: str,
        show_trend: bool = True,
        rolling_window: int = 5
    ) -> Figure:
        """
        Create a time series plot with optional trend lines
        
        Args:
            data: DataFrame with time series data
            x_col: Column to use for x-axis (typically a date/time column)
            y_cols: Columns to plot as y values
            title: Chart title
            show_trend: Whether to show trend lines
            rolling_window: Window size for rolling average
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add a trace for each y column
        for col in y_cols:
            # Main data trace
            fig.add_trace(
                go.Scatter(
                    x=data[x_col],
                    y=data[col],
                    mode='markers+lines',
                    name=col
                )
            )
            
            # Add rolling average if requested
            if show_trend and len(data) > 1:
                y_rolling = data[col].rolling(window=rolling_window, min_periods=1).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data[x_col],
                        y=y_rolling,
                        mode='lines',
                        line=dict(width=2, dash='dash'),
                        name=f'{col} ({rolling_window}-pt avg)'
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            hovermode='x unified'
        )
        
        return fig
    
    def create_radar_chart(self, metrics: Dict[str, float], max_value: float = 100) -> Figure:
        """
        Create a radar chart for visualizing multiple metrics
        
        Args:
            metrics: Dictionary mapping metric names to values
            max_value: Maximum value for the radar chart
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=list(metrics.values()),
            theta=list(metrics.keys()),
            fill='toself',
            name='Metrics'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_value]
                )
            ),
            showlegend=False
        )
        
        return fig
    
    def create_stacked_bar(
        self,
        categories: Dict[str, Union[int, float]],
        title: str,
        show_percentages: bool = True,
        orientation: str = 'h'
    ) -> Figure:
        """
        Create a stacked horizontal bar chart
        
        Args:
            categories: Dictionary mapping categories to values
            title: Chart title
            show_percentages: Whether to show percentages in the chart
            orientation: Bar orientation ('h' for horizontal, 'v' for vertical)
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Remove any zero values
        categories = {k: v for k, v in categories.items() if v > 0}
        
        # Calculate total for percentages
        total = sum(categories.values())
        percentages = {k: (v/total*100) for k, v in categories.items()}
        
        # Add a trace for each category
        for category, value in categories.items():
            if orientation == 'h':
                fig.add_trace(go.Bar(
                    x=[value],
                    y=['Values'],
                    name=category,
                    orientation='h',
                    text=[f"{category}: {percentages[category]:.1f}%"] if show_percentages else None,
                    textposition='inside'
                ))
            else:
                fig.add_trace(go.Bar(
                    x=['Values'],
                    y=[value],
                    name=category,
                    text=[f"{category}: {percentages[category]:.1f}%"] if show_percentages else None,
                    textposition='inside'
                ))
        
        # Configure the layout
        fig.update_layout(
            title=title,
            barmode='stack',
            showlegend=True
        )
        
        return fig
    
    def create_distribution_histogram(
        self,
        data: pd.DataFrame,
        value_col: str,
        color_col: Optional[str] = None,
        bins: int = 20,
        title: str = 'Distribution'
    ) -> Figure:
        """
        Create a histogram showing value distribution
        
        Args:
            data: DataFrame with the data
            value_col: Column containing values to plot
            color_col: Column to use for color coding
            bins: Number of histogram bins
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = px.histogram(
            data,
            x=value_col,
            color=color_col,
            nbins=bins,
            title=title
        )
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame, title: str = 'Correlation Matrix') -> Figure:
        """
        Create a correlation heatmap
        
        Args:
            data: DataFrame with numeric columns to correlate
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        # Calculate correlation matrix
        corr_df = data.select_dtypes(include=['number']).corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_df,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title=title,
            zmin=-1, zmax=1
        )
        
        return fig
