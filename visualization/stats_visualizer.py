import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from typing import List, Dict, Optional, Tuple, Any, Union
from .visualizer import Visualizer

class StatsVisualizer(Visualizer):
    """Visualizer for tennis statistics and metrics"""
    
    def create_score_history_chart(
        self, 
        df: pd.DataFrame,
        time_col: str = 'start_time',
        metrics: List[str] = ['session_score', 'consistency_score', 'power_score'],
        rolling_window: int = 5,
        show_rolling_avg: bool = True,
        show_trendline: bool = True
    ) -> Figure:
        """
        Create a chart showing score history over time
        
        Args:
            df: DataFrame with session data
            time_col: Column containing timestamps
            metrics: List of metrics to plot
            rolling_window: Window size for rolling average
            show_rolling_avg: Whether to show rolling average
            show_trendline: Whether to show trend lines
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="No data available for score history")
            return fig
            
        # Ensure time column is datetime type
        if time_col in df.columns and not pd.api.types.is_datetime64_dtype(df[time_col]):
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col])
            
        # Sort by time
        df = df.sort_values(time_col)
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each metric
        for metric in metrics:
            if metric in df.columns:
                # Add raw data trace
                fig.add_trace(
                    go.Scatter(
                        x=df[time_col],
                        y=df[metric],
                        mode='markers+lines',
                        name=metric,
                        line=dict(color=self.config.PLOT_COLORS.get(metric.split('_')[0], None))
                    )
                )
                
                # Add rolling average if requested
                if show_rolling_avg and len(df) > rolling_window:
                    rolling = df[metric].rolling(window=rolling_window, min_periods=1).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=df[time_col],
                            y=rolling,
                            mode='lines',
                            name=f"{metric} ({rolling_window}-pt avg)",
                            line=dict(
                                color=self.config.PLOT_COLORS.get(metric.split('_')[0], None),
                                dash='dot',
                                width=2
                            )
                        )
                    )
                
                # Add trend line if requested
                if show_trendline and len(df) > 2:
                    # Convert to numeric for polynomial fit
                    x_numeric = np.arange(len(df))
                    
                    # Fit a line to the data
                    z = np.polyfit(x_numeric, df[metric], 1)
                    p = np.poly1d(z)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df[time_col],
                            y=p(x_numeric),
                            mode='lines',
                            name=f"{metric} trend",
                            line=dict(
                                color=self.config.PLOT_COLORS.get(metric.split('_')[0], None),
                                dash='dash',
                                width=1
                            )
                        )
                    )
        
        # Update layout
        fig.update_layout(
            title="Score History",
            xaxis_title="Date",
            yaxis_title="Score",
            hovermode='x unified',
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def create_monthly_averages_chart(
        self, 
        df: pd.DataFrame,
        time_col: str = 'start_time',
        metrics: List[str] = ['session_score', 'consistency_score', 'power_score']
    ) -> Figure:
        """
        Create a bar chart showing monthly averages
        
        Args:
            df: DataFrame with session data
            time_col: Column containing timestamps
            metrics: List of metrics to show monthly averages for
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="No data available for monthly averages")
            return fig
            
        # Ensure time column is datetime type
        if time_col in df.columns and not pd.api.types.is_datetime64_dtype(df[time_col]):
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Create month column for grouping
        df['month'] = df[time_col].dt.strftime('%b %Y')
        
        # Group by month and calculate average for each metric
        monthly_data = []
        
        for month, group in df.groupby('month'):
            month_stats = {'month': month}
            
            for metric in metrics:
                if metric in df.columns:
                    month_stats[metric] = group[metric].mean()
                    
            monthly_data.append(month_stats)
            
        # Convert to dataframe
        monthly_df = pd.DataFrame(monthly_data)
        
        # Sort by month chronologically
        try:
            # Convert to datetime for sorting
            sort_dates = pd.to_datetime(monthly_df['month'], format='%b %Y')
            monthly_df = monthly_df.iloc[sort_dates.argsort()]
        except:
            # Fallback to original order if sorting fails
            pass
        
        # Create bar chart
        metric_cols = [col for col in metrics if col in monthly_df.columns]
        
        fig = px.bar(
            monthly_df,
            x='month',
            y=metric_cols,
            title="Monthly Average Scores",
            labels={'value': 'Score', 'variable': 'Metric', 'month': 'Month'},
            barmode='group'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Score",
            legend_title="Metric"
        )
        
        return fig
    
    def create_time_series(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        title: str,
        show_trend: bool = True,
        rolling_window: int = 5
    ) -> Figure:
        """
        Create a time series plot with optional trend lines
        
        Args:
            df: DataFrame with time series data
            x_col: Column to use for x-axis (typically a date/time column)
            y_cols: Columns to plot as y values
            title: Chart title
            show_trend: Whether to show trend lines
            rolling_window: Window size for rolling average
            
        Returns:
            Plotly figure object
        """
        if df.empty or not y_cols:
            fig = go.Figure()
            fig.update_layout(title="No data available")
            return fig
            
        # Create figure
        fig = go.Figure()
        
        # Ensure x column is datetime if possible
        if x_col in df.columns and not pd.api.types.is_datetime64_dtype(df[x_col]):
            try:
                plot_df = df.copy()
                plot_df[x_col] = pd.to_datetime(plot_df[x_col])
            except:
                plot_df = df
        else:
            plot_df = df
        
        # Add a trace for each y column
        for col in y_cols:
            if col in plot_df.columns:
                # Main data trace
                fig.add_trace(
                    go.Scatter(
                        x=plot_df[x_col],
                        y=plot_df[col],
                        mode='markers+lines',
                        name=col
                    )
                )
                
                # Add rolling average if requested
                if show_trend and len(plot_df) > 1:
                    y_rolling = plot_df[col].rolling(window=min(rolling_window, len(plot_df)), min_periods=1).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=plot_df[x_col],
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
    
    def create_distribution_histogram(
        self,
        df: pd.DataFrame,
        value_col: str,
        color_col: Optional[str] = None,
        bins: int = 20,
        title: str = 'Distribution'
    ) -> Figure:
        """
        Create a histogram showing value distribution
        
        Args:
            df: DataFrame with the data
            value_col: Column containing values to plot
            color_col: Column to use for color coding
            bins: Number of histogram bins
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        if df.empty or value_col not in df.columns:
            fig = go.Figure()
            fig.update_layout(title="No data available")
            return fig
            
        fig = px.histogram(
            df,
            x=value_col,
            color=color_col if color_col in df.columns else None,
            nbins=bins,
            title=title
        )
        
        return fig
    
    def create_speed_comparison_chart(
        self, 
        df: pd.DataFrame,
        shot_types: List[str] = ['serve', 'forehand', 'backhand'],
        metric: str = 'max_speed',
        conversion_factor: float = 2.25  # m/s to mph
    ) -> Figure:
        """
        Create a comparison chart for speeds of different shot types
        
        Args:
            df: DataFrame with session data
            shot_types: List of shot types to compare
            metric: Speed metric to compare ('max_speed' or 'avg_speed')
            conversion_factor: Factor to convert speeds
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="No data available for speed comparison")
            return fig
            
        # Create a list of metrics to plot
        metrics_to_plot = [f"{shot_type}_{metric}" for shot_type in shot_types]
        
        # Filter for existing columns
        existing_metrics = [m for m in metrics_to_plot if m in df.columns]
        
        if not existing_metrics:
            fig = go.Figure()
            fig.update_layout(title="No speed data available")
            return fig
        
        # Create a copy of the dataframe with converted speeds
        plot_df = df.copy()
        
        # Apply conversion
        for col in existing_metrics:
            plot_df[f"{col}_mph"] = plot_df[col] * conversion_factor
        
        # Create a list of converted metrics
        metrics_mph = [f"{m}_mph" for m in existing_metrics]
        
        # Create bar chart
        fig = px.bar(
            x=[m.split('_')[0].capitalize() for m in existing_metrics],
            y=[plot_df[m].mean() for m in metrics_mph],
            title=f"Average {metric.replace('_', ' ').title()} by Shot Type",
            labels={'x': 'Shot Type', 'y': f"{metric.replace('_', ' ').title()} (mph)"}
        )
        
        return fig
