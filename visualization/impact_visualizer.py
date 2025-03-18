import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from typing import List, Dict, Optional, Tuple, Any, Union
from .visualizer import Visualizer

class ImpactVisualizer(Visualizer):
    """Visualizer for tennis racket impact positions"""
    
    def create_scatter_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: Optional[str] = None,
        color_by: Optional[str] = None,
        plot_type: str = '2D',
        jitter: bool = False,
        jitter_amount: float = 0.05,
        title_suffix: str = ""
    ) -> Figure:
        """
        Create scatter plot for impact positions
        
        Args:
            df: DataFrame with impact data
            x_col: Column to use for x-axis
            y_col: Column to use for y-axis
            z_col: Column to use for z-axis (3D only)
            color_by: Column to use for color coding points
            plot_type: Plot type ('2D' or '3D')
            jitter: Whether to add jitter to the data
            jitter_amount: Amount of jitter to add
            title_suffix: Additional text to add to plot title
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title="No data available")
            return fig
            
        # Create a copy to avoid modifying the original
        plot_df = df.copy()
        
        # Apply jitter if requested
        if jitter:
            jitter_cols = [x_col, y_col]
            if z_col:
                jitter_cols.append(z_col)
                
            for col in jitter_cols:
                if col in plot_df.columns:
                    plot_df[col] = plot_df[col] + np.random.normal(0, jitter_amount, len(plot_df))
        
        # Create title
        base_title = f'{x_col} vs {y_col}'
        if z_col:
            base_title += f' vs {z_col}'
            
        title = f"{base_title} (Rotated {self.config.ROTATION_ANGLE}°{title_suffix})"
        
        # Create plot based on type
        if plot_type == '2D':
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_by,
                title=title,
                color_continuous_scale='Viridis' if color_by not in [None, 'None', 'stroke', 'impact_region'] else None
            )
            
            # Set marker size
            fig.update_traces(marker=dict(size=self.config.POINT_SIZE))
            
            # Add racket reference shape
            self._add_racket_reference_2d(fig)
            
        else:  # 3D plot
            if not z_col:
                z_col = 'impact_region'  # Fallback to impact region
                
            fig = px.scatter_3d(
                plot_df,
                x=x_col,
                y=y_col,
                z=z_col,
                color=color_by,
                title=title,
                color_continuous_scale='Viridis' if color_by not in [None, 'None', 'stroke', 'impact_region'] else None
            )
            
            # Set marker size
            fig.update_traces(marker=dict(size=self.config.POINT_SIZE))
            
            # Set 3D camera angle and aspect ratio
            fig.update_layout(
                scene=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    aspectmode='cube'
                )
            )
            
            # Add 3D reference planes
            self._add_racket_reference_3d(fig, plot_df, x_col, y_col, z_col)
        
        return fig
    
    def _add_racket_reference_2d(self, fig: Figure) -> None:
        """Add 2D racket reference shape to figure"""
        # Add racket outline (rectangle)
        fig.add_shape(
            type="rect",
            x0=-1, y0=-1, 
            x1=1, y1=1,
            line=dict(color="rgba(0,0,0,0.3)"),
            fillcolor="rgba(0,0,0,0)"
        )
        
        # Add center lines
        fig.add_shape(
            type="line",
            x0=-1, y0=0, 
            x1=1, y1=0,
            line=dict(color="rgba(0,0,0,0.3)", dash="dash")
        )
        
        fig.add_shape(
            type="line",
            x0=0, y0=-1, 
            x1=0, y1=1,
            line=dict(color="rgba(0,0,0,0.3)", dash="dash")
        )
        
        # Add sweet spot circle
        fig.add_shape(
            type="circle",
            x0=-0.3, y0=-0.3,
            x1=0.3, y1=0.3,
            line=dict(color="rgba(0,128,0,0.3)"),
            fillcolor="rgba(0,128,0,0.1)"
        )
    
    def _add_racket_reference_3d(
        self, 
        fig: Figure, 
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str
    ) -> None:
        """Add 3D racket reference planes to figure"""
        # Get axis ranges for proper plane sizing
        x_min, x_max = df[x_col].min(), df[x_col].max()
        y_min, y_max = df[y_col].min(), df[y_col].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.2
        y_padding = (y_max - y_min) * 0.2
        
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        # Create grid for racket plane
        x_grid = np.linspace(x_min, x_max, 10)
        y_grid = np.linspace(y_min, y_max, 10)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        z_mesh = np.zeros_like(x_mesh)  # Flat plane at z=0
        
        # Add racket plane
        fig.add_trace(
            go.Surface(
                x=x_mesh,
                y=y_mesh,
                z=z_mesh,
                colorscale=[[0, 'rgba(200,200,200,0.2)'], [1, 'rgba(200,200,200,0.2)']],
                showscale=False,
                name="Racket Plane",
                hoverinfo='skip'
            )
        )
        
        # Add grid lines
        for x in x_grid:
            fig.add_trace(
                go.Scatter3d(
                    x=[x, x],
                    y=[y_min, y_max],
                    z=[0, 0],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0.2)', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
        for y in y_grid:
            fig.add_trace(
                go.Scatter3d(
                    x=[x_min, x_max],
                    y=[y, y],
                    z=[0, 0],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0.2)', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
    
    def create_animated_plot(
        self,
        df: pd.DataFrame,
        session_frames: Dict[str, pd.DataFrame],
        selected_session: str,
        x_col: str,
        y_col: str,
        z_col: Optional[str] = None,
        color_by: Optional[str] = None,
        plot_type: str = '2D',
        jitter: bool = False,
        jitter_amount: float = 0.05
    ) -> Figure:
        """
        Create animated scatter plot showing points being added over time
        
        Args:
            df: Full dataframe with all sessions
            session_frames: Dictionary of session dataframes
            selected_session: Key of the selected session to animate
            x_col: Column to use for x-axis
            y_col: Column to use for y-axis
            z_col: Column to use for z-axis (3D only)
            color_by: Column to use for color coding points
            plot_type: Plot type ('2D' or '3D')
            jitter: Whether to add jitter to the data
            jitter_amount: Amount of jitter to add
            
        Returns:
            Plotly figure object
        """
        # Get the session data
        if selected_session not in session_frames:
            return self.create_scatter_plot(
                df.head(0),  # Empty dataframe
                x_col, y_col, z_col, color_by, plot_type,
                jitter, jitter_amount, " - No session selected"
            )
            
        session_df = session_frames[selected_session]
        
        # Apply jitter if selected
        if jitter:
            jitter_cols = [x_col, y_col]
            if z_col:
                jitter_cols.append(z_col)
                
            for col in jitter_cols:
                if col in session_df.columns:
                    session_df[col] = session_df[col] + np.random.normal(0, jitter_amount, len(session_df))
        
        # Get session date for title
        session_date = "Session"
        if 'session_date' in session_df.columns:
            session_date = session_df['session_date'].iloc[0]
            if hasattr(session_date, 'strftime'):
                session_date = session_date.strftime('%b %d, %Y')
                
        # Create title
        base_title = f'{x_col} vs {y_col}'
        if z_col:
            base_title += f' vs {z_col}'
            
        title = f"{base_title} - Session: {session_date} (Rotated {self.config.ROTATION_ANGLE}°)"
        
        # Check if session has data
        if session_df.empty:
            fig = go.Figure()
            fig.update_layout(title=f"{title} - No data available")
            return fig
        
        # Create frames for animation
        frames = []
        
        for i in range(1, len(session_df) + 1):
            frame_df = session_df.iloc[0:i]
            
            if plot_type == '2D':
                frame = go.Frame(
                    data=[go.Scatter(
                        x=frame_df[x_col],
                        y=frame_df[y_col],
                        mode='markers',
                        marker=dict(
                            color=frame_df[color_by] if color_by in frame_df.columns else None,
                            size=self.config.POINT_SIZE
                        )
                    )],
                    name=f'frame{i}'
                )
            else:  # 3D
                frame = go.Frame(
                    data=[go.Scatter3d(
                        x=frame_df[x_col],
                        y=frame_df[y_col],
                        z=frame_df[z_col] if z_col in frame_df.columns else [0] * len(frame_df),
                        mode='markers',
                        marker=dict(
                            color=frame_df[color_by] if color_by in frame_df.columns else None,
                            size=self.config.POINT_SIZE
                        )
                    )],
                    name=f'frame{i}'
                )
                
            frames.append(frame)
        
        # Create base figure with first frame
        if plot_type == '2D':
            fig = px.scatter(
                session_df.iloc[0:1],
                x=x_col,
                y=y_col,
                color=color_by if color_by in session_df.columns else None,
                title=title
            )
            # Add reference shapes
            self._add_racket_reference_2d(fig)
        else:
            fig = px.scatter_3d(
                session_df.iloc[0:1],
                x=x_col,
                y=y_col,
                z=z_col if z_col in session_df.columns else None,
                color=color_by if color_by in session_df.columns else None,
                title=title
            )
            # Set 3D camera angle
            fig.update_layout(
                scene=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    aspectmode='cube'
                )
            )
            # Add reference planes
            self._add_racket_reference_3d(fig, session_df, x_col, y_col, z_col if z_col in session_df.columns else 'impact_region')
        
        # Add frames to figure
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None, 
                            {"frame": {"duration": self.config.ANIMATION_SPEED_MS, "redraw": True},
                             "fromcurrent": True, 
                             "transition": {"duration": 0}}
                        ]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None], 
                            {"frame": {"duration": 0, "redraw": False},
                             "mode": "immediate",
                             "transition": {"duration": 0}}
                        ]
                    )
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 10},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'y': 0,
                'xanchor': 'right',
                'yanchor': 'top'
            }]
        )
        
        # Add slider
        sliders = [{
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [
                        [f'frame{i}'],
                        {'frame': {'duration': 0, 'redraw': False},
                         'mode': 'immediate',
                         'transition': {'duration': 0}}
                    ],
                    'label': f'{i}/{len(session_df)}',
                    'method': 'animate'
                }
                for i in range(1, len(session_df) + 1, max(1, len(session_df) // 10))
            ]
        }]
        
        fig.update_layout(sliders=sliders)
        
        return fig
    
    def create_sequential_plot(
        self,
        session_df: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: Optional[str] = None,
        color_by: Optional[str] = None,
        plot_type: str = '2D',
        frame_index: int = 0
    ) -> Figure:
        """
        Create a plot showing points sequentially for manual control
        
        Args:
            session_df: Session dataframe
            x_col: Column to use for x-axis
            y_col: Column to use for y-axis
            z_col: Column to use for z-axis (3D only)
            color_by: Column to use for color coding points
            plot_type: Plot type ('2D' or '3D')
            frame_index: Index of the current frame
            
        Returns:
            Plotly figure object
        """
        if session_df.empty:
            fig = go.Figure()
            fig.update_layout(title="No data available")
            return fig
            
        # Make sure frame_index is valid
        frame_index = min(max(0, frame_index), len(session_df) - 1)
        
        # Get session date for title
        session_date = "Current Session"
        if 'session_date' in session_df.columns:
            session_date = session_df['session_date'].iloc[0]
            if hasattr(session_date, 'strftime'):
                session_date = session_date.strftime('%b %d, %Y')
                
        # Create title
        base_title = f'{x_col} vs {y_col}'
        if z_col:
            base_title += f' vs {z_col}'
            
        title = f"{base_title} - Session: {session_date} (Frame {frame_index+1}/{len(session_df)})"
        
        # Get data up to current frame
        plot_df = session_df.iloc[0:frame_index+1].copy()
        
        # Ensure we have data to plot
        if plot_df.empty:
            plot_df = session_df.iloc[0:1].copy()
            
        # Calculate fixed axis ranges for consistent view
        x_min, x_max = session_df[x_col].min(), session_df[x_col].max()
        y_min, y_max = session_df[y_col].min(), session_df[y_col].max()
        
        # Add padding to axis ranges
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        x_range = [x_min - x_padding, x_max + x_padding]
        y_range = [y_min - y_padding, y_max + y_padding]
        
        # Create figure based on plot type
        if plot_type == '2D':
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_by if color_by in plot_df.columns else None,
                title=title,
                range_x=x_range,
                range_y=y_range
            )
            
            # Set marker size
            fig.update_traces(marker=dict(size=self.config.POINT_SIZE))
            
            # Add reference shapes
            self._add_racket_reference_2d(fig)
            
            # Show all session points in light gray for context
            fig.add_trace(
                go.Scatter(
                    x=session_df[x_col],
                    y=session_df[y_col],
                    mode='markers',
                    marker=dict(
                        color='rgba(200,200,200,0.3)',
                        size=self.config.POINT_SIZE - 2
                    ),
                    name='All Points',
                    hoverinfo='skip'
                )
            )
            
            # Move to bottom layer
            fig.data = fig.data[::-1]
            
        else:  # 3D plot
            if not z_col or z_col not in session_df.columns:
                z_col = 'impact_region' if 'impact_region' in session_df.columns else list(session_df.columns)[0]
                
            # Calculate z range for consistent view
            z_min, z_max = session_df[z_col].min(), session_df[z_col].max()
            z_padding = (z_max - z_min) * 0.1
            z_range = [z_min - z_padding, z_max + z_padding]
            
            fig = px.scatter_3d(
                plot_df,
                x=x_col,
                y=y_col,
                z=z_col,
                color=color_by if color_by in plot_df.columns else None,
                title=title,
                range_x=x_range,
                range_y=y_range,
                range_z=z_range
            )
            
            # Set marker size
            fig.update_traces(marker=dict(size=self.config.POINT_SIZE))
            
            # Set 3D camera angle
            fig.update_layout(
                scene=dict(
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    aspectmode='cube'
                )
            )
            
            # Add reference planes
            self._add_racket_reference_3d(fig, session_df, x_col, y_col, z_col)
            
            # Show all session points in light gray for context
            fig.add_trace(
                go.Scatter3d(
                    x=session_df[x_col],
                    y=session_df[y_col],
                    z=session_df[z_col],
                    mode='markers',
                    marker=dict(
                        color='rgba(200,200,200,0.3)',
                        size=self.config.POINT_SIZE - 2
                    ),
                    name='All Points',
                    hoverinfo='skip'
                )
            )
            
            # Move to bottom layer
            fig.data = fig.data[::-1]
        
        # Highlight the current point
        current_point = session_df.iloc[frame_index:frame_index+1]
        
        if plot_type == '2D':
            fig.add_trace(
                go.Scatter(
                    x=current_point[x_col],
                    y=current_point[y_col],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=self.config.POINT_SIZE + 2,
                        line=dict(
                            color='black',
                            width=1
                        )
                    ),
                    name='Current Point'
                )
            )
        else:  # 3D
            fig.add_trace(
                go.Scatter3d(
                    x=current_point[x_col],
                    y=current_point[y_col],
                    z=current_point[z_col],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=self.config.POINT_SIZE + 2,
                        line=dict(
                            color='black',
                            width=1
                        )
                    ),
                    name='Current Point'
                )
            )
        
        return fig
        
    def create_hit_points_chart(self, hit_points: List[List[float]], title: str) -> Figure:
        """
        Create a scatter plot for hit points
        
        Args:
            hit_points: List of [x, y] coordinate pairs
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        if not hit_points:
            fig = go.Figure()
            fig.update_layout(title="No hit points data available")
            return fig
            
        # Convert to dataframe
        df_hits = pd.DataFrame(hit_points, columns=['x', 'y'])
        
        fig = px.scatter(
            df_hits,
            x='x',
            y='y',
            title=title,
            labels={'x': 'Horizontal Position', 'y': 'Vertical Position'}
        )
        
        # Add racket reference frame
        self._add_racket_reference_2d(fig)
        
        # Add annotations
        fig.add_annotation(
            x=0, y=1.1,
            text="Top",
            showarrow=False
        )
        
        fig.add_annotation(
            x=0, y=-1.1,
            text="Bottom",
            showarrow=False
        )
        
        return fig
