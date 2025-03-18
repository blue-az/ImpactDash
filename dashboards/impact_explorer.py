import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from .base_dashboard import BaseDashboard
from core.data_manager import DataManager
from core.session_manager import SessionManager
from core.utils import get_feature_options, get_color_options
from visualization.impact_visualizer import ImpactVisualizer

class ImpactExplorer(BaseDashboard):
    """Dashboard for tennis racket impact position exploration"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        super().__init__(config, "Tennis Impact Position Explorer")
        
        # Initialize components
        self.data_manager = DataManager(config)
        self.session_manager = SessionManager(config)
        self.visualizer = ImpactVisualizer(config)
    
    def _create_sidebar(self):
        """Create sidebar with data and visualization controls"""
        # Data source section
        st.header("Data Source")
        db_path = st.text_input("Database File Path", value=str(self.config.DB_PATH))
        
        # Load initial data if needed
        if st.session_state.df.empty and db_path:
            with st.spinner("Loading initial data..."):
                try:
                    # Load and process data
                    initial_df = self.data_manager.load_data(db_path)
                    processed_df = self.data_manager.process_raw_data(initial_df)
                    
                    # Store in session state
                    st.session_state.df = processed_df
                    
                    # Calculate metrics
                    st.session_state.metrics = self.data_manager.get_session_metrics(processed_df)
                    
                    # Group into sessions
                    st.session_state.sessions = self.data_manager.group_into_sessions(processed_df)
                except Exception as e:
                    st.error(f"Error loading initial data: {str(e)}")
        
        # Data filtering section
        st.header("Data Filtering")
        
        # Create filter controls
        filters = self._create_filter_controls(
            st.session_state.df,
            date_col='date',
            stroke_col='stroke',
            region_col='impact_region'
        )
        
        # Visualization settings
        st.header("Visualization Settings")
        
        # Plot type selection
        plot_type = st.selectbox(
            "Plot Type", 
            ["2D", "3D"],
            index=0 if self.config.DEFAULT_PLOT_TYPE == "2D" else 1
        )
        
        # Feature selection for axes
        features = get_feature_options()
        
        if plot_type == '2D':
            x_axis = st.selectbox(
                "X-Axis Feature", 
                features, 
                index=features.index('rotated_x') if 'rotated_x' in features else 0
            )
            
            y_axis = st.selectbox(
                "Y-Axis Feature", 
                features, 
                index=features.index('modified_rotated_y') if 'modified_rotated_y' in features else 0
            )
            
            z_axis = None
        else:
            x_axis = st.selectbox(
                "X-Axis Feature", 
                features, 
                index=features.index('rotated_x') if 'rotated_x' in features else 0
            )
            
            y_axis = st.selectbox(
                "Y-Axis Feature", 
                features, 
                index=features.index('modified_rotated_y') if 'modified_rotated_y' in features else 0
            )
            
            z_axis = st.selectbox(
                "Z-Axis Feature", 
                features, 
                index=features.index('impact_region') if 'impact_region' in features else 0
            )
        
        # Color by option
        color_options = get_color_options()
        default_color_index = color_options.index(self.config.DEFAULT_COLOR_BY) if self.config.DEFAULT_COLOR_BY in color_options else 0
        
        color_by = st.selectbox(
            "Color Points By",
            color_options,
            index=default_color_index
        )
        
        # Jitter option
        jitter = st.checkbox("Add Jitter", value=self.config.DEFAULT_JITTER)
        
        # Jitter amount slider (only show if jitter is enabled)
        jitter_amount = self.config.DEFAULT_JITTER_AMOUNT
        if jitter:
            jitter_amount = st.slider(
                "Jitter Amount", 
                min_value=0.01, 
                max_value=0.2, 
                value=self.config.DEFAULT_JITTER_AMOUNT,
                step=0.01
            )
        
        # Store settings in session state
        if 'visualization_settings' not in st.session_state:
            st.session_state.visualization_settings = {}
            
        st.session_state.visualization_settings.update({
            'plot_type': plot_type,
            'x_axis': x_axis,
            'y_axis': y_axis,
            'z_axis': z_axis,
            'color_by': None if color_by == "None" else color_by,
            'jitter': jitter,
            'jitter_amount': jitter_amount
        })
        
        # Apply button to load data with selected filters
        if st.button("Apply Filters"):
            with st.spinner("Loading data..."):
                try:
                    # Load and filter data
                    filtered_df = self.data_manager.load_data(db_path)
                    processed_df = self.data_manager.process_raw_data(filtered_df)
                    
                    filtered_df = self.data_manager.filter_data(
                        processed_df,
                        filters['start_date'],
                        filters['end_date'],
                        filters['strokes'],
                        filters['regions']
                    )
                    
                    # Store in session state
                    st.session_state.df = filtered_df
                    
                    # Group into sessions
                    st.session_state.sessions = self.data_manager.group_into_sessions(filtered_df)
                    
                    # Reset current frame
                    st.session_state.current_frame = 0
                    
                    st.success(f"Data loaded successfully. {len(filtered_df)} records found.")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
    
    def _display_main_content(self):
        """Display main dashboard content"""
        if st.session_state.df.empty:
            st.info("No data to display. Please load data using the sidebar controls.")
            return
        
        # Get current settings
        settings = st.session_state.get('visualization_settings', {})
        plot_type = settings.get('plot_type', self.config.DEFAULT_PLOT_TYPE)
        x_axis = settings.get('x_axis', 'rotated_x')
        y_axis = settings.get('y_axis', 'modified_rotated_y')
        z_axis = settings.get('z_axis', None)
        color_by = settings.get('color_by', self.config.DEFAULT_COLOR_BY)
        jitter = settings.get('jitter', self.config.DEFAULT_JITTER)
        jitter_amount = settings.get('jitter_amount', self.config.DEFAULT_JITTER_AMOUNT)
        
        # Display multiple view types using tabs
        tab1, tab2 = st.tabs(["Static View", "Session Animation"])
        
        with tab1:
            self._display_static_view(
                st.session_state.df,
                plot_type, x_axis, y_axis, z_axis,
                color_by, jitter, jitter_amount
            )
            
        with tab2:
            self._display_animation_view(
                st.session_state.df,
                st.session_state.sessions,
                plot_type, x_axis, y_axis, z_axis,
                color_by, jitter, jitter_amount
            )
        
        # Display data summary
        st.header("Data Summary")
        self._display_data_summary(st.session_state.df)
    
    def _display_static_view(
        self, 
        df: pd.DataFrame,
        plot_type: str,
        x_axis: str,
        y_axis: str,
        z_axis: Optional[str],
        color_by: Optional[str],
        jitter: bool,
        jitter_amount: float
    ):
        """
        Display static view of all data points
        
        Args:
            df: DataFrame with impact data
            plot_type: Type of plot ('2D' or '3D')
            x_axis: Column to use for x-axis
            y_axis: Column to use for y-axis
            z_axis: Column to use for z-axis (3D only)
            color_by: Column to use for coloring points
            jitter: Whether to add jitter to the data
            jitter_amount: Amount of jitter to add
        """
        st.header("Impact Position Scatter Plot")
        
        # Create a copy to avoid modifying the original
        plot_df = df.copy()
        
        # Apply jitter if selected
        if jitter:
            jitter_columns = [col for col in [x_axis, y_axis, z_axis] 
                              if col is not None and col in plot_df.columns]
            
            if jitter_columns:
                plot_df = self.data_manager.add_jitter(plot_df, jitter_columns, jitter_amount)
        
        # Create scatter plot
        fig = self.visualizer.create_scatter_plot(
            plot_df, 
            x_axis, 
            y_axis, 
            z_axis, 
            color_by, 
            plot_type, 
            False,  # Jitter already applied above if needed
            jitter_amount
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_animation_view(
        self,
        df: pd.DataFrame,
        sessions: Dict[str, pd.DataFrame],
        plot_type: str,
        x_axis: str,
        y_axis: str,
        z_axis: Optional[str],
        color_by: Optional[str],
        jitter: bool,
        jitter_amount: float
    ):
        """
        Display animated view of session data
        
        Args:
            df: Full dataframe with all data
            sessions: Dictionary of session dataframes
            plot_type: Type of plot ('2D' or '3D')
            x_axis: Column to use for x-axis
            y_axis: Column to use for y-axis
            z_axis: Column to use for z-axis (3D only)
            color_by: Column to use for coloring points
            jitter: Whether to add jitter to the data
            jitter_amount: Amount of jitter to add
        """
        st.header("Session Animation")
        
        if not sessions:
            st.warning("No sessions available. Try adjusting your filters.")
            return
            
        # Session selector
        session_options = list(sessions.keys())
        selected_session = st.selectbox(
            "Select Session to Animate",
            session_options,
            format_func=lambda x: x if not pd.api.types.is_datetime64_dtype(pd.Series([x])) else pd.to_datetime(x).strftime('%A, %B %d, %Y'),
            index=0
        )
        
        # Animation type selector
        animation_type = st.radio(
            "Animation Type",
            ["Automatic (Play/Pause)", "Manual (Slider)"],
            horizontal=True
        )
        
        if animation_type == "Automatic (Play/Pause)":
            # Display automatic animation
            fig = self.visualizer.create_animated_plot(
                df,
                sessions,
                selected_session,
                x_axis,
                y_axis,
                z_axis,
                color_by,
                plot_type,
                jitter,
                jitter_amount
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Manual slider
            if selected_session in sessions:
                session_df = sessions[selected_session]
                
                if session_df.empty:
                    st.warning("This session has no data points.")
                    return
                
                # Apply jitter if selected
                if jitter:
                    jitter_columns = [col for col in [x_axis, y_axis, z_axis] 
                                    if col is not None and col in session_df.columns]
                    
                    if jitter_columns:
                        session_df = self.data_manager.add_jitter(session_df, jitter_columns, jitter_amount)
                
                # Frame slider
                max_frame = max(0, len(session_df) - 1)
                current_frame = min(st.session_state.current_frame, max_frame)
                
                if max_frame > 0:
                    frame_index = st.slider(
                        "Frame",
                        0,
                        max_frame,
                        current_frame
                    )
                else:
                    frame_index = 0
                    st.info("Only one data point available in this session.")
                
                # Update current frame in session state
                st.session_state.current_frame = frame_index
                
                # Create sequential plot
                fig = self.visualizer.create_sequential_plot(
                    session_df,
                    x_axis,
                    y_axis,
                    z_axis,
                    color_by,
                    plot_type,
                    frame_index
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display frame information
                if frame_index < len(session_df):
                    frame_data = session_df.iloc[frame_index]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Frame", f"{frame_index + 1}/{len(session_df)}")
                    with col2:
                        st.metric("Stroke", frame_data.get('stroke', 'N/A'))
                    with col3:
                        st.metric("Impact Region", str(frame_data.get('impact_region', 'N/A')))
        
        # Display session-specific data summary
        if selected_session in sessions:
            session_df = sessions[selected_session]
            
            st.header("Session Summary")
            self._display_session_summary(session_df, selected_session)
    
    def _display_session_summary(self, session_df: pd.DataFrame, session_name: str):
        """
        Display summary information for the current session
        
        Args:
            session_df: Session dataframe
            session_name: Name of the session
        """
        if session_df.empty:
            st.info("No data available for this session.")
            return
        
        # Format session date
        session_date = str(session_name)
        if pd.api.types.is_datetime64_dtype(pd.Series([session_name])):
            session_date = pd.to_datetime(session_name).strftime('%A, %B %d, %Y')
        elif 'session_date' in session_df.columns:
            try:
                session_date = session_df['session_date'].iloc[0]
                if isinstance(session_date, (datetime, date)):
                    session_date = session_date.strftime('%A, %B %d, %Y')
            except:
                pass
                
        # Display basic info
        st.write(f"**Session**: {session_date}")
        st.write(f"**Total Points**: {len(session_df)}")
        
        # Shot distribution
        col1, col2 = st.columns(2)
        
        with col1:
            if 'stroke' in session_df.columns:
                st.subheader("Stroke Distribution")
                
                stroke_counts = session_df['stroke'].value_counts()
                stroke_df = stroke_counts.reset_index()
                stroke_df.columns = ['Stroke', 'Count']
                
                # Calculate percentages
                total = stroke_df['Count'].sum()
                stroke_df['Percentage'] = (stroke_df['Count'] / total * 100).round(1)
                stroke_df['Percentage'] = stroke_df['Percentage'].astype(str) + '%'
                
                st.dataframe(stroke_df)
                
                # Create pie chart
                fig = self.visualizer.create_pie_chart(
                    stroke_df['Count'].tolist(),
                    stroke_df['Stroke'].tolist(),
                    'Stroke Distribution'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'impact_region' in session_df.columns:
                st.subheader("Impact Region Distribution")
                
                region_counts = session_df['impact_region'].value_counts()
                region_df = region_counts.reset_index()
                region_df.columns = ['Impact Region', 'Count']
                
                # Calculate percentages
                total = region_df['Count'].sum()
                region_df['Percentage'] = (region_df['Count'] / total * 100).round(1)
                region_df['Percentage'] = region_df['Percentage'].astype(str) + '%'
                
                st.dataframe(region_df)
                
                # Create bar chart
                fig = self.visualizer.create_bar_chart(
                    region_df,
                    'Impact Region',
                    'Count',
                    'Impact Region Distribution',
                    color_col='Impact Region'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Session statistics
        st.subheader("Session Statistics")
        
        # Get numerical columns
        numeric_cols = session_df.select_dtypes(include=['float', 'int']).columns.tolist()
        exclude_cols = ['l_id', 'animation_index', 'session_id', 'start_time', 'end_time']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if numeric_cols:
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            
            # Power metric
            power_col = next((col for col in numeric_cols if 'power' in col.lower()), None)
            if power_col:
                with col1:
                    avg_power = session_df[power_col].mean()
                    max_power = session_df[power_col].max()
                    st.metric("Average Power", f"{avg_power:.1f}", f"Max: {max_power:.1f}")
            
            # Speed metric
            speed_col = next((col for col in numeric_cols if 'speed' in col.lower()), None)
            if speed_col:
                with col2:
                    avg_speed = session_df[speed_col].mean()
                    max_speed = session_df[speed_col].max()
                    st.metric("Average Speed", f"{avg_speed:.1f}", f"Max: {max_speed:.1f}")
            
            # Spin metric
            spin_col = next((col for col in numeric_cols if 'spin' in col.lower()), None)
            if spin_col:
                with col3:
                    avg_spin = session_df[spin_col].mean()
                    st.metric("Average Spin", f"{avg_spin:.1f}")
            
            # Statistics table
            st.dataframe(session_df[numeric_cols].describe().round(2))
        
        # Session timeline
        if 'l_id' in session_df.columns and pd.api.types.is_datetime64_any_dtype(session_df['l_id']):
            st.subheader("Session Timeline")
            
            # Create timeline
            fig = self.visualizer.create_time_series(
                session_df,
                'l_id',
                [numeric_cols[0]] if numeric_cols else [],
                'Shot Timing'
            )
            
            st.plotly_chart(fig, use_container_width=True)
