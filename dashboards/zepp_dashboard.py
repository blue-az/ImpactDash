import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from .base_dashboard import BaseDashboard
from core.data_manager import DataManager
from core.session_manager import SessionManager
from core.utils import (
    calculate_shot_breakdown, safe_division, format_duration,
    get_standard_features
)
from visualization.visualizer import Visualizer
from visualization.stats_visualizer import StatsVisualizer

class ZeppDashboard(BaseDashboard):
    """Dashboard for Zepp tennis data analysis"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        super().__init__(config, "Zepp Tennis Analysis Dashboard")
        
        # Initialize components
        self.data_manager = DataManager(config)
        self.session_manager = SessionManager(config)
        self.visualizer = Visualizer(config)
        self.stats_visualizer = StatsVisualizer(config)
    
    def _create_sidebar(self):
        """Create sidebar with session selector and settings"""
        st.header("Data Source")
        db_path = st.text_input("Database File Path", value=str(self.config.DB_PATH))
        
        # Load sessions data if needed
        if 'sessions_df' not in st.session_state and db_path:
            with st.spinner("Loading sessions data..."):
                try:
                    st.session_state.sessions_df = self._load_sessions_data(db_path)
                except Exception as e:
                    st.error(f"Error loading sessions data: {str(e)}")
                    st.session_state.sessions_df = pd.DataFrame()
        
        # Session selector
        if 'sessions_df' in st.session_state and not st.session_state.sessions_df.empty:
            selected_id = st.selectbox(
                "Select Session ID", 
                st.session_state.sessions_df['_id'].tolist(),
                key="session_selector"
            )
            
            # Load selected session data
            selected_data = st.session_state.sessions_df[
                st.session_state.sessions_df['_id'] == selected_id
            ].iloc[0]
            
            # Store in session state
            if 'selected_session' not in st.session_state or st.session_state.selected_session != selected_id:
                st.session_state.selected_session = selected_id
                st.session_state.selected_data = selected_data
                
                # Extract session report data
                if 'report' in selected_data and not pd.isna(selected_data['report']):
                    session_json = self.data_manager.extract_json_data(selected_data['report'])
                    st.session_state.session_json = session_json
                else:
                    st.session_state.session_json = {}
            
            # Display session info in sidebar
            self._display_session_sidebar_info(st.session_state.selected_data)
        
        # Analysis view selector
        st.sidebar.header("Analysis View")
        st.session_state.analysis_view = st.sidebar.selectbox(
            "Select Analysis View",
            ["Session Analysis", "Shot Analysis", "Hit Points Analysis", "Historical Analysis"],
            key="analysis_view_selector"
        )
    
    def _display_main_content(self):
        """Display main dashboard content"""
        if 'sessions_df' not in st.session_state or st.session_state.sessions_df.empty:
            st.info("No sessions data available. Please load data using the sidebar controls.")
            return
            
        if 'selected_session' not in st.session_state:
            st.info("Please select a session from the sidebar.")
            return
        
        # Display content based on selected view
        view = st.session_state.analysis_view
        
        if view == "Session Analysis":
            self._display_session_analysis()
        elif view == "Shot Analysis":
            self._display_shot_analysis()
        elif view == "Hit Points Analysis":
            self._display_hit_points_analysis()
        elif view == "Historical Analysis":
            self._display_historical_analysis()
    
    def _load_sessions_data(self, db_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load session data from database
        
        Args:
            db_path: Path to database file
            
        Returns:
            DataFrame with sessions data
        """
        try:
            # Connect to database
            conn = self.data_manager.load_data(db_path)
            
            # Try to find session_report table
            sessions_df = pd.read_sql_query(
                "SELECT _id, session_id, start_time, end_time, session_score, report FROM session_report",
                conn
            )
            
            # Process timestamps
            if 'start_time' in sessions_df.columns:
                sessions_df['start_time'] = pd.to_datetime(sessions_df['start_time'], unit='ms')
                sessions_df['start_datetime'] = sessions_df['start_time']
                
                # Convert to local timezone
                sessions_df['start_datetime'] = sessions_df['start_datetime'].dt.tz_localize('UTC').dt.tz_convert(self.config.TIMEZONE)
                
                # Format for display
                sessions_df['formatted_date'] = sessions_df['start_datetime'].dt.strftime('%m-%d-%Y')
                sessions_df['formatted_time'] = sessions_df['start_datetime'].dt.strftime('%I:%M %p')
                sessions_df['formatted_datetime'] = sessions_df['start_datetime'].dt.strftime('%m-%d-%Y %I:%M %p')
            
            if 'end_time' in sessions_df.columns:
                sessions_df['end_time'] = pd.to_datetime(sessions_df['end_time'], unit='ms')
                sessions_df['end_datetime'] = sessions_df['end_time']
                
                # Calculate duration
                if 'start_time' in sessions_df.columns:
                    sessions_df['duration_ms'] = sessions_df['end_time'] - sessions_df['start_time']
                    sessions_df['duration_min'] = sessions_df['duration_ms'] / pd.Timedelta(minutes=1)
            
            return sessions_df
            
        except Exception as e:
            st.error(f"Error loading sessions data: {str(e)}")
            return pd.DataFrame()
    
    def _display_session_sidebar_info(self, session_data: pd.Series):
        """
        Display session information in sidebar
        
        Args:
            session_data: Series with session data
        """
        try:
            # Display session date and time
            if 'formatted_date' in session_data:
                st.sidebar.markdown(f"**Date:** {session_data['formatted_date']}")
                
            if 'formatted_time' in session_data:
                st.sidebar.markdown(f"**Start Time:** {session_data['formatted_time']}")
                
            # Display duration
            if 'start_time' in session_data and 'end_time' in session_data:
                duration = format_duration(
                    int((session_data['end_time'] - session_data['start_time']).total_seconds() * 1000)
                )
                st.sidebar.markdown(f"**Duration:** {duration}")
            
            # Display session score
            if 'session_score' in session_data:
                st.sidebar.metric("Session Score", f"{session_data['session_score']:.2f}")
            
            # Display total shots if available
            if 'session_json' in st.session_state:
                shot_stats = calculate_shot_breakdown(st.session_state.session_json)
                if shot_stats:
                    st.sidebar.metric("Total Shots", shot_stats['Total'])
                    
                    # Show shot distribution
                    fig = self.visualizer.create_stacked_bar(
                        {k: v for k, v in shot_stats.items() if k in ['Serve', 'Forehand', 'Backhand']},
                        "",
                        show_percentages=True,
                        orientation='h'
                    )
                    
                    st.sidebar.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.sidebar.warning(f"Error displaying session info: {str(e)}")
    
    def _display_session_analysis(self):
        """Display session analysis view"""
        if 'session_json' not in st.session_state or not st.session_state.session_json:
            st.warning("No session data available.")
            return
            
        st.header("Session Performance")
        
        session_json = st.session_state.session_json
        
        if 'session' in session_json and 'profile_snapshot' in session_json:
            session_data = session_json['session']
            profile_data = session_json['profile_snapshot']
            total_sessions = profile_data['swings'].get('total_sessions', 0)
            
            # Performance comparison section
            st.subheader("Session Performance vs Overall Stats")
            
            col1, col2, col3 = st.columns(3)
            
            # Serve comparison
            with col1:
                st.subheader("Serve Comparison")
                
                serve_stats = {
                    'session': {
                        'avg': session_data['swings']['serve']['serve_average_speed'],
                        'max': session_data['swings']['serve']['serve_max_speed']
                    },
                    'overall': {
                        'avg': profile_data['swings']['serve']['serve_average_speed'],
                        'max': profile_data['personal_bests']['fastest_serve']
                    }
                }
                
                st.metric(
                    "Average Serve Speed", 
                    f"{serve_stats['session']['avg']:.1f} mph",
                    f"{serve_stats['session']['avg'] - serve_stats['overall']['avg']:.1f} mph"
                )
                
                st.metric(
                    "Max Serve Speed", 
                    f"{serve_stats['session']['max']:.1f} mph",
                    f"{serve_stats['session']['max'] - serve_stats['overall']['max']:.1f} mph"
                )
            
            # Score comparison
            with col2:
                st.subheader("Session Scores")
                
                session_scores = session_data['swings']['scores']
                overall_scores = profile_data['swings']['scores']
                
                for score_type in ['consistency', 'power']:
                    current = session_scores[f'{score_type}_score']
                    avg = safe_division(
                        overall_scores[f'total_{score_type}_score'], 
                        total_sessions
                    )
                    
                    st.metric(
                        f"{score_type.title()} Score", 
                        f"{current:.1f}",
                        f"{current - avg:.1f}" if total_sessions > 0 else None
                    )
            
            # Activity metrics
            with col3:
                st.subheader("Activity Metrics")
                
                session_active_time = session_data['active_time'] / 60
                avg_active_time = safe_division(
                    profile_data['swings']['total_active_time'], 
                    total_sessions * 60
                )
                
                st.metric(
                    "Active Time", 
                    f"{session_active_time:.1f} min",
                    f"{session_active_time - avg_active_time:.1f} min" if total_sessions > 0 else None
                )
                
                rally_delta = (
                    session_data['longest_rally_swings'] - 
                    profile_data['personal_bests']['longest_rally_swings']
                )
                
                st.metric(
                    "Rally Length", 
                    str(session_data['longest_rally_swings']),
                    str(rally_delta) if rally_delta != 0 else None
                )
            
            # Session scores radar chart
            st.subheader("Session Performance Breakdown")
            
            session_scores = session_data['swings']['scores']
            
            # Map the correct score keys
            score_mapping = {
                'Consistency': 'consistency_score',
                'Intensity': 'intensity_score',
                'Power': 'power_score',
                'Overall': 'session_score'
            }
            
            # Extract scores
            radar_metrics = {
                metric: session_scores[score_key] 
                for metric, score_key in score_mapping.items() 
                if score_key in session_scores
            }
            
            # Create radar chart
            radar_fig = self.visualizer.create_radar_chart(radar_metrics)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        else:
            st.warning("Session data format not recognized.")
    
    def _display_shot_analysis(self):
        """Display shot analysis view"""
        if 'session_json' not in st.session_state or not st.session_state.session_json:
            st.warning("No session data available.")
            return
            
        st.header("Shot Analysis")
        
        session_json = st.session_state.session_json
        
        if 'session' in session_json:
            session_data = session_json['session']
            
            # Shot type selector
            shot_type = st.selectbox(
                "Select Shot Type", 
                ["serve", "forehand", "backhand"]
            )
            
            if shot_type in session_data['swings']:
                # Process data based on shot type
                if shot_type == "serve":
                    serve_data = session_data['swings']['serve']
                    
                    # Create dataframe for visualization
                    stroke_df = pd.DataFrame({
                        'Swing Type': ['Serve'],
                        'Average Speed': [serve_data['serve_average_speed']],
                        'Max Speed': [serve_data['serve_max_speed']],
                        'Sweet Spot Hits': [serve_data['serve_sweet_swings']],
                        'Total Swings': [serve_data['serve_swings']]
                    })
                    
                    # Display charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Speed comparison chart
                        fig_speed = self.visualizer.create_bar_chart(
                            stroke_df, 
                            'Swing Type', 
                            ['Average Speed', 'Max Speed'],
                            f"{shot_type.capitalize()} Speed Comparison", 
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig_speed, use_container_width=True)
                    
                    with col2:
                        # Distribution pie chart
                        fig_dist = self.visualizer.create_pie_chart(
                            [serve_data['serve_swings']],
                            ['Serve'],
                            f"{shot_type.capitalize()} Distribution"
                        )
                        
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Sweet spot analysis
                    fig_sweet = self.visualizer.create_bar_chart(
                        stroke_df,
                        'Swing Type',
                        ['Sweet Spot Hits', 'Total Swings'],
                        f"{shot_type.capitalize()} Sweet Spot vs Total Swings",
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig_sweet, use_container_width=True)
                    
                else:  # forehand or backhand
                    # Get stroke data
                    stroke_data = session_data['swings'][shot_type]
                    
                    # Create dataframe for visualization
                    stroke_df = pd.DataFrame({
                        'Swing Type': ['Flat', 'Slice', 'Topspin'],
                        'Average Speed': [
                            stroke_data[f'{spin}_average_speed'] 
                            for spin in ['flat', 'slice', 'topspin']
                        ],
                        'Max Speed': [
                            stroke_data[f'{spin}_max_speed'] 
                            for spin in ['flat', 'slice', 'topspin']
                        ],
                        'Sweet Spot Hits': [
                            stroke_data[f'{spin}_sweet_swings'] 
                            for spin in ['flat', 'slice', 'topspin']
                        ],
                        'Total Swings': [
                            stroke_data[f'{spin}_swings'] 
                            for spin in ['flat', 'slice', 'topspin']
                        ]
                    })
                    
                    # Display charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Speed comparison chart
                        fig_speed = self.visualizer.create_bar_chart(
                            stroke_df, 
                            'Swing Type', 
                            ['Average Speed', 'Max Speed'],
                            f"{shot_type.capitalize()} Speed Comparison", 
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig_speed, use_container_width=True)
                    
                    with col2:
                        # Distribution pie chart
                        fig_dist = self.visualizer.create_pie_chart(
                            stroke_df['Total Swings'].tolist(),
                            stroke_df['Swing Type'].tolist(),
                            f"{shot_type.capitalize()} Distribution"
                        )
                        
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Sweet spot analysis
                    fig_sweet = self.visualizer.create_bar_chart(
                        stroke_df,
                        'Swing Type',
                        ['Sweet Spot Hits', 'Total Swings'],
                        f"{shot_type.capitalize()} Sweet Spot vs Total Swings",
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig_sweet, use_container_width=True)
            else:
                st.info(f"No data available for {shot_type}.")
        else:
            st.warning("Session data format not recognized.")
    
    def _display_hit_points_analysis(self):
        """Display hit points analysis view"""
        if 'session_json' not in st.session_state or not st.session_state.session_json:
            st.warning("No session data available.")
            return
            
        st.header("Hit Points Analysis")
        
        session_json = st.session_state.session_json
        
        if 'session' in session_json:
            session_data = session_json['session']
            
            # Shot type selector
            shot_type = st.selectbox(
                "Select Shot Type",
                ["forehand", "backhand", "serve"]
            )
            
            if shot_type in ["forehand", "backhand"]:
                # Spin type selector for forehand/backhand
                spin_type = st.selectbox(
                    "Select Spin Type",
                    ["flat", "slice", "topspin"]
                )
                
                # Extract hit points
                hit_points = session_data['swings'][shot_type].get(f'{spin_type}_hit_points', [])
                title = f"Hit Points Distribution - {shot_type.capitalize()} {spin_type.capitalize()}"
            else:
                # Serve hit points
                hit_points = session_data['swings']['serve'].get('serve_hit_points', [])
                title = f"Hit Points Distribution - {shot_type.capitalize()}"
            
            if hit_points:
                # Convert to DataFrame for visualization
                df_hits = pd.DataFrame(hit_points, columns=['x', 'y'])
                
                # Create scatter plot
                fig = px.scatter(
                    df_hits,
                    x='x',
                    y='y',
                    title=title,
                    labels={'x': 'Horizontal Position', 'y': 'Vertical Position'}
                )
                
                # Add reference grid
                fig.update_layout(
                    shapes=[
                        # Racket outline
                        dict(
                            type="rect", 
                            x0=-1, y0=-1, 
                            x1=1, y1=1,
                            line=dict(color="rgba(0,0,0,0.3)"), 
                            fillcolor="rgba(0,0,0,0)"
                        ),
                        # Horizontal center line
                        dict(
                            type="line", 
                            x0=-1, y0=0, 
                            x1=1, y1=0,
                            line=dict(color="rgba(0,0,0,0.3)", dash="dash")
                        ),
                        # Vertical center line
                        dict(
                            type="line", 
                            x0=0, y0=-1, 
                            x1=0, y1=1,
                            line=dict(color="rgba(0,0,0,0.3)", dash="dash")
                        )
                    ]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display stats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Hit Points", len(hit_points))
                
                with col2:
                    avg_height = sum(point[1] for point in hit_points) / len(hit_points)
                    st.metric("Average Height", f"{avg_height:.2f}")
                
                # Advanced analysis
                if len(hit_points) > 1:
                    st.subheader("Advanced Hit Point Analysis")
                    
                    # Calculate centroid
                    centroid_x = sum(point[0] for point in hit_points) / len(hit_points)
                    centroid_y = sum(point[1] for point in hit_points) / len(hit_points)
                    
                    # Calculate dispersion
                    dispersion_x = np.std([point[0] for point in hit_points])
                    dispersion_y = np.std([point[1] for point in hit_points])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Centroid X", f"{centroid_x:.2f}")
                        st.metric("Dispersion X", f"{dispersion_x:.2f}")
                    
                    with col2:
                        st.metric("Centroid Y", f"{centroid_y:.2f}")
                        st.metric("Dispersion Y", f"{dispersion_y:.2f}")
                    
                    # Calculate sweet spot percentage
                    sweet_spot_hits = sum(1 for point in hit_points if (point[0]**2 + point[1]**2) <= 0.3**2)
                    sweet_spot_percentage = (sweet_spot_hits / len(hit_points)) * 100
                    
                    st.metric(
                        "Sweet Spot Hits", 
                        f"{sweet_spot_hits} ({sweet_spot_percentage:.1f}%)"
                    )
            else:
                st.info(f"No hit points data available for {shot_type}.")
        else:
            st.warning("Session data format not recognized.")
    
    def _display_historical_analysis(self):
        """Display historical analysis view"""
        if 'sessions_df' not in st.session_state or st.session_state.sessions_df.empty:
            st.warning("No historical data available.")
            return
        
        st.header("Historical Performance Analysis")
        
        # Process historical data
        historical_df = self._process_historical_data(st.session_state.sessions_df)
        
        if historical_df.empty:
            st.error("No historical data available for analysis.")
            return
            
        # Sort by date
        historical_df = historical_df.sort_values('date')
        
        # Metrics selection
        metric_options = {
            'Session Score': 'session_score',
            'Total Shots': 'total_shots',
            'Serve Average Speed': 'serve_avg_speed',
            'Consistency Score': 'consistency_score',
            'Power Score': 'power_score',
            'Intensity Score': 'intensity_score'
        }
        
        selected_metric = st.selectbox(
            "Select Metric to Analyze", 
            list(metric_options.keys())
        )
        
        metric_col = metric_options[selected_metric]
        
        # Calculate rolling averages
        window_size = min(5, len(historical_df))
        historical_df['rolling_metric'] = historical_df[metric_col].rolling(window=window_size).mean()
        
        # Create time series plot
        fig = self.stats_visualizer.create_time_series(
            historical_df,
            'date',
            [metric_col],
            f"{selected_metric} Over Time",
            show_trend=True,
            rolling_window=window_size
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        overall_avg = historical_df[metric_col].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Average", f"{overall_avg:.1f}")
        
        with col2:
            st.metric("Best Session", f"{historical_df[metric_col].max():.1f}")
        
        with col3:
            st.metric(
                "Recent Trend", 
                f"{historical_df[metric_col].iloc[-1]:.1f}",
                f"{historical_df[metric_col].iloc[-1] - overall_avg:.1f}"
            )
        
        # Monthly averages
        st.subheader("Monthly Averages")
        
        fig = self.stats_visualizer.create_monthly_averages_chart(
            historical_df,
            'date',
            [metric_col]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _process_historical_data(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process session data for historical analysis
        
        Args:
            sessions_df: DataFrame with session data
            
        Returns:
            Processed DataFrame for historical analysis
        """
        historical_data = []
        
        for _, row in sessions_df.iterrows():
            try:
                if pd.isna(row['report']) or not row['report']:
                    continue
                    
                session_json = json.loads(row['report'])
                
                if not all(key in session_json for key in ['session', 'profile_snapshot']):
                    continue
                    
                session_data = session_json['session']
                
                # Extract key metrics
                metrics = {
                    'session_id': row['_id'],
                    'date': pd.to_datetime(row['start_time'], unit='ms'),
                    'session_score': row['session_score'],
                    'total_shots': sum(
                        session_data['swings'][shot_type].get(f'{spin}_swings', 0)
                        for shot_type in ['forehand', 'backhand']
                        for spin in ['flat', 'slice', 'topspin']
                    ) + session_data['swings']['serve'].get('serve_swings', 0),
                    'serve_avg_speed': session_data['swings']['serve'].get('serve_average_speed', 0),
                    'consistency_score': session_data['swings']['scores'].get('consistency_score', 0),
                    'power_score': session_data['swings']['scores'].get('power_score', 0),
                    'intensity_score': session_data['swings']['scores'].get('intensity_score', 0)
                }
                
                historical_data.append(metrics)
                
            except Exception:
                continue
                
        return pd.DataFrame(historical_data)
