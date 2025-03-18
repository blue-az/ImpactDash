import streamlit as st
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

class BaseDashboard:
    """Base class for tennis analytics dashboards"""
    
    def __init__(self, config, title: str = "Tennis Analytics Dashboard"):
        """
        Initialize the dashboard
        
        Args:
            config: Configuration object
            title: Dashboard title
        """
        self.config = config
        self.title = title
        
        # Configure Streamlit page
        st.set_page_config(
            page_title=self.title,
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Set up session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for persistent data across reruns"""
        if 'df' not in st.session_state:
            st.session_state.df = pd.DataFrame()
            
        if 'sessions' not in st.session_state:
            st.session_state.sessions = {}
            
        if 'metrics' not in st.session_state:
            st.session_state.metrics = {}
            
        if 'current_session' not in st.session_state:
            st.session_state.current_session = None
            
        if 'current_frame' not in st.session_state:
            st.session_state.current_frame = 0
    
    def run(self):
        """Run the dashboard application"""
        st.title(self.title)
        
        try:
            # Create sidebar
            with st.sidebar:
                self._create_sidebar()
                
            # Display main content
            self._display_main_content()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    
    def _create_sidebar(self):
        """Create sidebar - to be implemented by subclasses"""
        st.header("Settings")
        st.warning("This is a base dashboard. Please use a specific dashboard class.")
    
    def _display_main_content(self):
        """Display main content - to be implemented by subclasses"""
        st.info("This is a base dashboard. Please use a specific dashboard class.")
    
    def _load_data(self, db_path: Union[str, Path]):
        """
        Load data from database - to be implemented by subclasses
        
        Args:
            db_path: Path to database file
        """
        pass
    
    def _display_data_summary(self, df: pd.DataFrame):
        """
        Display data summary information
        
        Args:
            df: DataFrame with data to summarize
        """
        if df.empty:
            st.info("No data available for summary.")
            return
            
        # Basic information
        st.write(f"Total records: {len(df)}")
        
        # Date range
        date_col = next((col for col in ['date', 'l_id', 'session_date'] 
                        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col])), 
                        None)
        
        if date_col:
            min_date = df[date_col].min()
            max_date = df[date_col].max()
            
            if hasattr(min_date, 'strftime') and hasattr(max_date, 'strftime'):
                st.write(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        
        # Numeric columns for statistics
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            st.subheader("Descriptive Statistics")
            stats_df = df[numeric_cols].describe().round(2)
            st.dataframe(stats_df)
    
    def _create_filter_controls(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        stroke_col: str = 'stroke',
        region_col: str = 'impact_region',
        default_all: bool = True
    ) -> Dict[str, Any]:
        """
        Create common filter controls
        
        Args:
            df: DataFrame with data to filter
            date_col: Column containing date information
            stroke_col: Column containing stroke information
            region_col: Column containing impact region information
            default_all: Whether to select all options by default
            
        Returns:
            Dictionary with filter selections
        """
        filters = {}
        
        # Date range
        min_date = date.today()
        max_date = date.today()
        
        if date_col in df.columns and not df.empty:
            if pd.api.types.is_datetime64_dtype(df[date_col]):
                min_date = df[date_col].min().date()
                max_date = df[date_col].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            filters['start_date'] = st.date_input(
                "Start Date", 
                min_date,
                min_value=min_date,
                max_value=max_date
            )
        
        with col2:
            filters['end_date'] = st.date_input(
                "End Date", 
                max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        # Stroke selection
        all_strokes = []
        if stroke_col in df.columns and not df.empty:
            all_strokes = sorted(df[stroke_col].unique().tolist())
            
        filters['strokes'] = st.multiselect(
            "Select Strokes",
            all_strokes,
            default=all_strokes if default_all else []
        )
        
        # Impact region selection
        all_regions = []
        if region_col in df.columns and not df.empty:
            all_regions = sorted(df[region_col].unique().tolist())
            
        filters['regions'] = st.multiselect(
            "Select Impact Regions",
            all_regions,
            default=all_regions if default_all else []
        )
        
        return filters
