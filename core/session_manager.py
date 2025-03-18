import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

class SessionManager:
    """Handles session detection, analysis and management"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.sessions = {}
        self.current_session_id = None
    
    def detect_sessions(self, df: pd.DataFrame, time_gap: timedelta = timedelta(minutes=30)) -> Dict[str, pd.DataFrame]:
        """
        Detect sessions in a dataframe based on time gaps
        
        Args:
            df: DataFrame with timestamp information
            time_gap: Minimum time gap between sessions
            
        Returns:
            Dictionary of session dataframes
        """
        if df.empty:
            return {}
            
        # Find timestamp column
        time_col = next((col for col in ['l_id', 'timestamp', 'date'] 
                        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col])), 
                        None)
        
        if time_col is None:
            # Try to create sessions by date
            if 'session_date' in df.columns:
                return self._group_by_date(df, 'session_date')
            else:
                # Create artificial sessions of 50 shots each
                return self._create_artificial_sessions(df)
        
        # Sort by timestamp
        sorted_df = df.sort_values(time_col).copy()
        
        # Calculate time differences between consecutive rows
        sorted_df['time_diff'] = sorted_df[time_col].diff()
        
        # Find session breaks (rows where time difference exceeds the gap)
        session_breaks = sorted_df[sorted_df['time_diff'] > time_gap].index.tolist()
        
        # Start new session at beginning and after each break
        session_starts = [sorted_df.index[0]] + session_breaks
        
        # Create sessions dictionary
        sessions = {}
        
        for i, start_idx in enumerate(session_starts):
            # End of this session is start of next session or end of dataframe
            end_idx = session_starts[i + 1] if i < len(session_starts) - 1 else sorted_df.index[-1]
            
            # Get session data
            session_df = sorted_df.loc[start_idx:end_idx].copy()
            
            # Generate session ID
            if time_col in session_df.columns:
                # Use date of first row in session
                session_start = session_df.iloc[0][time_col]
                if hasattr(session_start, 'strftime'):
                    session_id = session_start.strftime('%Y-%m-%d %H:%M')
                else:
                    session_id = f"Session_{i+1}"
            else:
                session_id = f"Session_{i+1}"
            
            # Add session to dictionary
            sessions[session_id] = session_df
        
        return sessions
    
    def _group_by_date(self, df: pd.DataFrame, date_col: str) -> Dict[str, pd.DataFrame]:
        """Group data by date"""
        sessions = {}
        
        for date_val, group in df.groupby(date_col):
            session_id = str(date_val)
            if hasattr(date_val, 'strftime'):
                session_id = date_val.strftime('%Y-%m-%d')
            
            sessions[session_id] = group.copy()
        
        return sessions
    
    def _create_artificial_sessions(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create artificial sessions of fixed size"""
        sessions = {}
        chunk_size = 50
        
        for i in range(0, len(df), chunk_size):
            end_idx = min(i + chunk_size, len(df))
            sessions[f"Session_{i//chunk_size + 1}"] = df.iloc[i:end_idx].copy()
        
        return sessions
    
    def calculate_session_stats(self, session_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistics for a session
        
        Args:
            session_df: DataFrame for a single session
            
        Returns:
            Dictionary of session statistics
        """
        stats = {
            'total_shots': len(session_df)
        }
        
        # Session date/time
        time_col = next((col for col in ['l_id', 'timestamp', 'date'] 
                         if col in session_df.columns and pd.api.types.is_datetime64_any_dtype(session_df[col])), 
                         None)
        
        if time_col:
            stats['start_time'] = session_df[time_col].min()
            stats['end_time'] = session_df[time_col].max()
            
            # Session duration
            duration_seconds = (stats['end_time'] - stats['start_time']).total_seconds()
            stats['duration_minutes'] = duration_seconds / 60
            
            # Shots per minute
            if stats['duration_minutes'] > 0:
                stats['shots_per_minute'] = stats['total_shots'] / stats['duration_minutes']
            else:
                stats['shots_per_minute'] = 0
        
        # Shot distribution
        if 'stroke' in session_df.columns:
            stroke_counts = session_df['stroke'].value_counts()
            stats['stroke_distribution'] = stroke_counts.to_dict()
            
            for stroke, count in stroke_counts.items():
                stats[f"{stroke}_count"] = count
                stats[f"{stroke}_percentage"] = 100 * count / stats['total_shots']
        
        # Impact region distribution
        if 'impact_region' in session_df.columns:
            region_counts = session_df['impact_region'].value_counts()
            stats['region_distribution'] = region_counts.to_dict()
            
            for region, count in region_counts.items():
                stats[f"region_{region}_count"] = count
                stats[f"region_{region}_percentage"] = 100 * count / stats['total_shots']
        
        # Performance metrics
        for metric in ['power', 'racket_speed', 'ball_spin']:
            if metric in session_df.columns:
                stats[f"avg_{metric}"] = session_df[metric].mean()
                stats[f"max_{metric}"] = session_df[metric].max()
                stats[f"min_{metric}"] = session_df[metric].min()
        
        return stats
    
    def get_session_durations(self, sessions: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate duration for multiple sessions
        
        Args:
            sessions: Dictionary of session dataframes
            
        Returns:
            Dictionary mapping session IDs to durations in minutes
        """
        durations = {}
        
        for session_id, session_df in sessions.items():
            time_col = next((col for col in ['l_id', 'timestamp', 'date'] 
                             if col in session_df.columns and pd.api.types.is_datetime64_any_dtype(session_df[col])), 
                             None)
            
            if time_col:
                start_time = session_df[time_col].min()
                end_time = session_df[time_col].max()
                
                # Calculate duration in minutes
                duration = (end_time - start_time).total_seconds() / 60
                durations[session_id] = duration
            else:
                durations[session_id] = 0
        
        return durations
    
    def set_current_session(self, session_id: str) -> bool:
        """
        Set the current session ID
        
        Args:
            session_id: ID of the session to set as current
            
        Returns:
            True if successful, False otherwise
        """
        if session_id in self.sessions:
            self.current_session_id = session_id
            return True
        return False
    
    def get_current_session(self) -> Optional[pd.DataFrame]:
        """
        Get the current session dataframe
        
        Returns:
            Current session dataframe or None if no session is selected
        """
        if self.current_session_id and self.current_session_id in self.sessions:
            return self.sessions[self.current_session_id]
        return None
