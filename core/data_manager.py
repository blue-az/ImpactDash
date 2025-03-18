import sqlite3
import pandas as pd
import numpy as np
import pytz
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

class DataManager:
    """Core data management functionality for tennis analytics"""
    
    def __init__(self, config):
        """Initialize the data manager with configuration"""
        self.config = config
        self.cache = {}
    
    def load_data(self, db_path: Union[str, Path], table: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from a SQLite database
        
        Args:
            db_path: Path to the SQLite database
            table: Specific table to load (attempts auto-detection if None)
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            # Connect to database
            conn = sqlite3.connect(db_path)
            
            # Find available tables if not specified
            if table is None:
                tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
                tables = pd.read_sql(tables_query, conn)['name'].tolist()
                
                # Try to find a suitable table (swings, session_report, etc.)
                candidate_tables = [t for t in tables if any(keyword in t.lower() 
                                    for keyword in ['swing', 'shot', 'session', 'report'])]
                
                table = candidate_tables[0] if candidate_tables else tables[0] if tables else None
                
                if table is None:
                    raise ValueError("No suitable tables found in the database")
            
            # Construct query
            query = f"SELECT * FROM {table}"
            
            # Read data from database
            df = pd.read_sql(query, conn)
            conn.close()
            
            # Process timestamps if present
            if 'l_id' in df.columns and pd.api.types.is_numeric_dtype(df['l_id']):
                df['l_id'] = pd.to_datetime(df['l_id'], unit='ms')
                df['date'] = df['l_id']
            
            # Convert timestamps to specified timezone
            if 'l_id' in df.columns and pd.api.types.is_datetime64_dtype(df['l_id']):
                try:
                    tz = pytz.timezone(self.config.TIMEZONE)
                    df['l_id'] = df['l_id'].dt.tz_localize('UTC').dt.tz_convert(tz)
                except Exception:
                    # Already localized or invalid timezone
                    pass
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame()

    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw data with standard transformations
        
        Args:
            df: Raw dataframe from database
            
        Returns:
            Processed dataframe
        """
        if df.empty:
            return df
            
        processed_df = df.copy()
        
        # Process impact positions with rotation if columns exist
        if all(col in processed_df.columns for col in ['impact_position_x', 'impact_position_y']):
            self._add_rotated_coordinates(processed_df)
        
        # Add stroke classification if necessary
        if 'stroke' not in processed_df.columns and 'swing_type' in processed_df.columns:
            self._add_stroke_classification(processed_df)
        
        # Add session date for grouping if needed
        if 'session_date' not in processed_df.columns:
            self._add_session_date(processed_df)
        
        # Add sequential index for animation
        if 'animation_index' not in processed_df.columns:
            processed_df['animation_index'] = range(len(processed_df))
        
        return processed_df
    
    def _add_rotated_coordinates(self, df: pd.DataFrame) -> None:
        """Add rotated coordinates to dataframe in-place"""
        df['rotated_x'] = 0.0
        df['rotated_y'] = 0.0
        df['modified_rotated_y'] = 0.0
        
        for idx, row in df.iterrows():
            try:
                # Apply rotation
                angle_rad = np.radians(self.config.ROTATION_ANGLE)
                x, y = row['impact_position_x'], row['impact_position_y']
                
                rx = x * np.cos(angle_rad) - y * np.sin(angle_rad)
                ry = x * np.sin(angle_rad) + y * np.cos(angle_rad)
                
                df.at[idx, 'rotated_x'] = rx
                df.at[idx, 'rotated_y'] = ry
                
                # Create modified y value with absolute value for region 2
                if 'impact_region' in df.columns and row['impact_region'] == 2:
                    df.at[idx, 'modified_rotated_y'] = abs(ry)
                else:
                    df.at[idx, 'modified_rotated_y'] = ry
            except Exception:
                # Fallback to original values
                df.at[idx, 'rotated_x'] = row['impact_position_x']
                df.at[idx, 'rotated_y'] = row['impact_position_y']
                df.at[idx, 'modified_rotated_y'] = row['impact_position_y']
    
    def _add_stroke_classification(self, df: pd.DataFrame) -> None:
        """Add stroke classification to dataframe in-place"""
        # Map hand type (if present)
        if 'swing_side' in df.columns and 'hand_type' not in df.columns:
            hand_type = {1: "BH", 0: "FH"}
            df['hand_type'] = df['swing_side'].replace(hand_type)
        
        # Map swing type (if present)
        if 'swing_type' in df.columns and not df['swing_type'].apply(lambda x: isinstance(x, str)).all():
            swing_type = {
                4: "VOLLEY", 3: "SERVE", 2: "TOPSPIN", 
                0: "SLICE", 1: "FLAT", 5: "SMASH"
            }
            df['swing_type'] = df['swing_type'].replace(swing_type)
        
        # Combine swing type and hand type to create stroke
        if 'swing_type' in df.columns and 'hand_type' in df.columns:
            df['stroke'] = df['swing_type'] + df['hand_type']
    
    def _add_session_date(self, df: pd.DataFrame) -> None:
        """Add session date to dataframe in-place"""
        # Try to add from date/time columns
        for col in ['l_id', 'date', 'timestamp']:
            if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
                df['session_date'] = df[col].dt.date
                return
        
        # Fallback to today
        df['session_date'] = datetime.now().date()

    def filter_data(self, 
                    df: pd.DataFrame, 
                    start_date: Optional[date] = None,
                    end_date: Optional[date] = None,
                    strokes: Optional[List[str]] = None,
                    impact_regions: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Filter data based on various criteria
        
        Args:
            df: DataFrame to filter
            start_date: Filter data on or after this date
            end_date: Filter data on or before this date
            strokes: List of stroke types to include
            impact_regions: List of impact regions to include
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        filtered_df = df.copy()
        
        # Date filtering
        if start_date and end_date:
            date_col = next((col for col in ['date', 'l_id', 'session_date'] 
                           if col in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df[col])), 
                           None)
            if date_col:
                filtered_df = filtered_df[
                    (filtered_df[date_col].dt.date >= start_date) &
                    (filtered_df[date_col].dt.date <= end_date)
                ]
        
        # Stroke filtering
        if strokes and 'stroke' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['stroke'].isin(strokes)]
        
        # Impact region filtering
        if impact_regions and 'impact_region' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['impact_region'].isin(impact_regions)]
        
        return filtered_df
    
    def group_into_sessions(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group data into sessions for analysis and visualization
        
        Args:
            df: Processed dataframe
            
        Returns:
            Dictionary mapping session keys to session dataframes
        """
        if df.empty:
            return {}
        
        sessions = {}
        
        # Find suitable date column for grouping
        date_col = next((col for col in ['session_date', 'date', 'l_id'] 
                         if col in df.columns), None)
        
        if date_col and date_col in df.columns:
            # Group by date
            for group_key, group_df in df.groupby(date_col):
                session_key = str(group_key)
                if hasattr(group_key, 'strftime'):
                    session_key = group_key.strftime('%Y-%m-%d')
                
                # Store sorted group
                sessions[session_key] = group_df.sort_values(date_col).copy()
        else:
            # Fallback to grouping by chunks of 50 shots
            chunk_size = 50
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size].copy()
                sessions[f"Session_{i//chunk_size + 1}"] = chunk
        
        return sessions
    
    def get_session_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate session metrics for summary statistics
        
        Args:
            df: Session dataframe
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if df.empty:
            return metrics
        
        # Date range
        date_col = next((col for col in ['date', 'l_id', 'session_date'] 
                        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col])), 
                        None)
        
        if date_col:
            metrics['min_date'] = df[date_col].min().date()
            metrics['max_date'] = df[date_col].max().date()
            
            if 'session_date' not in df.columns:
                df['session_date'] = df[date_col].dt.date
        
        # Unique values for filtering
        if 'stroke' in df.columns:
            metrics['unique_strokes'] = sorted(df['stroke'].unique().tolist())
        
        if 'impact_region' in df.columns:
            metrics['unique_impact_regions'] = sorted(df['impact_region'].unique().tolist())
        
        # Shot distribution
        if 'stroke' in df.columns:
            stroke_counts = df['stroke'].value_counts().to_dict()
            metrics['stroke_counts'] = stroke_counts
            metrics['total_shots'] = sum(stroke_counts.values())
        
        # Calculate speeds if available
        for col in ['power', 'racket_speed', 'ball_spin']:
            if col in df.columns:
                metrics[f'avg_{col}'] = df[col].mean()
                metrics[f'max_{col}'] = df[col].max()
        
        return metrics
    
    def extract_json_data(self, json_str: str) -> Dict[str, Any]:
        """
        Extract and parse JSON data from string
        
        Args:
            json_str: JSON string to parse
            
        Returns:
            Parsed JSON data as dictionary
        """
        try:
            if pd.isna(json_str) or not json_str:
                return {}
            
            return json.loads(json_str)
        except Exception:
            return {}
    
    def add_jitter(self, df: pd.DataFrame, columns: List[str], amount: float = 0.05) -> pd.DataFrame:
        """
        Add jitter to specified columns for better visualization
        
        Args:
            df: DataFrame to modify
            columns: Columns to add jitter to
            amount: Amount of jitter to add
            
        Returns:
            DataFrame with jitter added
        """
        if df.empty:
            return df
            
        result = df.copy()
        
        for col in columns:
            if col in result.columns:
                result[col] = result[col] + np.random.normal(0, amount, len(result))
        
        return result
