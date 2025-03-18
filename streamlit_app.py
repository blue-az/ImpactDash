#!/usr/bin/env python3
import streamlit as st
import argparse
import sys
from pathlib import Path
from config import Config
from dashboards.impact_explorer import ImpactExplorer
from dashboards.zepp_dashboard import ZeppDashboard

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Tennis Analytics Dashboard")
    
    parser.add_argument(
        "--dashboard", 
        choices=["impact", "zepp"], 
        default="impact",
        help="Dashboard type to launch"
    )
    
    parser.add_argument(
        "--db-path",
        help="Path to database file"
    )
    
    parser.add_argument(
        "--rotation",
        type=float,
        help="Rotation angle for impact positions (in degrees)"
    )
    
    # Parse known args only to avoid conflicts with Streamlit's own args
    return parser.parse_known_args()[0]

def main():
    """Main entry point for Streamlit app"""
    # Parse command-line arguments
    args = parse_args()
    
    # Create config with customizations
    config = Config()
    
    if args.db_path:
        config.DB_PATH = Path(args.db_path)
        
    if args.rotation:
        config.ROTATION_ANGLE = args.rotation
    
    # Launch the selected dashboard based on args or query parameters
    dashboard_param = st.query_params.get("dashboard", args.dashboard)
    
    if dashboard_param == "zepp":
        dashboard = ZeppDashboard(config)
    else:
        dashboard = ImpactExplorer(config)
    
    dashboard.run()

if __name__ == "__main__":
    main()
