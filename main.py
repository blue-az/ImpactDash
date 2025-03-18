#!/usr/bin/env python3
import argparse
from config import Config
from dashboards.impact_explorer import ImpactExplorer
from dashboards.zepp_dashboard import ZeppDashboard

def main():
    """Main entry point for command line usage"""
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
    
    args = parser.parse_args()
    
    # Create config with customizations
    config = Config()
    
    if args.db_path:
        config.DB_PATH = args.db_path
        
    if args.rotation:
        config.ROTATION_ANGLE = args.rotation
    
    # Launch the selected dashboard
    if args.dashboard == "impact":
        dashboard = ImpactExplorer(config)
    else:
        dashboard = ZeppDashboard(config)
    
    dashboard.run()

if __name__ == "__main__":
    main()
