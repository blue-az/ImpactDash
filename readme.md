# Tennis Analytics Dashboard

A comprehensive tennis data visualization and analysis platform, providing powerful tools for analyzing racket impact positions, shot metrics, and performance trends.

## Features

### Impact Position Explorer
- Visualize tennis racket impact positions in 2D and 3D
- Analyze impact patterns by stroke type, impact region, and other metrics
- Animate session data to see impact positions evolve over time
- Compare impact positions across different sessions and time periods

### Zepp Tennis Analysis
- Comprehensive analysis of Zepp tennis sensor data
- Session-level performance metrics and shot distribution
- Shot-by-shot analysis with detailed breakdowns by stroke and spin type
- Historical trend analysis to track improvement over time

### Core Features
- Interactive visualizations with customizable settings
- Filter data by date range, stroke type, and impact region
- Session detection and management
- Statistical analysis and performance metrics
- Shot distribution and impact region breakdowns
- Historical trend analysis with rolling averages and trendlines

## Architecture

The application is built with a modular, maintainable architecture:

- **Core**: Data management, session handling, and utility functions
- **Visualization**: Reusable visualization components for different data types
- **Dashboards**: User interfaces built on top of the core and visualization layers
- **Configuration**: Centralized settings management

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tennis-analytics
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Streamlit App

Launch the dashboard using Streamlit:

```bash
streamlit run streamlit_app.py
```

To specify dashboard type and options:

```bash
streamlit run streamlit_app.py -- --dashboard zepp --db-path /path/to/database.db
```

### Command Line

Run from the command line for additional options:

```bash
python main.py --dashboard impact --db-path /path/to/database.db --rotation 333.5
```

### Dashboard Options

- `--dashboard`: Choose between `impact` (default) or `zepp` dashboard
- `--db-path`: Path to SQLite database file
- `--rotation`: Rotation angle for impact positions (in degrees)

## Data Format

The application works with SQLite databases containing tennis sensor data. It supports:

- **Zepp Tennis Sensor** data from the Zepp Tennis 3.1 app
- **Impact position data** with x,y coordinates and stroke information

The dashboard automatically detects appropriate tables and adapts to the available data.

## Configuration

The `config.py` file allows customization of various aspects:

- **Database path**: Path to the SQLite database file
- **Timezone settings**: Configure local timezone for date/time display
- **Visualization defaults**: Default plot types, jitter, color schemes, etc.
- **Animation settings**: Speed and frame settings for animations
- **Conversion factors**: For converting speed units and other metrics

## Development

### Project Structure

```
tennis_analytics/
├── config.py                 # Configuration settings
├── core/
│   ├── __init__.py           # Package initialization
│   ├── data_manager.py       # Data loading and processing 
│   ├── session_manager.py    # Session management functionality
│   └── utils.py              # Utility functions and calculations
├── dashboards/
│   ├── __init__.py           # Package initialization
│   ├── base_dashboard.py     # Base dashboard class
│   ├── impact_explorer.py    # Impact position explorer dashboard
│   └── zepp_dashboard.py     # Zepp tennis data dashboard
├── visualization/
│   ├── __init__.py           # Package initialization
│   ├── visualizer.py         # Base visualization class
│   ├── impact_visualizer.py  # Impact position visualizations
│   └── stats_visualizer.py   # Statistical visualizations
├── main.py                   # Command line entry point
└── streamlit_app.py          # Streamlit app entry point
```

### Extending the Dashboard

To add new features or dashboards:

1. Create visualization components by extending the `Visualizer` class
2. Create a new dashboard class extending `BaseDashboard`
3. Implement custom data processing in the `DataManager` class as needed
4. Update `main.py` and `streamlit_app.py` to include the new dashboard

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly
- SQLite3

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project is based on earlier work by blue-az for tennis sensor data analysis.
