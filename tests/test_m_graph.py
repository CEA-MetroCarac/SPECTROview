"""
Tests for model/m_graph.py - Graph configuration model

Tests cover:
- Graph initialization with default properties
- save() and load() methods (serialization)
- get_display_name() method
- Property validation
"""

import pytest
from spectroview.model.m_graph import MGraph


class TestMGraphInitialization:
    """Tests for MGraph initialization."""
    
    def test_create_graph_with_id(self):
        """Test creating graph with specific ID."""
        graph = MGraph(graph_id=5)
        
        assert graph.graph_id == 5
    
    def test_create_graph_without_id(self):
        """Test creating graph without ID."""
        graph = MGraph()
        
        assert graph.graph_id is None
    
    def test_default_properties(self):
        """Test that graph has correct default properties."""
        graph = MGraph()
        
        # Data source
        assert graph.df_name is None
        assert graph.filters == []
        
        # Plot configuration
        assert graph.plot_style == "point"
        assert graph.plot_width == 480
        assert graph.plot_height == 420
        assert graph.dpi == 100
        
        # Axes
        assert graph.x is None
        assert graph.y == []
        assert graph.z is None
        assert graph.y2 is None
        assert graph.y3 is None
        
        # Limits (all None by default)
        assert graph.xmin is None
        assert graph.xmax is None
        assert graph.ymin is None
        assert graph.ymax is None
        
        # Scales
        assert graph.xlogscale is False
        assert graph.ylogscale is False
        
        # Labels
        assert graph.plot_title is None
        assert graph.xlabel is None
        assert graph.ylabel is None
        
        # Legend
        assert graph.legend_visible is True
        assert graph.legend_location == 'upper right'
        assert graph.legend_outside is False
        
        # Plot-specific
        assert graph.color_palette == "jet"
        assert graph.wafer_size == 300.0


class TestMGraphSaveLoad:
    """Tests for save() and load() methods."""
    
    def test_save_returns_dict(self):
        """Test that save() returns a dictionary."""
        graph = MGraph(graph_id=1)
        
        data = graph.save()
        
        assert isinstance(data, dict)
        assert 'graph_id' in data
        assert data['graph_id'] == 1
    
    def test_save_includes_all_properties(self):
        """Test that save() includes all graph properties."""
        graph = MGraph(graph_id=10)
        graph.df_name = "test_df"
        graph.plot_style = "line"
        graph.x = "X_column"
        graph.y = ["Y1", "Y2"]
        graph.xlabel = "X Label"
        graph.ylabel = "Y Label"
        
        data = graph.save()
        
        # Verify all important properties are saved
        assert data['df_name'] == "test_df"
        assert data['plot_style'] == "line"
        assert data['x'] == "X_column"
        assert data['y'] == ["Y1", "Y2"]
        assert data['xlabel'] == "X Label"
        assert data['ylabel'] == "Y Label"
    
    def test_load_from_dict(self):
        """Test loading graph from dictionary."""
        graph = MGraph()
        
        data = {
            'graph_id': 99,
            'df_name': 'loaded_df',
            'plot_style': 'bar',
            'x': 'Category',
            'y': ['Value1', 'Value2'],
            'xlabel': 'Categories',
            'ylabel': 'Values',
            'xmin': 0.0,
            'xmax': 10.0,
            'grid': True,
        }
        
        graph.load(data)
        
        # Verify properties are loaded correctly
        assert graph.graph_id == 99
        assert graph.df_name == 'loaded_df'
        assert graph.plot_style == 'bar'
        assert graph.x == 'Category'
        assert graph.y == ['Value1', 'Value2']
        assert graph.xlabel == 'Categories'
        assert graph.ylabel == 'Values'
        assert graph.xmin == 0.0
        assert graph.xmax == 10.0
        assert graph.grid is True
    
    def test_save_load_roundtrip(self):
        """Test that save then load preserves all data."""
        # Create graph with custom properties
        graph1 = MGraph(graph_id=42)
        graph1.df_name = "test_data"
        graph1.plot_style = "scatter"
        graph1.x = "X"
        graph1.y = ["Y1", "Y2", "Y3"]
        graph1.z = "Z"
        graph1.xlabel = "X Axis"
        graph1.ylabel = "Y Axis"
        graph1.zlabel = "Z Axis"
        graph1.xmin = 0.0
        graph1.xmax = 100.0
        graph1.ymin = -10.0
        graph1.ymax = 10.0
        graph1.xlogscale = True
        graph1.grid = True
        graph1.legend_location = "lower left"
        graph1.color_palette = "viridis"
        
        # Save to dict
        data = graph1.save()
        
        # Load into new graph
        graph2 = MGraph()
        graph2.load(data)
        
        # Verify all properties match
        assert graph2.graph_id == graph1.graph_id
        assert graph2.df_name == graph1.df_name
        assert graph2.plot_style == graph1.plot_style
        assert graph2.x == graph1.x
        assert graph2.y == graph1.y
        assert graph2.z == graph1.z
        assert graph2.xlabel == graph1.xlabel
        assert graph2.ylabel == graph1.ylabel
        assert graph2.zlabel == graph1.zlabel
        assert graph2.xmin == graph1.xmin
        assert graph2.xmax == graph1.xmax
        assert graph2.ymin == graph1.ymin
        assert graph2.ymax == graph1.ymax
        assert graph2.xlogscale == graph1.xlogscale
        assert graph2.grid == graph1.grid
        assert graph2.legend_location == graph1.legend_location
        assert graph2.color_palette == graph1.color_palette
    
    def test_load_handles_none_limits(self):
        """Test that loading handles None/empty limit values."""
        graph = MGraph()
        
        data = {
            'xmin': None,
            'xmax': '',
            'ymin': '5.5',
            'ymax': None,
        }
        
        graph.load(data)
        
        # Verify None and empty strings are handled
        assert graph.xmin is None
        assert graph.xmax is None
        assert graph.ymin == 5.5
        assert graph.ymax is None
    
    def test_load_converts_string_limits_to_float(self):
        """Test that string limit values are converted to float."""
        graph = MGraph()
        
        data = {
            'xmin': '0.5',
            'xmax': '10.5',
            'ymin': '−5',  # This might fail conversion
            'ymax': '5.0',
        }
        
        graph.load(data)
        
        # Verify conversion
        assert graph.xmin == 0.5
        assert graph.xmax == 10.5
        assert graph.ymax == 5.0


class TestMGraphDisplayName:
    """Tests for get_display_name() method."""
    
    def test_display_name_with_all_axes(self):
        """Test display name when all axes are set."""
        graph = MGraph(graph_id=1)
        graph.plot_style = "line"
        graph.x = "Time"
        graph.y = ["Temperature"]
        graph.z = "Pressure"
        
        display_name = graph.get_display_name()
        
        assert display_name == "1-line_plot: [Time] - [Temperature] - [Pressure]"
    
    def test_display_name_with_missing_axes(self):
        """Test display name when some axes are None."""
        graph = MGraph(graph_id=5)
        graph.plot_style = "scatter"
        graph.x = "X"
        graph.y = []
        graph.z = None
        
        display_name = graph.get_display_name()
        
        assert display_name == "5-scatter_plot: [X] - [None] - [None]"
    
    def test_display_name_with_default_values(self):
        """Test display name with default graph properties."""
        graph = MGraph(graph_id=0)
        
        display_name = graph.get_display_name()
        
        assert display_name == "0-point_plot: [None] - [None] - [None]"
    
    def test_display_name_with_multiple_y_values(self):
        """Test display name when y has multiple columns."""
        graph = MGraph(graph_id=3)
        graph.plot_style = "point"
        graph.x = "X"
        graph.y = ["Y1", "Y2", "Y3"]
        
        display_name = graph.get_display_name()
        
        # Should only show first y-value
        assert display_name == "3-point_plot: [X] - [Y1] - [None]"


class TestMGraphProperties:
    """Tests for setting and modifying graph properties."""
    
    def test_set_plot_configuration(self):
        """Test setting plot configuration properties."""
        graph = MGraph()
        
        graph.plot_width = 800
        graph.plot_height = 600
        graph.dpi = 150
        
        assert graph.plot_width == 800
        assert graph.plot_height == 600
        assert graph.dpi == 150
    
    def test_set_axis_limits(self):
        """Test setting axis limits."""
        graph = MGraph()
        
        graph.xmin = -10.0
        graph.xmax = 10.0
        graph.ymin = 0.0
        graph.ymax = 100.0
        
        assert graph.xmin == -10.0
        assert graph.xmax == 10.0
        assert graph.ymin == 0.0
        assert graph.ymax == 100.0
    
    def test_set_filters(self):
        """Test setting data filters."""
        graph = MGraph()
        
        graph.filters = ["X > 5", "Y < 100"]
        
        assert len(graph.filters) == 2
        assert "X > 5" in graph.filters
