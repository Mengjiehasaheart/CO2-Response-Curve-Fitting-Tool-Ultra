"""
GUI module for ACI_py - Graphical User Interface components.

This module provides a web-based interface using Streamlit for 
interactive photosynthesis curve analysis.
"""

# Version info
__version__ = '0.1.0'

# GUI framework choice
GUI_FRAMEWORK = 'streamlit'

# Module info
__all__ = ['GUI_FRAMEWORK', '__version__', 'launch_gui']

# Usage instructions
def launch_gui():
    """
    Launch the ACI_py GUI application.
    
    This function provides instructions for running the Streamlit app.
    In future versions, it may directly launch the application.
    """
    print("\n🌿 ACI_py GUI Launcher")
    print("=" * 50)
    print("\nTo launch the ACI_py GUI, run:")
    print("  streamlit run aci_py/gui/streamlit_app.py")
    print("\nFor the demo version:")
    print("  streamlit run aci_py/gui/streamlit_demo.py")
    print("\nThe app will open in your default web browser.")
    print("=" * 50)