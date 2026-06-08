import sys
import cProfile
import pstats
from PySide6.QtWidgets import QApplication
from spectroview.main import Main

def run_profile():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    main_window = Main()
    main_window.show()
    
    print("Profiling toggle_theme...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    main_window.toggle_theme("light")
    
    profiler.disable()
    
    with open("profile_results.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats(30)
    
    print("Done. Saved to profile_results.txt")

if __name__ == "__main__":
    run_profile()
