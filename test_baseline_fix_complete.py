
try:
    import numpy as np
    from fitspy.core.baseline import BaseLine, generate_penalties
    
    x = np.linspace(0, 100, 200)
    y_base_true = 10 + 0.1 * x + 5 * np.exp(-((x-50)/20)**2) 
    y = y_base_true + np.exp(-((x-30)/2)**2) * 10
    
    # 1. Warm (Preview behavior)
    # We clear cache BEFORE preview (as implemented)
    generate_penalties.cache_clear()
    
    bl_warm = BaseLine()
    bl_warm.mode = "arpls"
    for c in np.linspace(1.0, 5.0, 10):
        # We clear cache BEFORE each step of preview (as implemented)
        generate_penalties.cache_clear()
        bl_warm.coef = c
        bl_warm.eval(x, y)
    y_final_warm = bl_warm.y_eval.copy()
    
    # 2. Fresh (Paste behavior)
    # We clear cache BEFORE paste (as implemented)
    generate_penalties.cache_clear()
    
    bl_fresh = BaseLine()
    bl_fresh.mode = "arpls"
    bl_fresh.coef = 5.0
    bl_fresh.eval(x, y)
    y_final_fresh = bl_fresh.y_eval.copy()
    
    print(f"Diff with FULL FIX (Clean Warm vs Clean Fresh): {np.max(np.abs(y_final_warm - y_final_fresh))}")
    
    if np.max(np.abs(y_final_warm - y_final_fresh)) < 1e-9:
        print("SUCCESS: Result uses clean cache in both cases.")
    else:
        print("FAILURE: Difference detected.")

except Exception as e:
    print(e)
