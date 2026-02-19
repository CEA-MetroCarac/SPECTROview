
try:
    import numpy as np
    from fitspy.core.baseline import BaseLine, generate_penalties
    
    x = np.linspace(0, 100, 200)
    y_base_true = 10 + 0.1 * x + 5 * np.exp(-((x-50)/20)**2) 
    y = y_base_true + np.exp(-((x-30)/2)**2) * 10
    
    # 1. Warm
    bl_warm = BaseLine()
    bl_warm.mode = "arpls"
    for c in np.linspace(1.0, 5.0, 10):
        bl_warm.coef = c
        bl_warm.eval(x, y)
    y_warm = bl_warm.y_eval.copy()
    
    # 2. Fresh (Normal) - potentially affected by cache
    bl_fresh = BaseLine()
    bl_fresh.mode = "arpls"
    bl_fresh.coef = 5.0
    bl_fresh.eval(x, y)
    y_fresh = bl_fresh.y_eval.copy()
    
    print(f"Warm vs Fresh (Dirty Cache): {np.max(np.abs(y_warm - y_fresh))}")
    
    # 3. Clean Cache Fresh
    generate_penalties.cache_clear()
    
    bl_clean = BaseLine()
    bl_clean.mode = "arpls"
    bl_clean.coef = 5.0
    bl_clean.eval(x, y)
    y_clean = bl_clean.y_eval.copy()
    
    print(f"Warm vs CleanCache: {np.max(np.abs(y_warm - y_clean))}")
    print(f"Fresh vs CleanCache: {np.max(np.abs(y_fresh - y_clean))}")

except Exception as e:
    print(e)
