
try:
    import numpy as np
    from fitspy.core.baseline import BaseLine
    
    # Create data
    x = np.linspace(0, 100, 200)
    y_base_true = 10 + 0.1 * x + 5 * np.exp(-((x-50)/20)**2) 
    y = y_base_true + np.exp(-((x-30)/2)**2) * 10
    
    # 1. Simulate "Warm Start" (Previous behavior)
    bl_warm = BaseLine()
    bl_warm.mode = "arpls"
    bl_warm.coef = 1.0
    bl_warm.eval(x, y) # init
    
    # Drag slider
    for c in np.linspace(1.0, 5.0, 10):
        bl_warm.coef = c
        bl_warm.eval(x, y) # Uses cached y_eval
    y_final_warm = bl_warm.y_eval.copy()
    
    # 2. Simulate "Fresh Start" (Paste behavior)
    bl_fresh = BaseLine()
    bl_fresh.mode = "arpls"
    bl_fresh.coef = 5.0
    bl_fresh.eval(x, y)
    y_final_fresh = bl_fresh.y_eval.copy()
    
    print(f"Diff without fix (Warm vs Fresh): {np.max(np.abs(y_final_warm - y_final_fresh))}")
    
    
    # 3. Simulate "Fix" (Clearing y_eval)
    bl_fix = BaseLine()
    bl_fix.mode = "arpls"
    bl_fix.coef = 1.0
    bl_fix.eval(x, y)
    
    # Drag slider with clear
    for c in np.linspace(1.0, 5.0, 10):
        bl_fix.coef = c
        bl_fix.y_eval = None # FIX: Clear cache
        bl_fix.eval(x, y)
    y_final_fix = bl_fix.y_eval.copy()
    
    print(f"Diff with fix (Cleared vs Fresh): {np.max(np.abs(y_final_fix - y_final_fresh))}")

except Exception as e:
    print(e)
