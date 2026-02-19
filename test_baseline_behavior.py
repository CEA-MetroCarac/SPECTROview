
try:
    import numpy as np
    from fitspy.core.baseline import BaseLine
    
    x = np.linspace(0, 100, 100)
    y = np.sin(x/10)
    
    bl = BaseLine()
    bl.mode = "arpls"
    bl.coef = 5.0
    bl.eval(x, y)
    
    print("After first eval, y_eval exists:", bl.y_eval is not None)
    
    # Change coef
    bl.coef = 6.0
    print("After changing coef, y_eval exists:", bl.y_eval is not None)
    
    # Change it back (simulating slight move or just change)
    bl.coef = 5.5
    
    # If y_eval persists, let's see if eval() uses it
    # We can't easily see IF it uses it, but we saw stability diff earlier.
    
    # Try manual clear
    bl.y_eval = None
    print("After manual clear, y_eval exists:", bl.y_eval is not None)

except Exception as e:
    print(e)
