# Computed Columns in Fit Results

## Overview

The **Computed Column** feature allows you to create new columns in the Fit Results table by evaluating mathematical expressions that reference existing columns. This is useful for calculating derived quantities, ratios, differences, or any mathematical combination of your fit parameters.

## Location

This feature is available in the **Fit Results** tab of the Spectra Workspace, located in the left panel below the filename split controls and above the "Send to Graphs Workspace" section.

## How to Use

### Basic Steps

1. **Collect Fit Results**: First, click the "Collect Fit Results" button to populate the fit results table with data from your fitted spectra.

2. **Enter Column Name**: In the "New Column Name" field, type a descriptive name for your new computed column (e.g., `Peak_Ratio`, `Total_Area`, `Normalized_Position`).

3. **Enter Mathematical Expression**: In the expression field, type a mathematical formula using existing column names (e.g., `x0_p1 - x0_p2`).

4. **Compute**: Click the "Compute & Add" button to evaluate the expression and add the new column to your results.

### Supported Operations

The expression evaluator supports standard mathematical operations:

| Operation | Symbol | Example |
|-----------|--------|---------|
| Addition | `+` | `column1 + column2` |
| Subtraction | `-` | `column1 - column2` |
| Multiplication | `*` | `column1 * 2` |
| Division | `/` | `column1 / column2` |
| Power | `**` | `column1 ** 2` |
| Modulo | `%` | `column1 % 10` |
| Parentheses | `()` | `(column1 + column2) / 2` |

### ⚠️ Important: Column Names with Special Characters

If your column names contain special characters like parentheses `()`, spaces, hyphens `-`, or operators `+*/-`, you **must** wrap them in **backticks** (`` ` ``):

**Examples:**

| Column Name | Correct Syntax | Incorrect Syntax |
|-------------|----------------|------------------|
| `x0_LO(M)` | `` `x0_LO(M)` `` | `'x0_LO(M)'` or `x0_LO(M)` |
| `Peak Area` | `` `Peak Area` `` | `'Peak Area'` or `Peak Area` |
| `x0-A1g` | `` `x0-A1g` `` | `'x0-A1g'` or `x0-A1g` |
| `area_p1+p2` | `` `area_p1+p2` `` | `area_p1+p2` |

**Complete Expression Examples:**
```
(x0_A1g - x0_E1_2g) * `x0_LO(M)`
`Peak Area` / `Background Area` 
(`x0-A1g` + `x0-E1g`) / 2
```

> [!TIP]
> When in doubt, you can always use backticks around column names - they work for normal names too!
> Example: `` `x0_p1` - `x0_p2` `` is valid even though the backticks aren't strictly required.

### Expression Examples

Here are some practical examples you can use:

#### 1. **Peak Position Difference**
Calculate the distance between two peak centers:
```
x0_p1 - x0_p2
```

#### 2. **Average Amplitude**
Calculate the average of two peak amplitudes:
```
(ampli_p1 + ampli_p2) / 2
```

#### 3. **Area Ratio**
Calculate the ratio between two peak areas:
```
area_p1 / area_p2
```

#### 4. **Percentage Difference**
Calculate percentage difference between two values:
```
(area_p1 - area_p2) * 100 / area_p1
```

#### 5. **Normalized Width**
Normalize FWHM by peak position:
```
fwhm_p1 / x0_p1
```

#### 6. **Combined Area**
Calculate total area from multiple peaks:
```
area_p1 + area_p2 + area_p3
```

#### 7. **Intensity Product**
Calculate product of amplitudes:
```
ampli_p1 * ampli_p2
```

#### 8. **Square of Parameter**
Calculate squared value (e.g., for variance):
```
fwhm_p1 ** 2
```

## Column Names

### Using Column Names in Expressions

- **Standard Column Names**: Use column names exactly as they appear in the fit results table
- **Case Sensitive**: Column names are case-sensitive (`x0_p1` ≠ `X0_p1`)
- **No Spaces**: If column names contain spaces, the system will handle them automatically

### Finding Column Names

To see available column names:
1. Look at the column headers in the Fit Results table
2. If you get an error, the error message will show you all available columns

Common column name patterns:
- `x0_[peak_name]` - Peak center position
- `fwhm_[peak_name]` - Full width at half maximum
- `ampli_[peak_name]` - Peak amplitude/intensity
- `area_[peak_name]` - Integrated peak area
- `sigma_[peak_name]` - Gaussian sigma parameter
- `gamma_[peak_name]` - Lorentzian gamma parameter

## Error Handling

The feature includes comprehensive error handling for common issues:

### Division by Zero

If your expression divides by a column containing zero values, you'll receive a warning:
```
Warning: Expression resulted in X infinite value(s). 
This may be due to division by zero.
```

The computation will still complete, with problematic values shown as `inf` or `NaN` in the table.

### Invalid Column Names

If you reference a column that doesn't exist:
```
Error: Invalid column name in expression.
Available columns: Filename, x0_p1, fwhm_p1, ...
```

**Solution**: Check the available columns listed in the error message and correct your expression.

### Syntax Errors

If your expression has invalid syntax:
```
Error: Invalid expression syntax.
Examples: 'column1 - column2', '(col1 + col2) * 2'
```

**Solution**: Review your expression for typos, missing operators, or unmatched parentheses.

### Duplicate Column Names

If you try to create a column that already exists:
```
Column 'MyColumn' already exists. Please choose a different name.
```

**Solution**: Use a unique name for your new column or delete the existing column first.

## Tips and Best Practices

### 1. **Start Simple**
Begin with simple expressions (e.g., `column1 + column2`) and gradually build complexity.

### 2. **Use Parentheses**
When combining operations, use parentheses to ensure correct order:
- Good: `(area_p1 + area_p2) / 2`
- Risky: `area_p1 + area_p2 / 2` (division happens first!)

### 3. **Handle Division Carefully**
Before dividing, consider if the denominator could be zero:
- Add a small constant: `column1 / (column2 + 0.001)`
- Check the data first in the table

### 4. **Descriptive Names**
Use clear, descriptive column names that explain what the calculation represents:
- Good: `Peak_Ratio_p1_p2`
- Poor: `calc1`

### 5. **Document Your Calculations**
Keep notes about complex expressions for future reference.

### 6. **Test Incrementally**
For complex calculations, break them into steps:
1. Create intermediate columns
2. Verify each step
3. Combine for final result

### 7. **Check Results**
After computation:
- Review the values in the table
- Look for unexpected `NaN` or `inf` values
- Spot-check a few rows manually

## Integration with Other Features

### Saving Results

Computed columns are included when you:
- **Save to Excel**: Computed columns appear in the saved Excel file
- **Save to CSV**: Computed columns are included in CSV exports
- **Send to Graphs**: Computed columns can be visualized in the Graphs workspace

### Workflow Example

1. Collect fit results from all spectra
2. Split filename to extract sample information
3. Add computed columns for derived quantities
4. Send entire dataframe to Graphs workspace for visualization
5. Save to Excel for further analysis

## Technical Details

### Safe Expression Evaluation

The feature uses `pandas.DataFrame.eval()` for expression evaluation, which:
- **Is Safe**: Only allows mathematical expressions, not arbitrary code
- **Is Fast**: Optimized for numerical operations
- **Handles Errors**: Gracefully manages division by zero and type mismatches

### Numerical Precision

- Results are automatically rounded to 3 decimal places for consistency
- Original precision is maintained internally for further calculations

## Troubleshooting

| Problem | Possible Cause | Solution |
|---------|---------------|----------|
| "No fit results available" | Haven't collected results yet | Click "Collect Fit Results" first |
| "Please enter a column name" | Column name field is empty | Enter a name for the new column |
| "Please enter a mathematical expression" | Expression field is empty | Enter a valid mathematical expression |
| Values show as `NaN` | Invalid mathematical operation | Check for division by zero or incompatible operations |
| Values show as `inf` | Division by zero | Add small constant to denominator or filter data |

## Examples by Use Case

### Quality Control

**Check peak position consistency:**
```
Position_Variation = abs(x0_p1 - x0_p2)
```

### Normalization

**Normalize areas to first peak:**
```
Normalized_Area = area_p2 / area_p1
```

### Statistical Analysis

**Calculate standard error:**
```
SE_Amplitude = ampli_p1 / sqrt(N)
```
*(Note: You'd need to add N as a column first)*

### Peak Characterization

**Calculate peak asymmetry:**
```
Asymmetry = (fwhm_right - fwhm_left) / fwhm_p1
```

## See Also

- **Filename Splitting**: Extract information from spectrum filenames
- **Graphs Workspace**: Visualize computed columns
- **Excel Export**: Save results with computed columns

---

*For additional help or to report issues, please contact the development team.*
