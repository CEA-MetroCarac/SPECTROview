The `pandas.DataFrame.query()` method allows you to filter rows from a
DataFrame based on a boolean expression. It's a powerful and flexible way to
subset your DataFrame to include only the rows that meet specific conditions.

Here's how you can use the `query()` method:

1. **Basic Syntax**:
   ```
   df.query('expression')
   ```
    - `df`: The DataFrame you want to filter.
    - `'expression'`: A string containing the filtering expression.

2. **Filtering Expression**:
    - The expression is a string that contains a condition or a combination of
      conditions.
    - You can use column names directly in the expression.
    - You can use comparison operators (e.g., `==`, `<`, `>`, `<=`, `>=`, `!=`)
      and logical operators (e.g., `and`, `or`, `not`) to build complex
      conditions.

3. **Example**:
   ```python
   filtered_df = df.query('Age >= 25 and Gender == "Male"')
   ```
   In this example, `filtered_df` will contain only the rows where the "Age"
   column is greater than or equal to 25 and the "Gender" column is "Male."

4. **Variables in the Expression**:
    - You can use variables in the expression by prefixing them with the `@`
      symbol.
    - For example, if you have a variable `age_threshold`, you can use it like
      this:
      ```python
      age_threshold = 25
      filtered_df = df.query('Age >= @age_threshold')
      ```

5. **String Quoting**:
    - When dealing with string values in the expression, you need to enclose
      them in single or double quotes.
    - For example:
      ```python
      filtered_df = df.query('Name == "John"')
      ```

6. **Escaping Quotes**:
    - If your string contains quotes, you can escape them with a backslash:
      ```python
      filtered_df = df.query('Name == "John\'s Pizza"')
      ```

7. **Multiple Conditions**:
    - You can use parentheses to group conditions and create complex
      expressions:
      ```python
      filtered_df = df.query('(Age >= 25 and Gender == "Male") or (Age < 25 and Gender == "Female")')
      ```

8. **Return Type**:
    - The `query()` method returns a new DataFrame containing the filtered
      rows. The original DataFrame remains unchanged.

