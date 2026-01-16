# Automated Testing Prompt

## Project Context

My Python project is now working as expected and includes all required features.

Please **read and analyze the entire codebase** to fully understand:
- the project architecture,
- the application logic,
- the signal and data flow between components.

You may also refer to `doc/dev_instructions.md` for additional context and design explanations.

---

## Objective

Design and implement **automated tests** for this project.

- Write the test code as Python `.py` files
- Place **all tests inside a `tests/` directory**
- The tests must allow validation of all core functionalities when:
  - deploying new versions
  - developing new features

---

## Project Structure

The project contains **three distinct workspaces**, each requiring dedicated tests.

---

## 1. Maps Workspace & Spectra Workspace

Write tests that validate the following functionalities:

### File Handling
- Open all supported spectral file formats:
  - 2D maps
  - Single spectra

### Data Processing & Fitting
- Crop spectral ranges
- Define baseline anchor points
- Subtract baseline
- Define peak model(s)
- Perform peak fitting

### Persistence
- Save fitting models
- Save workspace state
- Reload workspace state and verify consistency

---

## 2. Graphs Workspace

Write tests that validate the following functionalities:

### Data Handling
- Load data from Excel files

### Plotting
- Generate figures for all supported plot types

### Data Manipulation
- Apply and validate data filtering features

### Persistence
- Save workspace state
- Reload workspace state and verify consistency

---

## Testing Requirements

- Prefer **pytest**
- Tests should be:
  - deterministic
  - isolated
  - reproducible
- Use fixtures where appropriate
- Mock GUI components and file dialogs when necessary
- Avoid hard-coded absolute paths

If some components are difficult to test directly, propose appropriate **testing strategies** such as:
- mocking
- integration tests
- regression tests

Please organize the tests logically and explain any important design decisions.

---

## Architecture Notes (Optional but Recommended)

The project follows a **MVVM architecture using PySide6**.

- Prefer testing **Models and ViewModels** directly
- Mock Views and GUI-related components when possible
- Focus on logic, data integrity, and signal flow rather than UI rendering
