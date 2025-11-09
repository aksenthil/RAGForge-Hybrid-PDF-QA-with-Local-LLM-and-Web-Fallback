### Contributing

Thanks for your interest in improving this project! Please follow these guidelines:

- Open an issue describing the change or bug before submitting a PR.
- Keep PRs focused and small; include screenshots for UI changes.
- Run the app locally and smoke-test before submitting.
- Avoid committing large binaries or datasets.

Development quickstart (Windows PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Coding standards:
- Prefer readability; avoid deep nesting.
- Add concise comments only for non-obvious logic.
- Keep functions focused and testable.


