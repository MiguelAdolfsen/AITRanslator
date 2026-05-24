# Testing

Fast tests live in `.testing/tests` and use Python's built-in `unittest`.

Run the default checks:

```powershell
.\.testing\run_tests.ps1
```

Run the same checks plus one lightweight image smoke test:

```powershell
.\.testing\run_tests.ps1 -WithSmoke
```

The smoke test writes output under `.testing/output`.
