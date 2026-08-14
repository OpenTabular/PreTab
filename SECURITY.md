# Security Policy

## Supported versions

PreTab follows [Semantic Versioning](https://semver.org/). Security fixes are
made against the latest released minor version on
[PyPI](https://pypi.org/project/pretab/); older releases do not receive
backported fixes.

## Reporting a vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Report vulnerabilities privately through
[GitHub Security Advisories](https://github.com/OpenTabular/PreTab/security/advisories/new)
for this repository. Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce, or a minimal proof of concept
- The affected version(s) of PreTab

We aim to acknowledge new reports within five business days and will work with
you to understand and address the issue before any public disclosure.

> **Note:** PreTab's most security-relevant surface is deserialization.
> Loading a fitted preprocessor via `Preprocessor.from_spec` is designed to
> never execute estimator code, unlike `pickle`; it reconstructs objects
> through an allow-listed decoder over a fixed set of library modules. A
> vulnerability that breaks this guarantee is a high-priority report.
