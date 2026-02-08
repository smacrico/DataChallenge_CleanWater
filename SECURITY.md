# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: security@your-organization.com

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## Security Best Practices

### Credentials Management

- **Never commit credentials** to version control
- Use `.env` files locally (listed in `.gitignore`)
- In production, use environment variables or secret management services
- Rotate credentials regularly

### Snowflake Security

- Use role-based access control (RBAC)
- Limit warehouse and database permissions
- Enable multi-factor authentication (MFA)
- Use network policies to restrict access by IP

### Dependencies

- Regularly update dependencies via `pip install --upgrade`
- Monitor security advisories for Python packages
- Use `pip-audit` or similar tools to scan for vulnerabilities

### Data Privacy

- This project uses **only public datasets**
- Do not upload proprietary or sensitive data to Snowflake stages
- Ensure compliance with data protection regulations (GDPR, POPIA, etc.)

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find similar problems
3. Prepare fixes for all supported versions
4. Release patches as soon as possible

Thank you for helping keep this project secure! 🔒
