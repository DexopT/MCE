# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.0.1   | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability in MCE, **please do not open a public issue.**

Instead, report it privately:

1. **Email**: [dexopt1@gmail.com](mailto:dexopt1@gmail.com)
2. **Subject**: `[MCE Security] Brief description`
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

You will receive an acknowledgment within **48 hours** and a detailed response within **7 days**.

## Security Considerations

MCE acts as a transparent proxy between AI agents and tool servers. Keep in mind:

- **Policy Engine**: MCE blocks destructive commands (`rm -rf`, `mkfs`, etc.) by default. Review and customize `config.yaml → policy` for your environment.
- **Network Exposure**: By default MCE binds to `127.0.0.1` (localhost only). Do not expose it to the public internet without authentication.
- **Upstream Trust**: MCE forwards requests to upstream MCP servers. Only configure trusted upstream URLs.
- **HitL Prompts**: High-risk operations require terminal approval. In non-interactive environments, these are automatically denied.
- **No Authentication**: MCE v0.0.1 does not include authentication. If deploying in a shared environment, place it behind a reverse proxy with auth.

## Responsible Disclosure

We follow responsible disclosure practices. Security researchers who report valid vulnerabilities will be credited in the changelog (with permission).
