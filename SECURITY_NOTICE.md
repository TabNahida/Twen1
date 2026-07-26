# Profiler artifact security notice

Nsight Systems reports and exported databases can embed the complete launch
environment in capture metadata. PyTorch traces and profiler-side JSON may also
contain host paths, process metadata, or environment-derived strings. Treat every
file under `artifacts/profiles/` as local confidential data, even when its filename
looks like a summary.

- Keep raw `.nsys-rep`, `.sqlite`, trace JSON, and related profiler artifacts mode
  `0600`. Start future captures under `umask 077`, then explicitly run `chmod 600`
  and verify the resulting modes; some profiler/export paths may set permissions
  independently of the caller's umask.
- Never publish, attach, upload, or force-add raw profiler artifacts to Git. The
  repository-wide `/artifacts/` ignore rule already covers this directory.
- Performance report bundles must exclude raw profiler files. Include only reviewed,
  derived aggregate tables or summaries that contain no environment metadata,
  credentials, remote-control payloads, private host details, or user paths.
- Launch profiled processes from a clean `env -i` allowlist. Do not pass API keys,
  access tokens, secrets, passwords, authorization values, cookies, private keys,
  remote payloads, or unrelated session variables to the profiler.
- If a credential was captured by an existing report, regard that credential as
  exposed in a local artifact, rotate it, keep the artifact private, and regenerate
  any evidence intended for sharing from a clean environment.

Before accepting a new capture, inspect its exported metadata by variable name only.
The audit must not print or copy environment values. A shareable derived report must
also be checked independently before it enters a report bundle.
