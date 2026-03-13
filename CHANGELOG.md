# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-13
### Added
- Initial release of muTimer.
- Hierarchical timing utility with nested context manager support.
- Accumulation of time across multiple calls to the same timer.
- Call counting for repeated operations.
- Hierarchical summary output in tabular format.
- Optional depth limiting for nested timers.
- Optional memory tracking (RSS) using `psutil`.
