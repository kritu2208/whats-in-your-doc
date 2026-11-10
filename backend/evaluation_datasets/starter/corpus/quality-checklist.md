# Synthetic Quality Checklist

<!-- eval-chunk:c01 -->
## Automated gates
Every release candidate must pass unit tests, API contract tests, and dependency vulnerability scanning. A failed gate blocks release approval; repeating a failed test without investigating does not count as a pass.

<!-- eval-chunk:c02 -->
## Manual review
The reviewer checks keyboard navigation, narrow-screen layout, and citation rendering against the API response. The review record includes the build identifier, reviewer, date, and any accepted exceptions.

<!-- eval-chunk:c03 -->
## Defect policy
A release-blocking defect has no documented workaround and affects a critical user flow. The service owner may accept a non-blocking defect only when its impact, owner, and follow-up date are recorded.
