# Synthetic Release Calendar

<!-- eval-chunk:c01 -->
## Cadence
The fictional Northstar application has a planned production release every second Wednesday. A release may be postponed if its required quality gates are incomplete.

<!-- eval-chunk:c02 -->
## Change freeze
A change freeze begins at 16:00 UTC on the Monday before a planned release and ends when the release is verified. Only an approved emergency fix may be deployed during the freeze.

<!-- eval-chunk:c03 -->
## Release approval
The release coordinator publishes the go/no-go decision by 12:00 UTC on release day. Approval requires a passing automated test report and sign-off from the service owner.
