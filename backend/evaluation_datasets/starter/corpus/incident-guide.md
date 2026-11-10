# Synthetic Incident Guide

<!-- eval-chunk:c01 -->
## Severity
A fictional Northstar incident is P1 when a critical customer-facing function is unavailable to all users or when unauthorized access is confirmed. A partial degradation affecting one team is P2 unless there is evidence of unauthorized access.

<!-- eval-chunk:c02 -->
## Communication
For a P1 incident, the incident lead posts the first internal update within 15 minutes of declaration and then every 30 minutes until mitigation. Customer communications require approval from the communications lead.

<!-- eval-chunk:c03 -->
## Recovery
The incident lead may initiate rollback when a deployment is the likely cause and the rollback procedure is documented. The on-call engineer executes and timestamps the rollback. A review meeting is scheduled within two business days after recovery.
