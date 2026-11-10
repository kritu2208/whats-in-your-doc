# Synthetic Operations Handbook

<!-- eval-chunk:c01 -->
## Service hours
The fictional Northstar service desk is staffed from 08:00 to 18:00 UTC on weekdays. It is not staffed on public holidays. The on-call engineer covers urgent incidents outside those hours.

<!-- eval-chunk:c02 -->
## Support routing
Routine account questions go to the service desk queue. Suspected security incidents go directly to the security response queue. A service desk agent must not reclassify a suspected security incident as routine without security-team approval.

<!-- eval-chunk:c03 -->
## Approvals
Changes to production settings require approval from both the service owner and the on-call engineer. Either person may request a rollback, but the on-call engineer records the rollback time in the change log.
