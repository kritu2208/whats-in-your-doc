# Synthetic Data Handling Note

<!-- eval-chunk:c01 -->
## Retention
Fictional Northstar support attachments are retained for 30 calendar days after ticket closure. Audit records are retained for 180 calendar days. These two retention periods apply to different record types.

<!-- eval-chunk:c02 -->
## Deletion
A verified deletion request is completed within five business days. The operator records the request reference and completion date in the audit log. Backups expire on their separately documented 35-day rotation.

<!-- eval-chunk:c03 -->
## Exports
Bulk exports require approval from the data steward. The export record includes the requester, approval reference, exported record count, and export date. Individual users may download their own profile data without bulk-export approval.
