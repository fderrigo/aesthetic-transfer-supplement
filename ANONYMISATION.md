# What was removed or changed to make this copy anonymous

The analysis scripts run on this copy and give the same numbers as on the original data (checked for the replication,
VLM-replication and decomposition analyses: every estimate identical).

## Participants

Participants were anonymous from the start: no name, e-mail, telephone, address, IP address or user agent was ever
collected. In addition, for this copy:

| Field | Change |
|---|---|
| participant identifier | replaced by a new number in random order, the same in every table |
| anonymous code (the code that let a participant resume a session) | replaced by `P0001`…; the original codes are not here |
| dates and times of every participant event | replaced by the time elapsed since that participant's first event, written as a date starting at 2000-01-01T00:00:00Z. Order, durations and response times are kept; the calendar date and time of day are not |
| country | kept only where at least 10 participants share it; the others are `other` |
| professional role | categories with fewer than 10 participants merged into `Other` |
| age range, self-assessed expertise, interface language | unchanged (already coarse) |
| free-text reasons written by staff for an exclusion | unchanged (they describe the session, not the person) |

## Researchers and infrastructure

| Item | Change |
|---|---|
| staff account names in every table and JSON file | replaced by `staff` or removed |
| the audit log of the platform | not included; `protocol/timeline_from_audit_log.csv` keeps the study-level events (protocol, selection rule, datasets, prompts, plans, training runs, cloud jobs, defect screening) without users. Events about individual participants are left out |
| identifiers of cloud instances, projects and accounts | removed |
| provider response objects of the VLM requests | removed (they carry request identifiers of the account); every attempt is kept with model, time, outcome, token usage and the parsed answer |
| API keys, passwords, host names | never stored in the material; the configuration file here has empty fields |
| paths, user names, version-control history | removed; this copy starts from a single commit by an anonymous author |

## Third parties

`data/export/image_metadata.csv` credits the photographers of the 600 source photographs (name, licence, source URL):
this is the attribution required by their licences and is kept. E-mail addresses that appeared in those credits were removed.

## Checked

The whole tree was scanned for names, e-mail addresses, account and host identifiers, key patterns and user paths
before the commit. Images are re-encoded JPEG files without metadata.
