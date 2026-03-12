# Security Operations Guide

This guide covers CI security gates and SQL alert queries for runtime security monitoring.

## CI Security Gates

The CI pipeline enforces:

1. `pip-audit` dependency vulnerability scan.
2. `gitleaks` secret scan (`--no-git`) against the repo contents.

If either step fails, merge is blocked.

## Runtime Security Events

Security-relevant actions are written to `security_events`:

```sql
security_events (
  id SERIAL PRIMARY KEY,
  user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
  event_type TEXT NOT NULL,
  ip TEXT NOT NULL,
  details_json TEXT NOT NULL,
  created_at TEXT NOT NULL
)
```

Login lockouts are tracked in `auth_throttles`:

```sql
auth_throttles (
  identity_hash TEXT PRIMARY KEY,
  failures INTEGER NOT NULL,
  window_started_at TEXT NOT NULL,
  locked_until TEXT,
  updated_at TEXT NOT NULL
)
```

Automatic IP blocking state is tracked in `ip_blocks`:

```sql
ip_blocks (
  ip TEXT PRIMARY KEY,
  blocked_until TEXT NOT NULL,
  reason TEXT NOT NULL,
  updated_at TEXT NOT NULL
)
```

If `ASE_RATE_LIMIT_BACKEND=database`, shared limiter events are tracked in `rate_limit_events`.

## Recommended Alerts

Run these queries on a schedule (e.g., every 5 minutes) and trigger alerts on non-empty results.

### 1) Brute-force login attempts per IP (15 minutes)

```sql
SELECT
  ip,
  COUNT(*) AS failed_login_count
FROM security_events
WHERE event_type = 'auth_login_failed'
  AND created_at >= (NOW() AT TIME ZONE 'UTC' - INTERVAL '15 minutes')::text
GROUP BY ip
HAVING COUNT(*) >= 20
ORDER BY failed_login_count DESC;
```

### 2) Registration abuse per IP (30 minutes)

```sql
SELECT
  ip,
  COUNT(*) AS register_attempts
FROM security_events
WHERE event_type IN ('auth_register_conflict', 'auth_register_success')
  AND created_at >= (NOW() AT TIME ZONE 'UTC' - INTERVAL '30 minutes')::text
GROUP BY ip
HAVING COUNT(*) >= 30
ORDER BY register_attempts DESC;
```

### 3) Job flooding by user (15 minutes)

```sql
SELECT
  user_id,
  COUNT(*) AS jobs_created
FROM security_events
WHERE event_type = 'job_created'
  AND created_at >= (NOW() AT TIME ZONE 'UTC' - INTERVAL '15 minutes')::text
GROUP BY user_id
HAVING COUNT(*) >= 15
ORDER BY jobs_created DESC;
```

### 4) Shared IP across multiple accounts (24 hours)

```sql
SELECT
  ip,
  COUNT(DISTINCT user_id) AS unique_users,
  COUNT(*) AS total_events
FROM security_events
WHERE user_id IS NOT NULL
  AND created_at >= (NOW() AT TIME ZONE 'UTC' - INTERVAL '24 hours')::text
GROUP BY ip
HAVING COUNT(DISTINCT user_id) >= 10
ORDER BY unique_users DESC, total_events DESC;
```

### 5) Top risky IP activity summary (24 hours)

```sql
SELECT
  ip,
  SUM(CASE WHEN event_type = 'auth_login_failed' THEN 1 ELSE 0 END) AS failed_logins,
  SUM(CASE WHEN event_type = 'auth_register_conflict' THEN 1 ELSE 0 END) AS register_conflicts,
  SUM(CASE WHEN event_type = 'job_created' THEN 1 ELSE 0 END) AS jobs_created
FROM security_events
WHERE created_at >= (NOW() AT TIME ZONE 'UTC' - INTERVAL '24 hours')::text
GROUP BY ip
ORDER BY failed_logins DESC, register_conflicts DESC, jobs_created DESC
LIMIT 50;
```

### 6) Active login lockouts

```sql
SELECT
  identity_hash,
  failures,
  window_started_at,
  locked_until,
  updated_at
FROM auth_throttles
WHERE locked_until IS NOT NULL
  AND locked_until >= (NOW() AT TIME ZONE 'UTC')::text
ORDER BY locked_until DESC;
```

### 7) Active IP blocks

```sql
SELECT
  ip,
  blocked_until,
  reason,
  updated_at
FROM ip_blocks
WHERE blocked_until >= (NOW() AT TIME ZONE 'UTC')::text
ORDER BY blocked_until DESC;
```

### 8) MFA adoption and activity (24 hours)

```sql
SELECT
  event_type,
  COUNT(*) AS count
FROM security_events
WHERE event_type IN ('auth_mfa_setup', 'auth_mfa_enabled', 'auth_mfa_disabled')
  AND created_at >= (NOW() AT TIME ZONE 'UTC' - INTERVAL '24 hours')::text
GROUP BY event_type
ORDER BY count DESC;
```

## Incident Response Checklist

1. Identify abusive IPs and accounts from alert query output.
2. Temporarily lower rate-limit thresholds via env vars if attack is ongoing.
3. Revoke active sessions for impacted users:

```sql
DELETE FROM sessions
WHERE user_id = '<user_id>';
```

4. Preserve evidence (`security_events`, job IDs, timestamps) before cleanup.
5. Rotate keys/secrets if exposure is suspected.
