# KnowledgeVault Email Ingestion Feature Plan
Swarm: paid
Phase: 8 [IN PROGRESS] | Updated: 2026-02-19

## Overview
Add email ingestion via IMAP polling to allow users to send documents to vaults via email.

**Architecture**: IMAP polling (30-60s) → Parse email → Extract vault from subject tag → Save attachments → Queue for processing

## Phase 8: Email Ingestion Implementation

### 8.1: Add IMAP Settings to Config [SMALL]
- Add IMAP configuration fields to `backend/app/config.py` from environment variables ONLY:
  - `IMAP_HOST` (required if enabled), `IMAP_PORT` (default 993), `IMAP_USERNAME`, `IMAP_PASSWORD` (SecretStr)
  - `IMAP_MAILBOX` (default "INBOX"), `IMAP_POLL_INTERVAL` (default 60)
  - `IMAP_ENABLED` (boolean toggle, default false)
  - `IMAP_MAX_ATTACHMENT_SIZE` (default 10MB)
  - `IMAP_ALLOWED_MIME_TYPES` (list of MIME types, default PDF/DOCX/TXT/etc)
- Password field MUST use SecretStr - never logged or exposed
- Add `.env.example` entries with placeholder values
- Validation: if IMAP_ENABLED=true, host/username/password are required

### 8.2: Create Email Service Module [MEDIUM]
New file: `backend/app/services/email_service.py`

**Responsibilities:**
- IMAP connection management (aioimaplib) with exponential backoff
- Email fetching and parsing (email module)
- Subject tag extraction: `[VaultName]` or `#vaultname` patterns (first valid tag wins, case-insensitive)
- HTML body text extraction with bleach sanitization
- Attachment validation: MIME type whitelist + size limits
- Secure temp file handling for attachments

**Key Classes:**
- `EmailIngestionService`: Main service class
  - `__init__(settings, pool, background_processor)`
  - `start_polling()`: Entry point, runs infinite poll loop
  - `stop_polling()`: Graceful shutdown
  - `_poll_once()`: Single poll iteration with error handling
  - `_connect_with_backoff()`: IMAP connection with exponential backoff (5s→15s→30s→60s max)
  - `_process_email()`: Parse single email, validate attachments
  - `_extract_vault_name()`: Regex to find vault tag, fallback to default vault
  - `_validate_attachment()`: Check MIME type whitelist + size limit
  - `_save_attachment()`: Save to secure temp dir
  - `is_healthy()`: Returns health status for monitoring

**Error Handling:**
- Connection failures: exponential backoff, max 60s interval
- Authentication errors: log error, stop polling (requires manual intervention)
- Parse errors: log and skip email, continue polling
- Vault not found: log warning, use default vault

**Dependencies:**
- `aioimaplib>=0.9.0` (add to requirements.txt)
- `bleach>=6.0.0` (add to requirements.txt)

**Security:**
- Credentials NEVER logged
- HTML sanitized with bleach (strip scripts, iframes)
- Attachment extension validated before saving
- Temp files created with restrictive permissions

### 8.3: Integrate with Document Pipeline [MEDIUM]

**Integration points:**

1. **deps.py**: Add `get_email_service()` dependency factory
   - Returns EmailIngestionService with pool + BackgroundProcessor

2. **main.py**: 
   - Add email service to app.state if `imap_enabled=True`
   - Start polling on lifespan startup (if enabled)
   - Graceful stop on shutdown

3. **Vault resolution logic**:
   - Query database for vault matching extracted name (case-insensitive)
   - If not found, use vault_id=1 (default)
   - Log warning when vault not found

4. **Attachment handling**:
   - Save attachments to temp directory
   - Call `BackgroundProcessor.enqueue(file_path, vault_id=vault_id)`
   - Clean up temp files after processing (or use tmpfile cleanup)

### 8.4: Add Email-Ingested Document Tracking [SMALL]

**New database columns** in `files` table:
- `source` TEXT NOT NULL DEFAULT 'upload': enum('upload', 'scan', 'email')
- `email_subject` TEXT (nullable): original email subject line
- `email_sender` TEXT (nullable): sender email address

**Migration details**:
- Add columns via ALTER TABLE in `backend/app/models/database.py` migrate function
- Backfill existing rows: SET source='upload' WHERE source IS NULL
- Update INSERT statements to include source field
- Update ProcessedDocument model with new fields

**Data flow**:
- Upload API: source='upload'
- File watcher: source='scan'
- Email service: source='email', populate email_subject/sender

### 8.5: Add Email Ingestion Status Endpoint [SMALL]

New endpoint: `GET /api/email/status`

**Response:**
```json
{
  "enabled": true,
  "last_poll": "2026-02-19T10:30:00Z",
  "emails_processed_today": 42,
  "emails_failed_today": 2,
  "unseen_emails": 5
}
```

**Authorization**: Admin only (reuse existing auth patterns)

### 8.6: Error Handling & Monitoring [SMALL]

- Exponential backoff for IMAP connection failures (5s → 15s → 30s → 60s max)
- Log all email processing (success/failure)
- Track metrics: emails_processed, attachments_extracted, vault_not_found_errors
- Alert on repeated connection failures

### 8.7: Tests [MEDIUM]

1. **Unit tests** for EmailIngestionService:
   - Test vault name extraction regex
   - Test attachment filtering by extension
   - Test HTML sanitization
   - Test email parsing with various multipart structures

2. **Integration tests**:
   - Mock IMAP server responses
   - Test end-to-end flow: email → attachment → queued for processing

### 8.8: Documentation [SMALL]

Update `docs/` or README with:
- Email ingestion setup guide
- How to format subject lines for vault routing
- Supported attachment types
- MailEnable IMAP configuration
- Security considerations

---

## Dependencies

```
aioimaplib>=0.9.0
bleach>=6.0.0
```

## Rollback Strategy

1. Disable via `IMAP_ENABLED=false` in .env (immediate)
2. Git revert if needed
3. Database migration is backward compatible (nullable columns)

## Acceptance Criteria

- [ ] IMAP settings loaded from environment variables ONLY
- [ ] IMAP password uses SecretStr, never logged
- [ ] Service polls IMAP every N seconds when enabled
- [ ] Connection failures use exponential backoff (5→15→30→60s max)
- [ ] Subject tags like `[Finance]` route to correct vault (case-insensitive, first match)
- [ ] Fallback to default vault when tag not found (logged warning)
- [ ] Document attachments validated by MIME type whitelist
- [ ] Non-document attachments are skipped with log
- [ ] Oversized attachments (>10MB) are rejected gracefully
- [ ] HTML body sanitized with bleach (scripts/iframes stripped)
- [ ] Status endpoint shows ingestion metrics + health status
- [ ] Database migration adds source/email columns with proper defaults
- [ ] All tests pass (existing + new)
- [ ] Documentation updated with security considerations

---

## Task Order & Dependencies

```
8.1 (config)
    ↓
8.2 (email service) ← 8.1
    ↓
8.4 (db migration) ← can run parallel with 8.2
    ↓
8.3 (integration) ← 8.2, 8.4, existing deps.py patterns
    ↓
8.5 (endpoint) ← 8.3
    ↓
8.6 (monitoring) ← 8.3
    ↓
8.7 (tests) ← all above
    ↓
8.8 (docs) ← 8.7
```
