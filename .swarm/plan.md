# Refactoring Mission Plan
Swarm: mega
Phase: 3 | Updated: 2026-02-04

## Phase 1: Discovery & Audit [COMPLETE]
- [x] 1.1: Backend Deep Scan (Security, Performance, Architecture)
- [x] 1.2: Frontend Deep Scan (Performance, Re-renders, Bundle size, Anti-patterns)
- [x] 1.3: Generate Refactoring Roadmap

## Phase 2: Refactoring - Batch 1: Critical Security (Backend) [COMPLETE]
- [x] 2.1: Fix Path Traversal in File Upload (backend/app/api/routes/documents.py)
- [x] 2.2: Add File Size & Type Validation (backend/app/api/routes/documents.py)
- [x] 2.3: Fix Unsafe Query Parameter in Memory Search (backend/app/services/memory_store.py)
- [x] 2.4: Remove Duplicate Health Check Endpoints (backend/app/main.py)
- [x] 2.5: Fix Global HTTP Client Leak (backend/app/services/llm_client.py)

## Phase 3: Refactoring - Batch 2: Core Architecture (Backend) [IN PROGRESS]
- [ ] 3.1: Implement Dependency Injection for Services (Remove Global Singletons)
- [ ] 3.2: Implement Database Connection Pooling (Remove new connection per request)
- [ ] 3.3: Standardize Async/Sync boundaries (Fix blocking I/O)

## Phase 4: Refactoring - Batch 3: Frontend Modernization [PENDING]
- [ ] 4.1: Decompose Monolithic App.tsx into Page Components
- [ ] 4.2: Extract Custom Hooks (useDebounce, useDocuments, etc.)
- [ ] 4.3: optimize Re-renders (Memoization for Markdown & Lists)
- [ ] 4.4: Fix Chat History Loading Loop

## Phase 5: Validation & Cleanup [PENDING]
- [ ] 5.1: Add Authentication/Authorization (Basic Auth or Token)
- [ ] 5.2: Comprehensive Regression Testing
- [ ] 5.3: Final Codebase Polish (Docstrings, Type Hints)
