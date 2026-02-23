# KnowledgeVault UI/UX Improvement Plan
Swarm: paid
Phase: UI/UX Improvements | Updated: 2026-02-22

## Overview
Comprehensive UI/UX improvements focusing on the two highest user-touching pages:
1. **ChatPage** - Primary interaction surface
2. **DocumentsPage** - Document management and upload

**Goals:**
- Fix mobile responsiveness issues (tables, navigation, touch targets)
- Achieve WCAG 2.1 AA accessibility compliance
- Add missing chat controls and document management features
- Improve upload UX with queue management

## Phase 1: Mobile Navigation & Layout [HIGH]

### 1.1: Mobile Navigation Redesign [MEDIUM]
**File:** `frontend/src/components/layout/NavigationRail.tsx`

**Current Issues:**
- NavigationRail with 5 items is too crowded on mobile
- No mobile-specific navigation pattern

**Changes:**
- Add responsive behavior: show bottom tab bar on mobile (< md breakpoint)
- Primary 3 items (Chat, Documents, Memory) in bottom tabs
- Remaining items (Vaults, Settings) in "More" overflow drawer
- Ensure all tap targets ≥44×44px

**Dependencies:** None
**Acceptance Criteria:**
- [ ] Bottom tab bar visible on mobile
- [ ] All navigation items accessible
- [ ] Tap targets meet 44×44px minimum
- [ ] Active state visible on mobile

### 1.2: Document List Mobile Cards [MEDIUM]
**Files:**
- `frontend/src/pages/DocumentsPage.tsx`
- `frontend/src/components/documents/DocumentCard.tsx` (NEW)

**Current Issues:**
- Table requires horizontal scroll on mobile with no indicator
- Action buttons too small for touch

**Changes:**
- Create responsive grid that switches to card layout below `sm` breakpoint
- Card shows: title, status badge, size, actions dropdown
- Keep table for desktop (`hidden sm:block`)
- Add horizontal scroll indicator for desktop table

**Dependencies:** None
**Acceptance Criteria:**
- [ ] Cards display on mobile (< 640px)
- [ ] Table displays on desktop (≥ 640px)
- [ ] All document info visible on cards
- [ ] Touch-friendly action buttons (≥44×44px)

### 1.3: Chat Sources Accordion [SMALL]
**File:** `frontend/src/pages/ChatPage.tsx`

**Current Issues:**
- Sources panel takes full column, wasting vertical space on mobile

**Changes:**
- Collapse sources into Accordion on mobile
- Default collapsed state on mobile
- "Sources" toggle button with aria-expanded
- Full column layout preserved on desktop

**Dependencies:** 1.1
**Acceptance Criteria:**
- [ ] Sources collapsible on mobile
- [ ] Accordion has proper ARIA attributes
- [ ] Toggle button visible and accessible

## Phase 2: Accessibility Improvements [HIGH]

### 2.1: Chat Accessibility [MEDIUM]
**Files:**
- `frontend/src/pages/ChatPage.tsx`
- `frontend/src/components/shared/MessageContent.tsx`

**Changes:**
- Add aria-label="Message input" to textarea
- Add role="log" with aria-live="polite" to message list
- Add aria-live="assertive" for completed streaming messages
- **Install:** `react-aria-live` for reliable live region updates
- **Focus management:** Ensure new messages don't steal focus

**Dependencies:** None
**Acceptance Criteria:**
- [ ] Textarea has aria-label
- [ ] Message list is live region
- [ ] Streaming completion announced
- [ ] react-aria-live installed and working

### 2.2: Document Table Accessibility [MEDIUM]
**File:** `frontend/src/pages/DocumentsPage.tsx`

**Changes:**
- Add `<caption>` describing table purpose
- Add `scope="col"` to all header cells
- Add `scope="row"` to filename cells
- Add search input with visible label
- Add status filtering with accessible combobox

**Dependencies:** None
**Acceptance Criteria:**
- [ ] Table has caption
- [ ] Headers have scope attributes
- [ ] Search input has label
- [ ] Status filter is accessible

### 2.3: Global ARIA & Focus Management [MEDIUM]
**Files:**
- `frontend/src/components/layout/NavigationRail.tsx`
- `frontend/src/components/shared/StatusBadge.tsx`
- All new dropdown/dialog components

**Changes:**
- Add aria-current to active nav item
- Add aria-label to icon-only buttons
- Ensure status badges have descriptive text
- **Focus trapping:** All modals/drawers trap focus
- **Keyboard navigation:** All interactive elements keyboard-operable
- **Escape key:** All modals/dropdowns close on Escape
- **Color contrast:** Verify all new components meet WCAG AA (4.5:1)

**Dependencies:** None
**Acceptance Criteria:**
- [ ] Active nav item has aria-current
- [ ] All icon buttons have aria-label
- [ ] Status badges are screen-reader friendly
- [ ] Focus trapping works in all modals
- [ ] All interactive elements keyboard-accessible
- [ ] Color contrast verified (Lighthouse ≥ 95)

## Phase 3: Chat Controls & Features [MEDIUM]

### 3.1: Chat Action Menu [MEDIUM]
**Files:**
- `frontend/src/pages/ChatPage.tsx`
- `frontend/src/hooks/useChatStore.ts`
- `frontend/src/lib/api.ts`

**Changes:**
- Add "More" dropdown (DropdownMenu) at top-right of chat area
- Actions: Rename Chat, Delete Chat, Export to Text, Copy All Messages
- Add clear chat/reset button with confirmation dialog
- Extend useChatStore with chat metadata actions
- **Dropdown is responsive:** Works with both NavigationRail and bottom tab bar layouts

**Dependencies:** 1.1 (Navigation must be stable first)
**Acceptance Criteria:**
- [ ] Dropdown menu visible and functional
- [ ] Dropdown works in both navigation layouts
- [ ] Rename works and persists
- [ ] Delete has confirmation dialog
- [ ] Export generates text file download
- [ ] Clear chat has confirmation
- [ ] Focus trapping works in dropdown

### 3.2: Message Actions [SMALL]
**Files:**
- `frontend/src/components/shared/MessageActions.tsx` (NEW)
- `frontend/src/pages/ChatPage.tsx`

**Changes:**
- Add copy button to each message (user and assistant)
- Add regenerate button to assistant messages (if supported by backend)
- Buttons appear on hover/focus
- **Keyboard accessible:** Tab to focus, Enter to activate

**Dependencies:** None
**Acceptance Criteria:**
- [ ] Copy button copies message text
- [ ] Buttons accessible via keyboard
- [ ] Visual feedback on copy
- [ ] Buttons meet 44×44px touch target

### 3.3: Keyboard Shortcuts Help [SMALL]
**Files:**
- `frontend/src/components/shared/KeyboardShortcuts.tsx` (NEW)
- `frontend/src/pages/ChatPage.tsx`

**Changes:**
- Create keyboard shortcuts help dialog
- Trigger with `?` key (Shift + /)
- Show shortcuts in definition list format (`<dl>` with `<dt>`/`<dd>`)
- Add `title` attributes to shortcut-enabled elements
- **Accessible:** Dialog has proper focus trapping and ARIA labels

**Dependencies:** None
**Acceptance Criteria:**
- [ ] Dialog opens with `?` key
- [ ] Shortcuts listed clearly in definition list
- [ ] Title attributes on elements
- [ ] Focus trapped in dialog
- [ ] Closes with Escape key

## Phase 4: Document Upload Improvements [MEDIUM]

**NOTE:** Concurrent uploads (Phase 4.1) require backend API changes. This phase focuses on **frontend-only improvements** using existing sequential upload API. Backend concurrent upload support will be addressed in a separate backend phase.

### 4.1: Upload Queue Visualization [MEDIUM]
**Files:**
- `frontend/src/hooks/useUploadQueue.ts` (NEW)
- `frontend/src/pages/DocumentsPage.tsx`
- `frontend/src/components/upload/UploadQueue.tsx` (NEW)

**Changes:**
- Create upload queue state management
- Show stacked list of files with individual progress bars
- Add cancel button (✕) for each upload (cancels pending uploads only)
- Show summary toast with success/failure counts
- **Sequential upload preserved:** Files upload one at a time using existing API

**Dependencies:** None
**Acceptance Criteria:**
- [ ] Queue displays all pending uploads
- [ ] Each file has progress bar
- [ ] Cancel button cancels pending uploads (not in-progress)
- [ ] Summary toast on completion

### 4.2: Upload Status Polling [SMALL]
**File:** `frontend/src/pages/DocumentsPage.tsx`

**Changes:**
- Poll document status every 5 seconds while documents are processing
- Auto-refresh document list when status changes
- Visual indicator for "processing" vs "pending"
- Stop polling when no documents in processing state

**Dependencies:** 4.1
**Acceptance Criteria:**
- [ ] Status polling every 5s when processing documents exist
- [ ] List updates automatically
- [ ] Visual state transitions smooth
- [ ] Polling stops when idle

## Phase 5: Visual Polish [LOW]

### 5.1: Empty State Illustrations [SMALL]
**Files:**
- `frontend/src/components/shared/EmptyState.tsx` (NEW)
- Multiple page files

**Changes:**
- Create reusable EmptyState component with Lucide icon + message + CTA
- Replace icon-only empty states
- Add helpful messages and action buttons

**Dependencies:** None
**Acceptance Criteria:**
- [ ] EmptyState component created
- [ ] Used on ChatPage, DocumentsPage, MemoryPage
- [ ] Consistent styling across pages

## Dependencies Graph

```
1.1 Mobile Navigation
    ↓
1.3 Chat Sources Accordion

1.2 Document Cards
    ↓
2.2 Document Table Accessibility

2.1 Chat Accessibility
    ↓
3.1 Chat Action Menu
    ↓
3.2 Message Actions

3.3 Keyboard Shortcuts (independent)

4.1 Upload Queue
    ↓
4.2 Status Polling

5.1 Empty States (independent)
```

## Task Order

### Sprint 1 (High Impact)
1. 1.1 - Mobile Navigation
2. 1.2 - Document Cards
3. 1.3 - Chat Sources Accordion
4. 2.1 - Chat Accessibility

### Sprint 2 (Core Features)
5. 2.2 - Document Table Accessibility
6. 2.3 - Global ARIA & Focus Management
7. 3.1 - Chat Action Menu

### Sprint 3 (Upload & Polish)
8. 4.1 - Upload Queue Visualization
9. 4.2 - Status Polling
10. 3.2 - Message Actions
11. 3.3 - Keyboard Shortcuts Help

### Sprint 4 (Final Polish)
12. 5.1 - Empty States

## New Components to Create

1. `frontend/src/components/layout/MobileNavigation.tsx`
2. `frontend/src/components/documents/DocumentCard.tsx`
3. `frontend/src/components/shared/MessageActions.tsx`
4. `frontend/src/components/upload/UploadQueue.tsx`
5. `frontend/src/components/shared/EmptyState.tsx`
6. `frontend/src/components/shared/KeyboardShortcuts.tsx`
7. `frontend/src/hooks/useUploadQueue.ts`

## Dependencies to Install

```bash
npm install @radix-ui/react-accordion
npm install @radix-ui/react-dropdown-menu
npm install react-aria-live
```

## Acceptance Criteria Summary

### Phase 1
- [ ] Mobile navigation works on < md breakpoint
- [ ] Document cards display on < sm breakpoint
- [ ] Chat sources collapsible on mobile

### Phase 2
- [ ] All WCAG 2.1 AA requirements met
- [ ] Screen reader can navigate chat
- [ ] Screen reader can navigate document list

### Phase 3
- [ ] Chat actions (rename, delete, export) functional
- [ ] Message copy buttons work
- [ ] Clear chat with confirmation
- [ ] Keyboard shortcuts help accessible

### Phase 4
- [ ] Upload queue displays progress
- [ ] Upload cancellation works for pending uploads
- [ ] Status polling auto-refreshes

### Phase 5
- [ ] Empty states have illustrations with Lucide icons

## Rollback Strategy

1. All changes are additive (new components, props)
2. No breaking changes to existing APIs
3. Can disable features via feature flags if needed
4. Git revert available for any commit

## Testing Strategy

1. **Manual Testing:**
   - Mobile devices (iOS Safari, Android Chrome)
   - Desktop (Chrome, Firefox, Safari, Edge)
   - Screen readers (VoiceOver, NVDA)

2. **Automated Testing:**
   - Frontend build verification
   - Accessibility linting (axe-core)
   - Component tests for new components

## Success Metrics

- Mobile usability score ≥ 90 (Lighthouse)
- Accessibility score ≥ 95 (Lighthouse)
- Zero critical accessibility violations (axe-core)
- User can complete core tasks on mobile without horizontal scroll
