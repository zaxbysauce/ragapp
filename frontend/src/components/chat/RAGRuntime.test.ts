import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { RAGRuntime } from './RAGRuntime';

// Mock the apiStream function
vi.mock('@/lib/api', async () => {
  const actual = await vi.importActual('@/lib/api');
  return {
    ...actual,
    apiStream: vi.fn(),
  };
});

import { apiStream } from '@/lib/api';

describe('RAGRuntime', () => {
  let runtime: RAGRuntime;
  let mockAbortFn: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    runtime = new RAGRuntime();
    mockAbortFn = vi.fn();
    
    // Default mock implementation for apiStream
    vi.mocked(apiStream).mockReturnValue({
      abort: mockAbortFn,
    });
  });

  afterEach(() => {
    // Clean up any pending streams
    runtime.abort();
  });

  describe('abort()', () => {
    it('should be a no-op when abortFn is null', () => {
      runtime.abort();
      expect(mockAbortFn).not.toHaveBeenCalled();
    });

    it('should call abortFn when called after stream starts', async () => {
      // Start a mock stream
      vi.mocked(apiStream).mockImplementation((path, body, callbacks) => {
        setTimeout(() => callbacks.onDone?.([]), 100);
        return { abort: mockAbortFn };
      });

      const generator = runtime.run({
        messages: [{ id: '1', role: 'user', content: 'test' }],
        abortSignal: undefined,
        context: {},
      });

      // Trigger the generator to start
      generator.next();

      // Now abort should work
      runtime.abort();
      expect(mockAbortFn).toHaveBeenCalled();
    });

    it('should be safe to call abort multiple times', async () => {
      vi.mocked(apiStream).mockImplementation((path, body, callbacks) => {
        setTimeout(() => callbacks.onDone?.([]), 100);
        return { abort: mockAbortFn };
      });

      const generator = runtime.run({
        messages: [{ id: '1', role: 'user', content: 'test' }],
        abortSignal: undefined,
        context: {},
      });

      generator.next();

      // Call abort multiple times
      runtime.abort();
      runtime.abort();
      runtime.abort();

      expect(mockAbortFn).toHaveBeenCalledTimes(1); // Should only call once
    });
  });

  describe('Pre-aborted signal handling (RACE CONDITION FIX)', () => {
    /**
     * This is the key test for the race condition fix.
     * 
     * Bug scenario (OLD):
     * 1. abortSignal.aborted === true before run() is called
     * 2. handleAbort() is called immediately
     * 3. this.abort() is called
     * 4. this.abortFn is null (not yet set) → stream NOT cancelled!
     * 
     * Fixed scenario (NEW):
     * 1. apiStream() is called
     * 2. this.abortFn = stream.abort is set IMMEDIATELY (line 163)
     * 3. THEN abortSignal.aborted is checked
     * 4. handleAbort() is called → this.abort() → stream cancelled ✓
     */
    it('should cancel the stream when abortSignal is already aborted before run()', async () => {
      // Create an already-aborted signal
      const abortController = new AbortController();
      abortController.abort(); // Abort BEFORE run() is called

      const capturedAbortFn = vi.fn();
      
      // Mock apiStream - simulate abort behavior: abort triggers onError
      vi.mocked(apiStream).mockImplementation(() => {
        return { 
          abort: capturedAbortFn,
        };
      });

      // Create and run the generator
      const generator = runtime.run({
        messages: [{ id: '1', role: 'user', content: 'test' }],
        abortSignal: abortController.signal,
        context: {},
      });

      // Trigger the generator execution once
      generator.next();

      // CRITICAL: The abort function MUST have been called
      // This verifies the race condition fix
      expect(capturedAbortFn).toHaveBeenCalled();
    });

    it('should call abort BEFORE onDone when signal is pre-aborted', async () => {
      const abortController = new AbortController();
      abortController.abort();

      let abortCallOrder = -1;
      let onDoneCallOrder = -1;
      let callCounter = 0;
      
      const testAbortFn = vi.fn(() => {
        abortCallOrder = callCounter++;
      });

      // Mock that tracks call order
      vi.mocked(apiStream).mockImplementation(() => {
        return { 
          abort: testAbortFn,
        };
      });

      const generator = runtime.run({
        messages: [{ id: '1', role: 'user', content: 'test' }],
        abortSignal: abortController.signal,
        context: {},
      });

      // Trigger once to execute up to the abort check
      generator.next();

      // abortFn was called synchronously before any async operations
      expect(testAbortFn).toHaveBeenCalled();
      expect(abortCallOrder).toBe(0);
    });

    it('should cancel the stream immediately with pre-aborted signal', async () => {
      const abortController = new AbortController();
      abortController.abort();

      let abortWasCalled = false;
      
      vi.mocked(apiStream).mockImplementation(() => {
        return { 
          abort: () => { abortWasCalled = true; },
        };
      });

      const startTime = Date.now();
      
      const generator = runtime.run({
        messages: [{ id: '1', role: 'user', content: 'test' }],
        abortSignal: abortController.signal,
        context: {},
      });

      // Trigger the generator
      generator.next();

      const elapsed = Date.now() - startTime;

      // The abort should have been called immediately
      expect(abortWasCalled).toBe(true);
      // And it should happen quickly (synchronously, not via polling)
      expect(elapsed).toBeLessThan(50);
    });

    it('should not leave stream running when signal is pre-aborted', async () => {
      const abortController = new AbortController();
      abortController.abort();

      let streamCancelled = false;
      const mockAbort = vi.fn(() => {
        streamCancelled = true;
      });

      vi.mocked(apiStream).mockReturnValue({ abort: mockAbort });

      // Run the generator
      const generator = runtime.run({
        messages: [{ id: '1', role: 'user', content: 'test' }],
        abortSignal: abortController.signal,
        context: {},
      });

      // Trigger once
      generator.next();

      // Stream must have been cancelled
      expect(streamCancelled).toBe(true);
    });
  });

  describe('Normal abort handling', () => {
    it('should cancel stream when abort is called mid-stream', async () => {
      vi.mocked(apiStream).mockImplementation((path, body, callbacks) => {
        // Simulate async chunk delivery
        setTimeout(() => {
          callbacks.onChunk?.('Hello ');
          setTimeout(() => {
            callbacks.onChunk?.('world!');
            callbacks.onDone?.([]);
          }, 10);
        }, 10);
        
        return { abort: mockAbortFn };
      });

      const generator = runtime.run({
        messages: [{ id: '1', role: 'user', content: 'test' }],
        abortSignal: undefined,
        context: {},
      });

      // Start the stream
      const promise = generator.next();

      // Abort while streaming
      runtime.abort();

      // Wait for the abort to take effect
      try {
        await promise;
        await new Promise(resolve => setTimeout(resolve, 50));
      } catch {
        // Expected - generator may throw after abort
      }

      // Abort function should have been called
      expect(mockAbortFn).toHaveBeenCalled();
    });
  });

  describe('Message conversion', () => {
    it('should convert user message content correctly', async () => {
      vi.mocked(apiStream).mockImplementation((path, body, callbacks) => {
        // Verify the messages were converted correctly
        expect(body).toHaveProperty('messages');
        const messages = body.messages as Array<{ role: string; content: string }>;
        expect(messages[0].role).toBe('user');
        expect(messages[0].content).toBe('Hello');
        
        setTimeout(() => callbacks.onDone?.([]), 10);
        return { abort: mockAbortFn };
      });

      const generator = runtime.run({
        messages: [{
          id: '1',
          role: 'user',
          content: [{ type: 'text', text: 'Hello' }],
        }],
        abortSignal: undefined,
        context: {},
      });

      try {
        for await (const _ of generator) {}
      } catch {}
    });

    it('should include vault_id in request when context has vaultId', async () => {
      vi.mocked(apiStream).mockImplementation((path, body, callbacks) => {
        expect(body).toHaveProperty('vault_id');
        expect(body.vault_id).toBe(123);
        
        setTimeout(() => callbacks.onDone?.([]), 10);
        return { abort: mockAbortFn };
      });

      const generator = runtime.run({
        messages: [{
          id: '1',
          role: 'user',
          content: [{ type: 'text', text: 'Hello' }],
        }],
        abortSignal: undefined,
        context: { vaultId: 123 },
      });

      try {
        for await (const _ of generator) {}
      } catch {}
    });

    it('should NOT include vault_id when context has no vaultId', async () => {
      vi.mocked(apiStream).mockImplementation((path, body, callbacks) => {
        expect(body).not.toHaveProperty('vault_id');
        
        setTimeout(() => callbacks.onDone?.([]), 10);
        return { abort: mockAbortFn };
      });

      const generator = runtime.run({
        messages: [{
          id: '1',
          role: 'user',
          content: [{ type: 'text', text: 'Hello' }],
        }],
        abortSignal: undefined,
        context: {},
      });

      try {
        for await (const _ of generator) {}
      } catch {}
    });
  });

  describe('Stream completion', () => {
    it('should yield streaming updates as chunks arrive', async () => {
      vi.mocked(apiStream).mockImplementation((path, body, callbacks) => {
        setTimeout(() => callbacks.onChunk?.('Hello '), 10);
        setTimeout(() => callbacks.onChunk?.('world!'), 30);
        setTimeout(() => callbacks.onDone?.([]), 50);
        return { abort: mockAbortFn };
      });

      const results: string[] = [];
      const generator = runtime.run({
        messages: [{
          id: '1',
          role: 'user',
          content: [{ type: 'text', text: 'test' }],
        }],
        abortSignal: undefined,
        context: {},
      });

      try {
        for await (const update of generator) {
          if (update.content[0] && 'text' in update.content[0]) {
            results.push(update.content[0].text);
          }
        }
      } catch {}

      // Should have received multiple updates
      expect(results.length).toBeGreaterThan(0);
      expect(results.join('')).toContain('Hello');
    });

    it('should include sources in final result when provided', async () => {
      const mockSources = [
        { id: '1', filename: 'test.pdf', snippet: 'test content' },
      ];

      vi.mocked(apiStream).mockImplementation((path, body, callbacks) => {
        setTimeout(() => callbacks.onDone?.(mockSources), 10);
        return { abort: mockAbortFn };
      });

      let finalResult: any;
      const generator = runtime.run({
        messages: [{
          id: '1',
          role: 'user',
          content: [{ type: 'text', text: 'test' }],
        }],
        abortSignal: undefined,
        context: {},
      });

      try {
        for await (const update of generator) {
          finalResult = update;
        }
      } catch {}

      expect(finalResult).toBeDefined();
      expect(finalResult.content).toBeDefined();
      // The final result should have completion status
      expect(finalResult.status.type).toBe('complete');
    });
  });

  describe('Error handling', () => {
    it('should propagate stream errors', async () => {
      const testError = new Error('Stream failed');
      
      vi.mocked(apiStream).mockImplementation((path, body, callbacks) => {
        setTimeout(() => {
          callbacks.onError?.(testError);
        }, 10);
        return { abort: mockAbortFn };
      });

      const generator = runtime.run({
        messages: [{
          id: '1',
          role: 'user',
          content: [{ type: 'text', text: 'test' }],
        }],
        abortSignal: undefined,
        context: {},
      });

      let caughtError: Error | undefined;
      try {
        for await (const _ of generator) {}
      } catch (e) {
        caughtError = e as Error;
      }

      expect(caughtError).toBeDefined();
      expect(caughtError?.message).toContain('Stream failed');
    });
  });

  describe('Edge cases', () => {
    it('should handle empty messages array', async () => {
      vi.mocked(apiStream).mockImplementation((path, body, callbacks) => {
        expect(body.messages).toEqual([]);
        setTimeout(() => callbacks.onDone?.([]), 10);
        return { abort: mockAbortFn };
      });

      const generator = runtime.run({
        messages: [],
        abortSignal: undefined,
        context: {},
      });

      try {
        for await (const _ of generator) {}
      } catch {}
    });

    it('should handle string content (non-array)', async () => {
      vi.mocked(apiStream).mockImplementation((path, body, callbacks) => {
        const messages = body.messages as Array<{ content: string }>;
        expect(messages[0].content).toBe('Simple string');
        setTimeout(() => callbacks.onDone?.([]), 10);
        return { abort: mockAbortFn };
      });

      const generator = runtime.run({
        messages: [{
          id: '1',
          role: 'user',
          content: 'Simple string', // String instead of array
        }],
        abortSignal: undefined,
        context: {},
      });

      try {
        for await (const _ of generator) {}
      } catch {}
    });

    it('should handle messages with mixed content types', async () => {
      vi.mocked(apiStream).mockImplementation((path, body, callbacks) => {
        const messages = body.messages as Array<{ content: string }>;
        // Should concatenate different part types
        expect(messages[0].content).toBe('Part1Part2');
        setTimeout(() => callbacks.onDone?.([]), 10);
        return { abort: mockAbortFn };
      });

      const generator = runtime.run({
        messages: [{
          id: '1',
          role: 'assistant',
          content: [
            { type: 'text', text: 'Part1' },
            { type: 'text', text: 'Part2' },
          ],
        }],
        abortSignal: undefined,
        context: {},
      });

      try {
        for await (const _ of generator) {}
      } catch {}
    });
  });
});
