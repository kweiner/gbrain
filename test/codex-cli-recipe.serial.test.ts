/**
 * Tests for the codex-cli LanguageModelV2 implementation that the
 * `codex-cli` recipe instantiates.
 *
 * Strategy: a POSIX shell stub at GBRAIN_CODEX_CLI_BIN emits a scripted
 * final message into the `-o <file>` argument, mirroring `codex exec`'s
 * output channel. Tests exercise the LanguageModelV2 doGenerate surface:
 * text round trip, tool-call extraction (single + multiple parallel),
 * abort semantics, context-isolation flags. No Codex CLI installation or
 * subscription required.
 *
 * Recipe registration is also smoke-tested: getRecipe('codex-cli')
 * returns a chat-only Recipe with the right model list.
 *
 * Env isolation: GBRAIN_CODEX_CLI_BIN is set per-test via withEnv(),
 * NOT in beforeAll. The provider reads the env var at spawn time so
 * withEnv's save/restore in try/finally is sufficient; no leakage to
 * sibling test files in the same bun-test process.
 */
import { describe, test, expect, beforeAll, afterAll, mock } from 'bun:test';
import { writeFileSync, chmodSync, mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import type { LanguageModelV2CallOptions } from '@ai-sdk/provider';
import { withEnv } from './helpers/with-env.ts';

const stubDir = join(tmpdir(), `codex-cli-recipe-stub-${process.pid}`);
const stubBin = join(stubDir, 'codex');
const stubResponsePath = join(stubDir, 'codex_response.txt');

/**
 * Default stub: consume stdin, require the `exec` subcommand and `--json`
 * flag, and cat the staged JSONL event stream to stdout — the same channel
 * `codex exec --json` uses for the final agent message + turn usage.
 */
function fastStubScript(): string {
  return [
    '#!/bin/sh',
    'cat > /dev/null',
    'case " $* " in',
    '  *" exec "*|"exec "*) ;;',
    '  *) echo "missing exec subcommand in argv: $*" >&2; exit 64 ;;',
    'esac',
    'case " $* " in',
    '  *" --json "*) ;;',
    '  *) echo "missing --json in argv: $*" >&2; exit 66 ;;',
    'esac',
    `cat "${stubResponsePath}"`,
  ].join('\n');
}

/** Default staged usage for tests that don't care about the exact numbers. */
const STUB_USAGE = { input_tokens: 123, cached_input_tokens: 0, cache_write_input_tokens: 0, output_tokens: 45, reasoning_output_tokens: 0 };

/** Builds the `--json` JSONL event stream a real `codex exec --json` run emits. */
function jsonEventStream(text: string, usage: typeof STUB_USAGE = STUB_USAGE): string {
  return [
    JSON.stringify({ type: 'turn.started' }),
    JSON.stringify({ type: 'item.completed', item: { id: 'item_0', type: 'agent_message', text } }),
    JSON.stringify({ type: 'turn.completed', usage }),
  ].join('\n') + '\n';
}

beforeAll(() => {
  mkdirSync(stubDir, { recursive: true });
  writeFileSync(stubBin, fastStubScript());
  chmodSync(stubBin, 0o755);
});

afterAll(() => {
  rmSync(stubDir, { recursive: true, force: true });
});

function withStubEnv<T>(fn: () => T | Promise<T>): Promise<T> {
  return withEnv({ GBRAIN_CODEX_CLI_BIN: stubBin }, fn);
}

function stageResponse(text: string): void {
  writeFileSync(stubResponsePath, jsonEventStream(text));
}

function restoreFastStub(): void {
  writeFileSync(stubBin, fastStubScript());
  chmodSync(stubBin, 0o755);
}

function userMessage(text: string): LanguageModelV2CallOptions['prompt'][number] {
  return { role: 'user', content: [{ type: 'text', text }] };
}

describe('codex-cli recipe registration', () => {
  test('getRecipe returns chat-only Recipe with the documented models', async () => {
    const { getRecipe } = await import('../src/core/ai/recipes/index.ts');
    const recipe = getRecipe('codex-cli');
    expect(recipe).toBeDefined();
    expect(recipe!.id).toBe('codex-cli');
    expect(recipe!.implementation).toBe('codex-cli');
    expect(recipe!.touchpoints.chat).toBeDefined();
    expect(recipe!.touchpoints.chat!.supports_tools).toBe(true);
    expect(recipe!.touchpoints.chat!.supports_subagent_loop).toBe(true);
    expect(recipe!.touchpoints.chat!.models).toContain('gpt-5.6-terra');
    expect(recipe!.touchpoints.embedding).toBeUndefined();
    expect(recipe!.touchpoints.expansion).toBeUndefined();
  });

  test('recipe aliases map short names to canonical model ids', async () => {
    const { getRecipe } = await import('../src/core/ai/recipes/index.ts');
    const recipe = getRecipe('codex-cli');
    expect(recipe!.aliases!['terra']).toBe('gpt-5.6-terra');
    expect(recipe!.aliases!['sol']).toBe('gpt-5.6-sol');
  });

  // Discrimination test: proves the recipe DERIVES its cost fields from
  // CANONICAL_PRICING at import time rather than carrying a hardcoded copy
  // that happens to currently match. A plain value-equality check (recipe
  // value === canonical value) would pass even on a hardcoded literal that
  // was copy-pasted from today's canonical number — exactly the "proves
  // nothing" shallow test this project's review flags. Mocking the
  // canonical lookup to a value nothing would hand-copy (99.99/199.99) and
  // asserting the recipe follows it is what actually distinguishes "sourced
  // from" from "coincidentally equal to". Fails on a reverted/hardcoded
  // recipe: mock.restore() below undoes the mock, but the reverted recipe
  // would still show 1.25/10.0 instead of the mocked value while the mock
  // is active, which is the failure this test exists to catch.
  test('cost fields track CANONICAL_PRICING even when the canonical value changes (not a hardcoded duplicate)', async () => {
    const pricingModule = await import('../src/core/model-pricing.ts');
    const original = pricingModule.CANONICAL_PRICING['openai:gpt-5.6-terra'];
    const mockRate = { input: 99.99, output: 199.99 };
    mock.module('../src/core/model-pricing.ts', () => ({
      ...pricingModule,
      canonicalLookup: (id: string) => (id === 'openai:gpt-5.6-terra' ? mockRate : pricingModule.canonicalLookup(id)),
    }));
    try {
      // Bust the module cache so the recipe re-evaluates its top-level
      // canonicalLookup() call against the mock, not a previously-cached import.
      const recipeModule = await import(`../src/core/ai/recipes/codex-cli.ts?bust=${Date.now()}`);
      expect(recipeModule.codexCli.touchpoints.chat!.cost_per_1m_input_usd).toBe(mockRate.input);
      expect(recipeModule.codexCli.touchpoints.chat!.cost_per_1m_output_usd).toBe(mockRate.output);
    } finally {
      mock.module('../src/core/model-pricing.ts', () => pricingModule);
      expect(original).toBeDefined(); // sanity: the real entry still exists post-restore
    }
  });
});

describe('codex-cli LanguageModel — text-only round trip', () => {
  test('returns a single text content block with stop finish reason and real usage from turn.completed', async () => {
    await withStubEnv(async () => {
      stageResponse('hello world');
      const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-terra');
      const result = await model.doGenerate({
        prompt: [userMessage('hi')],
      } as LanguageModelV2CallOptions);

      expect(result.finishReason).toBe('stop');
      expect(result.content).toHaveLength(1);
      expect(result.content[0]).toEqual({ type: 'text', text: 'hello world' });
      // Real token counts parsed from the --json event stream's turn.completed.
      expect(result.usage.inputTokens).toBe(STUB_USAGE.input_tokens);
      expect(result.usage.outputTokens).toBe(STUB_USAGE.output_tokens);
      expect(result.usage.totalTokens).toBe(STUB_USAGE.input_tokens + STUB_USAGE.output_tokens);
    });
  });

  test('usage is undefined when turn.completed carries no usage field', async () => {
    await withStubEnv(async () => {
      writeFileSync(
        stubResponsePath,
        [
          JSON.stringify({ type: 'turn.started' }),
          JSON.stringify({ type: 'item.completed', item: { id: 'item_0', type: 'agent_message', text: 'hi' } }),
          JSON.stringify({ type: 'turn.completed' }),
        ].join('\n') + '\n',
      );
      const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-terra');
      const result = await model.doGenerate({
        prompt: [userMessage('hi')],
      } as LanguageModelV2CallOptions);

      expect(result.usage.inputTokens).toBeUndefined();
      expect(result.usage.outputTokens).toBeUndefined();
      expect(result.usage.totalTokens).toBeUndefined();
    });
  });

  test('strips provider prefixes from the model id', async () => {
    const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
    const model = new CodexCliLanguageModel('codex-cli:gpt-5.6-terra');
    expect(model.modelId).toBe('gpt-5.6-terra');
  });
});

describe('codex-cli LanguageModel — tool use', () => {
  // Discrimination test for a real production incident: replaying a real
  // failing prompt against Codex CLI 0.146.0 showed a single turn can emit
  // TWO SEPARATE item.completed/agent_message events — a short commentary
  // preamble first, then the actual <use_tools> block as its own later
  // event. The original parseJsonEvents kept only the LAST agent_message;
  // when commentary happened to come first this worked by accident, but
  // nothing guaranteed that order. This test stages the events in the
  // OPPOSITE order (tool block first, trailing commentary second) — the
  // shape that silently discarded every real tool call in production,
  // making the subagent loop look like it had "lost access to its tools"
  // when it had actually just formatted its response across two events.
  test('finds the <use_tools> block even when a separate agent_message event follows it (real Codex CLI 0.146.0 shape)', async () => {
    await withStubEnv(async () => {
      writeFileSync(
        stubResponsePath,
        [
          JSON.stringify({ type: 'turn.started' }),
          JSON.stringify({
            type: 'item.completed',
            item: {
              id: 'item_0',
              type: 'agent_message',
              text: '<use_tools>\n[{"id": "toolu_split", "name": "search", "input": {"query": "split across events"}}]\n</use_tools>',
            },
          }),
          JSON.stringify({
            type: 'item.completed',
            item: { id: 'item_1', type: 'agent_message', text: 'Running that search now.' },
          }),
          JSON.stringify({ type: 'turn.completed', usage: STUB_USAGE }),
        ].join('\n') + '\n',
      );
      const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-terra');
      const result = await model.doGenerate({
        prompt: [userMessage('split-event tool call')],
        tools: [{ type: 'function', name: 'search', description: '', inputSchema: { type: 'object', properties: {} } }],
      } as LanguageModelV2CallOptions);

      // The discriminating assertion: on the pre-fix code this is 'stop'
      // with zero tool-calls, because only 'Running that search now.'
      // (item_1) survived — the entire <use_tools> block was discarded.
      expect(result.finishReason).toBe('tool-calls');
      const calls = result.content.filter(c => c.type === 'tool-call');
      expect(calls).toHaveLength(1);
      expect((calls[0] as { toolName: string }).toolName).toBe('search');
    });
  });

  test('parses <use_tools> block into LanguageModelV2 tool-call content', async () => {
    await withStubEnv(async () => {
      stageResponse(
        [
          'I will look up the pattern first.',
          '<use_tools>',
          '[{"id": "toolu_01ABC", "name": "search", "input": {"query": "n+1 query"}}]',
          '</use_tools>',
        ].join('\n'),
      );
      const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-terra');
      const result = await model.doGenerate({
        prompt: [userMessage('find n+1 queries')],
        tools: [
          {
            type: 'function',
            name: 'search',
            description: 'Search the brain',
            inputSchema: { type: 'object', properties: { query: { type: 'string' } } },
          },
        ],
      } as LanguageModelV2CallOptions);

      expect(result.finishReason).toBe('tool-calls');
      expect(result.content).toHaveLength(2);
      expect(result.content[0]).toMatchObject({ type: 'text', text: 'I will look up the pattern first.' });
      expect(result.content[1]).toMatchObject({
        type: 'tool-call',
        toolCallId: 'toolu_01ABC',
        toolName: 'search',
        input: '{"query":"n+1 query"}',
      });
    });
  });

  test('parses multiple parallel tool calls in a single block', async () => {
    await withStubEnv(async () => {
      stageResponse(
        [
          '<use_tools>',
          '[',
          '  {"id": "toolu_A", "name": "search", "input": {"query": "foo"}},',
          '  {"id": "toolu_B", "name": "get_page", "input": {"slug": "areas/x"}}',
          ']',
          '</use_tools>',
        ].join('\n'),
      );
      const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-terra');
      const result = await model.doGenerate({
        prompt: [userMessage('multi')],
        tools: [
          { type: 'function', name: 'search', description: 's', inputSchema: { type: 'object', properties: {} } },
          { type: 'function', name: 'get_page', description: 'g', inputSchema: { type: 'object', properties: {} } },
        ],
      } as LanguageModelV2CallOptions);

      const calls = result.content.filter(c => c.type === 'tool-call');
      expect(calls).toHaveLength(2);
      expect(calls.map(c => (c as { toolName: string }).toolName)).toEqual(['search', 'get_page']);
      expect(result.finishReason).toBe('tool-calls');
    });
  });

  test('tolerates fenced JSON inside <use_tools>', async () => {
    await withStubEnv(async () => {
      stageResponse(
        [
          '<use_tools>',
          '```json',
          '[{"id": "toolu_F", "name": "search", "input": {"q": "x"}}]',
          '```',
          '</use_tools>',
        ].join('\n'),
      );
      const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-terra');
      const result = await model.doGenerate({
        prompt: [userMessage('fenced')],
        tools: [{ type: 'function', name: 'search', description: '', inputSchema: { type: 'object', properties: {} } }],
      } as LanguageModelV2CallOptions);

      const calls = result.content.filter(c => c.type === 'tool-call');
      expect(calls).toHaveLength(1);
    });
  });

  test('synthesizes an id when the model omits it', async () => {
    await withStubEnv(async () => {
      stageResponse(
        [
          '<use_tools>',
          '[{"name": "search", "input": {"q": "x"}}]',
          '</use_tools>',
        ].join('\n'),
      );
      const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-terra');
      const result = await model.doGenerate({
        prompt: [userMessage('no id')],
        tools: [{ type: 'function', name: 'search', description: '', inputSchema: { type: 'object', properties: {} } }],
      } as LanguageModelV2CallOptions);

      const call = result.content.find(c => c.type === 'tool-call') as { toolCallId: string } | undefined;
      expect(call).toBeDefined();
      expect(call!.toolCallId).toMatch(/^toolu_codex_cli_/);
    });
  });

  test('falls back to text on malformed JSON', async () => {
    await withStubEnv(async () => {
      stageResponse(
        [
          '<use_tools>',
          'not valid json',
          '</use_tools>',
        ].join('\n'),
      );
      const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-terra');
      const result = await model.doGenerate({
        prompt: [userMessage('malformed')],
        tools: [{ type: 'function', name: 'search', description: '', inputSchema: { type: 'object', properties: {} } }],
      } as LanguageModelV2CallOptions);

      expect(result.content.filter(c => c.type === 'tool-call')).toHaveLength(0);
      expect(result.finishReason).toBe('stop');
    });
  });

  test('returns text-only stop when tools are offered but model declines to call any', async () => {
    // Real-world case: the model decides the user's request does not require
    // a tool call, ignores the use_tools protocol, and answers directly.
    // The recipe still must return clean LanguageModelV2 output so the
    // caller (gateway.toolLoop) can treat the text as the final answer
    // rather than wedge waiting for tool calls that never come.
    await withStubEnv(async () => {
      stageResponse('I do not actually need to call any tools for this. The answer is 42.');
      const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-terra');
      const result = await model.doGenerate({
        prompt: [userMessage('what is the meaning of life? you may use tools but do not need to')],
        tools: [{ type: 'function', name: 'compute', description: 'Compute things', inputSchema: { type: 'object', properties: {} } }],
      } as LanguageModelV2CallOptions);

      expect(result.content.filter(c => c.type === 'tool-call')).toHaveLength(0);
      const textBlocks = result.content.filter(c => c.type === 'text');
      expect(textBlocks).toHaveLength(1);
      expect((textBlocks[0] as { text: string }).text).toContain('42');
      expect(result.finishReason).toBe('stop');
    });
  });

  test('drops the block when the close tag is missing', async () => {
    await withStubEnv(async () => {
      stageResponse(
        [
          '<use_tools>',
          '[{"id": "toolu_X", "name": "search", "input": {}}',
        ].join('\n'),
      );
      const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-terra');
      const result = await model.doGenerate({
        prompt: [userMessage('unterminated')],
        tools: [{ type: 'function', name: 'search', description: '', inputSchema: { type: 'object', properties: {} } }],
      } as LanguageModelV2CallOptions);

      expect(result.content.filter(c => c.type === 'tool-call')).toHaveLength(0);
      expect(result.finishReason).toBe('stop');
    });
  });
});

describe('codex-cli LanguageModel — context isolation', () => {
  test('argv carries the isolation flags, stdin marker, and system section; cwd is the dedicated tmpdir', async () => {
    await withStubEnv(async () => {
      const argvLog = join(stubDir, 'argv.log');
      const cwdLog = join(stubDir, 'cwd.log');
      const stdinLog = join(stubDir, 'stdin.log');
      const recordStub = [
        '#!/bin/sh',
        `printf "%s\\n" "$@" > "${argvLog}"`,
        `pwd > "${cwdLog}"`,
        `cat > "${stdinLog}"`,
        `cat "${stubResponsePath}"`,
      ].join('\n');
      writeFileSync(stubBin, recordStub);
      chmodSync(stubBin, 0o755);
      stageResponse('ok');

      try {
        const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
        const model = new CodexCliLanguageModel('gpt-5.6-terra');
        await model.doGenerate({
          prompt: [
            { role: 'system', content: 'You are gbrain subagent.' },
            userMessage('hi'),
          ],
        } as LanguageModelV2CallOptions);

        const fs = require('node:fs');
        const argv = fs.readFileSync(argvLog, 'utf8').split('\n').filter(Boolean);
        const cwd = fs.readFileSync(cwdLog, 'utf8').trim();
        const stdin = fs.readFileSync(stdinLog, 'utf8');

        expect(argv[0]).toBe('exec');
        // Agent-isolation hardening: no user config (MCP servers, model
        // defaults), read-only sandbox, clean cwd, no repo probe.
        expect(argv).toContain('--ignore-user-config');
        expect(argv).toContain('--sandbox');
        expect(argv).toContain('read-only');
        expect(argv).toContain('--skip-git-repo-check');
        expect(argv).toContain('--json');
        expect(argv).toContain('-m');
        expect(argv).toContain('gpt-5.6-terra');
        // Prompt arrives on stdin (argv has a hard size ceiling).
        expect(argv[argv.length - 1]).toBe('-');
        expect(cwd).toMatch(/gbrain-codex-cli-cwd-\d+$/);
        // No --system-prompt flag on codex: system text leads the stdin prompt.
        expect(stdin).toContain('## System');
        expect(stdin).toContain('You are gbrain subagent.');
        expect(stdin).toContain('User: hi');
      } finally {
        restoreFastStub();
      }
    });
  });

  test('scrubs OPENAI_* credentials from the child env (subscription-only auth)', async () => {
    await withStubEnv(async () => {
      await withEnv(
        {
          OPENAI_API_KEY: 'sk-should-never-leak',
          OPENAI_BASE_URL: 'https://proxy.should.never.leak',
        },
        async () => {
          const envLog = join(stubDir, 'env.log');
          const envStub = [
            '#!/bin/sh',
            `printf "key=%s\\nbase=%s\\n" "\${OPENAI_API_KEY:-UNSET}" "\${OPENAI_BASE_URL:-UNSET}" > "${envLog}"`,
            'cat > /dev/null',
            `cat "${stubResponsePath}"`,
          ].join('\n');
          writeFileSync(stubBin, envStub);
          chmodSync(stubBin, 0o755);
          stageResponse('ok');

          try {
            const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
            const model = new CodexCliLanguageModel('gpt-5.6-terra');
            await model.doGenerate({
              prompt: [userMessage('hi')],
            } as LanguageModelV2CallOptions);

            const fs = require('node:fs');
            const seen = fs.readFileSync(envLog, 'utf8');
            expect(seen).toContain('key=UNSET');
            expect(seen).toContain('base=UNSET');
          } finally {
            restoreFastStub();
          }
        },
      );
    });
  });
});

describe('codex-cli LanguageModel — abort + error surfaces', () => {
  test('SIGTERMs the child on AbortSignal', async () => {
    await withStubEnv(async () => {
      const slowStub = [
        '#!/bin/sh',
        'cat > /dev/null',
        'sleep 30',
      ].join('\n');
      writeFileSync(stubBin, slowStub);
      chmodSync(stubBin, 0o755);
      try {
        const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
        const model = new CodexCliLanguageModel('gpt-5.6-terra');
        const ac = new AbortController();
        const promise = model.doGenerate({
          prompt: [userMessage('slow')],
          abortSignal: ac.signal,
        } as LanguageModelV2CallOptions);
        setTimeout(() => ac.abort(), 30);
        await expect(promise).rejects.toThrow(/aborted/);
      } finally {
        restoreFastStub();
      }
    });
  });

  test('rejects when the CLI exits non-zero', async () => {
    await withStubEnv(async () => {
      const failStub = [
        '#!/bin/sh',
        'cat > /dev/null',
        'echo "usage limit reached" >&2',
        'exit 1',
      ].join('\n');
      writeFileSync(stubBin, failStub);
      chmodSync(stubBin, 0o755);
      try {
        const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
        const model = new CodexCliLanguageModel('gpt-5.6-terra');
        await expect(
          model.doGenerate({ prompt: [userMessage('x')] } as LanguageModelV2CallOptions),
        ).rejects.toThrow(/codex-cli exited 1.*usage limit reached/s);
      } finally {
        restoreFastStub();
      }
    });
  });

  test('appends a cwd-reachability hint to sandbox-confinement failures only', async () => {
    await withStubEnv(async () => {
      const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
      const failWith = async (stderrLine: string): Promise<string> => {
        writeFileSync(stubBin, ['#!/bin/sh', 'cat > /dev/null', `echo "${stderrLine}" >&2`, 'exit 1'].join('\n'));
        chmodSync(stubBin, 0o755);
        const model = new CodexCliLanguageModel('gpt-5.6-terra');
        return model
          .doGenerate({ prompt: [userMessage('x')] } as LanguageModelV2CallOptions)
          .then(() => '', (e: unknown) => (e instanceof Error ? e.message : String(e)));
      };
      try {
        // The snap failure mode: an opaque errno from a cwd the child cannot open.
        const confined = await failWith('Error: No such file or directory (os error 2)');
        expect(confined).toMatch(/--- hint ---/);
        expect(confined).toMatch(/TMPDIR/);
        // Ordinary CLI failures stay clean.
        expect(await failWith('usage limit reached')).not.toMatch(/--- hint ---/);
      } finally {
        restoreFastStub();
      }
    });
  });

  test('rejects when the CLI exits 0 without emitting an agent_message event', async () => {
    await withStubEnv(async () => {
      const silentStub = [
        '#!/bin/sh',
        'cat > /dev/null',
        'exit 0',
      ].join('\n');
      writeFileSync(stubBin, silentStub);
      chmodSync(stubBin, 0o755);
      try {
        const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
        const model = new CodexCliLanguageModel('gpt-5.6-terra');
        await expect(
          model.doGenerate({ prompt: [userMessage('x')] } as LanguageModelV2CallOptions),
        ).rejects.toThrow(/produced no agent_message event/);
      } finally {
        restoreFastStub();
      }
    });
  });

  test('rejects cleanly when the codex binary is missing (no worker crash)', async () => {
    // A missing binary must surface as a rejected promise via the spawn 'error'
    // handler; the child stdin 'error' (EPIPE) handler swallows the pipe failure
    // so it never escalates to an unhandled rejection that would down the worker.
    await withEnv({ GBRAIN_CODEX_CLI_BIN: join(stubDir, 'nonexistent-codex') }, async () => {
      const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-terra');
      await expect(
        model.doGenerate({ prompt: [userMessage('x')] } as LanguageModelV2CallOptions),
      ).rejects.toThrow(/codex-cli spawn failed/);
    });
  });

  test('doStream throws not-supported', async () => {
    const { CodexCliLanguageModel } = await import('../src/core/ai/providers/codex-cli-language-model.ts');
    const model = new CodexCliLanguageModel('gpt-5.6-terra');
    await expect(model.doStream()).rejects.toThrow(/does not support streaming/);
  });
});
