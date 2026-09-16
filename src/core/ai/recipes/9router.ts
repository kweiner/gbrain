import type { Recipe } from '../types.ts';

// Cast through `unknown`: TS's `typeof fetch` includes a `preconnect` member
// this shim doesn't implement (matches azure-openai.ts, openrouter.ts).
//
// Force `stream: false` onto the request body when the caller didn't specify
// a `stream` value at all.
//
// Why: the AI SDK's non-streaming `doGenerate()` omits the `stream` key
// entirely (per the OpenAI spec, absence should mean non-streaming). 9Router's
// combo route (`chatCore.js`: `let stream = providerRequiresStreaming ? true :
// (body.stream !== false)`) defaults an absent field to STREAMING instead,
// gated back to non-streaming only for a `clientPrefersJson`-recognized
// user-agent — which the AI SDK's UA (`ai/<ver> ai-sdk/provider-utils/<ver>
// runtime/bun/<ver>`) isn't. gbrain's chat path always expects a single JSON
// object back, so an unrecognized client gets an SSE body (`data: {...}`
// chunks) where `JSON.parse` throws `Invalid JSON response` — every
// subagent-loop call against a 9Router combo failed outright until this
// shipped (confirmed 2026-09-16 against a live 9Router instance). An explicit
// `stream: true` from a real caller is left untouched; this only disambiguates
// the absent case gbrain itself always means as non-streaming.
//
// Candidate for upstreaming: this shim is 9Router-specific but structurally
// identical to a bug class any "default-to-streaming on missing `stream`"
// gateway could hit. If gbrain grows more than one such gateway target, this
// might be worth generalizing into a shared `openai-compatible` guard rather
// than a per-recipe fetch shim.
const nineRouterCompatFetch = (async (
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> => {
  if (!init?.body || typeof init.body !== 'string') return fetch(input, init);
  let body: unknown;
  try {
    body = JSON.parse(init.body);
  } catch {
    return fetch(input, init);
  }
  if (!body || typeof body !== 'object' || Array.isArray(body) || 'stream' in body) {
    return fetch(input, init);
  }
  const rewritten = JSON.stringify({ ...body, stream: false });
  const headers = new Headers(init.headers);
  headers.set('content-length', String(new TextEncoder().encode(rewritten).length));
  return fetch(input, { ...init, headers, body: rewritten });
}) as unknown as typeof fetch;

/**
 * 9Router — local/remote AI gateway (https://github.com/decolua/9router)
 * exposing an OpenAI-compatible REST surface over dozens of providers plus
 * user-defined "combos" (named model lists with fallback/round-robin/fusion
 * strategies). Point gbrain at a combo the same way any other model is
 * addressed: `9router:<combo-name>` or `9router:<provider>/<model>`.
 *
 * `id: '9router'` (not `ninerouter`) so `gbrain config get models.tier.*`
 * reads the way the product is actually named. The env vars can't follow
 * suit — POSIX/shell identifiers may not start with a digit, so
 * `NINEROUTER_BASE_URL` / `NINEROUTER_API_KEY` stay spelled out.
 *
 * Distinct from the `litellm` recipe even though both are generic
 * `openai-compatible` templates: kept separate so `gbrain config get
 * models.tier.*` reads honestly — nothing here runs the actual LiteLLM
 * proxy project.
 */
export const nineRouter: Recipe = {
  id: '9router',
  name: '9Router',
  tier: 'openai-compat',
  implementation: 'openai-compatible',
  base_url_default: 'http://localhost:20128/v1', // 9Router default (docker-compose PORT=20128)
  auth_env: {
    required: [], // NINEROUTER_API_KEY is optional (users may run 9Router with requireApiKey=false)
    optional: ['NINEROUTER_BASE_URL', 'NINEROUTER_API_KEY'],
    setup_url: 'https://github.com/decolua/9router',
  },
  touchpoints: {
    embedding: {
      // Models depend on which providers the 9Router instance has configured;
      // declare empties so the wizard prompts the user, matching litellm-proxy.ts.
      models: [],
      user_provided_models: true,
      default_dims: 0,
      trust_custom_dims: true,
      cost_per_1m_tokens_usd: undefined,
      price_last_verified: '2026-09-16',
      no_batch_cap: true,
      supports_multimodal: true,
    },
    expansion: {
      models: [],
      cost_per_1m_tokens_usd: undefined,
      price_last_verified: '2026-09-16',
    },
    chat: {
      models: [],
      supports_tools: true,
      supports_subagent_loop: true,
      // 9Router's combo route doesn't surface a cache_control passthrough;
      // treat as uncacheable like every other generic openai-compatible recipe.
      supports_prompt_cache: false,
      max_context_tokens: 200_000,
      cost_per_1m_input_usd: undefined,
      cost_per_1m_output_usd: undefined,
      price_last_verified: '2026-09-16',
    },
  },
  setup_hint: 'Run 9Router (https://github.com/decolua/9router) or point at a remote instance; set NINEROUTER_BASE_URL (include the /v1 suffix, e.g. http://localhost:20128/v1) and NINEROUTER_API_KEY if the instance requires one, then use 9router:<combo-or-provider/model>.',
  compat: { fetch: nineRouterCompatFetch },
};
