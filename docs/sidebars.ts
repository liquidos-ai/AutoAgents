import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'index',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/quick-start',
        'getting-started/python-bindings',
        'getting-started/first-agent',
      ],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'core-concepts/architecture',
        'core-concepts/agents',
        'core-concepts/tools',
        'core-concepts/memory',
        'core-concepts/executors',
        'core-concepts/executors_guide',
        'core-concepts/actor_agents',
        'core-concepts/advanced_patterns',
        'core-concepts/telemetry',
      ],
    },
    'serve-cli',
    {
      type: 'category',
      label: 'LLM Providers',
      items: [
        'llm-providers/overview',
        'llm-providers/optimization-pipelines',
        'llm-providers/guardrails',
      ],
    },
    {
      type: 'category',
      label: 'Developer',
      items: [
        'developer/development-setup',
        'developer/code-style',
        'developer/testing',
        'developer/workspace-architecture',
      ],
    },
  ],
};

export default sidebars;
