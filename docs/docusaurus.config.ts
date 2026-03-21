import type {Config} from '@docusaurus/types';
import {themes as prismThemes} from 'prism-react-renderer';

const baseUrl = process.env.DOCUSAURUS_BASE_URL ?? '/';

const config: Config = {
  title: 'AutoAgents',
  tagline: 'Production-grade multi-agent framework docs for Rust and Python builders.',
  favicon: 'img/logo.png',
  url: 'https://liquidos-ai.github.io',
  baseUrl,
  organizationName: 'liquidos-ai',
  projectName: 'AutoAgents',
  onBrokenLinks: 'throw',
  trailingSlash: true,
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },
  themes: [],
  presets: [
    [
      'classic',
      {
        docs: {
          path: 'content',
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/liquidos-ai/AutoAgents/tree/main/docs/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],
  themeConfig: {
    image: 'img/logo.png',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'LiquidOS',
      hideOnScroll: true,
      logo: {
        alt: 'LiquidOS AutoAgents',
        src: 'img/logo.png',
        width: 24,
        height: 24,
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/quick-start/',
          label: 'Getting Started',
          position: 'left',
        },
        {
          href: 'https://docs.rs/releases/search?query=autoagents',
          label: 'docs.rs',
          position: 'left',
        },
        {
          href: 'https://github.com/liquidos-ai/AutoAgents',
          position: 'right',
          className: 'navbar-github-link',
          'aria-label': 'GitHub Repository',
        },
      ],
    },
    footer: {
      style: 'light',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Overview',
              to: '/',
            },
            {
              label: 'Quick Start',
              to: '/quick-start/',
            },
            {
              label: 'Architecture',
              to: '/architecture/',
            },
          ],
        },
        {
          title: 'Reference',
          items: [
            {
              label: 'docs.rs',
              href: 'https://docs.rs/releases/search?query=autoagents',
            },
            {
              label: 'Python Bindings',
              to: '/python-bindings/',
            },
          ],
        },
        {
          title: 'Project',
          items: [
            {
              label: 'Repository',
              href: 'https://github.com/liquidos-ai/AutoAgents',
            },
            {
              label: 'Contributing',
              href: 'https://github.com/liquidos-ai/AutoAgents/blob/main/CONTRIBUTING.md',
            },
          ],
        },
      ],
      copyright: `Copyright ${new Date().getFullYear()} <a href="https://liquidos.ai" target="_blank" rel="noopener noreferrer">LiquidOS AI</a>`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'json', 'yaml', 'toml', 'rust', 'python'],
    },
  },
};

export default config;
