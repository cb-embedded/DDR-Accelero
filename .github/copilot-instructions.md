# Copilot Instructions

## Code Generation Philosophy

Generate minimal, synthetic code. Write only what is necessary.

## Code Style

- Prioritize explicit, concise naming over comments
- Use comments sparingly, only when naming cannot convey intent
- No emojis
- Split logic into focused functions
- Keep functions small and single-purpose

## Development Approach

- User steers development actively
- Generate small, pertinent code increments
- Be prudent: avoid over-anticipation
- Do not create README files unless explicitly requested
- Ask user for clarification when needed
- Minimize throwaway code and rollbacks

## Interaction Pattern

- Create TODO.md files to communicate with user
- Use TODO files to:
  - Request missing data or specifications
  - Present concise options for user decision
  - Suggest next steps without implementing them
- Keep suggestions brief and actionable

## Code Organization

- One function per logical operation
- Separate concerns into distinct modules
- Avoid premature abstraction
- Implement only requested features

## Quality Over Quantity

- Small, correct code beats large, approximate code
- Verify assumptions before expanding scope
- Wait for user guidance on ambiguous requirements
